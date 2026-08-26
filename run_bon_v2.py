"""
GeoReward V2 Best-of-N: Generate videos with Wan2.2 I2V and select the best
using 4RC reconstruction quality reward.

Usage:
    python run_bon_v2.py \
        --ckpt_dir /path/to/wan2.2/checkpoints \
        --fourrc_model /path/to/4RC \
        --image /path/to/first_frame.png \
        --prompt "pick up the red cube" \
        --N 8 --size 480*832

For offline scoring of pre-generated videos:
    python run_bon_v2.py --mode score --video_dir /path/to/videos/ \
        --fourrc_model /path/to/4RC
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Wan2.2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "4RC-main", "4RC-main"))

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

from geo_reward import ReconstructionReward, ReconRewardConfig, GeoRewardBoNProgressiveV2, GeoRewardBoNTreeBranching
from geo_reward.bon_pipeline import GeoRewardBoNTreeBranchingGuided
from geo_reward.utils import wan_output_to_pil, sample_frames
from geo_reward.guidance import GeometricGuidance


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GeoReward V2 (4RC) Best-of-N Pipeline")

    # Mode
    parser.add_argument("--mode", type=str, default="bon", choices=["bon", "score"],
                        help="'bon': generate + select; 'score': score existing videos.")

    # Wan2.2 generation args
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Wan2.2 checkpoint directory.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the first frame image.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (action instruction).")
    parser.add_argument("--size", type=str, default="480*832",
                        choices=["720*1280", "1280*720", "480*832", "832*480"],
                        help="Output resolution.")
    parser.add_argument("--frame_num", type=int, default=81,
                        help="Number of output frames (must be 4n+1).")
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=5.0,
                        help="Noise schedule shift (3.0 for 480p recommended).")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--t5_cpu", action="store_true",
                        help="Keep T5 on CPU to save VRAM.")
    parser.add_argument("--no_offload_model", action="store_true",
                        help="Disable offloading inactive DiT model to CPU (needs more VRAM).")

    # BoN args
    parser.add_argument("--N", type=int, default=8,
                        help="Number of candidates for Best-of-N.")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="Base seed (candidates use seed_base+i).")

    # Progressive elimination args
    parser.add_argument("--no_progressive", action="store_true",
                        help="Disable progressive elimination (use sequential BoN).")
    parser.add_argument("--sigma_checkpoints", type=float, nargs="+",
                        default=[0.83, 0.63],
                        help="σ thresholds for checkpoints (default: 0.83 0.63).")
    parser.add_argument("--elimination_ratio", type=float, default=0.5,
                        help="Fraction to eliminate at each checkpoint.")
    parser.add_argument("--min_survivors", type=int, default=2,
                        help="Minimum survivors at any checkpoint.")
    parser.add_argument("--score_epsilon", type=float, default=0.02,
                        help="Score indistinguishability threshold.")
    parser.add_argument("--early_max_frames", type=int, default=12,
                        help="Frames to sample at early checkpoint.")

    # Intermediate video saving
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate checkpoint videos (default: off, only save best).")

    # Tree Branching args
    parser.add_argument("--tree_branching", action="store_true",
                        help="Enable Tree Branching acceleration (shared trunk + branch).")
    parser.add_argument("--num_trunks", type=int, default=2,
                        help="Number of trunk trajectories (default: 2).")
    parser.add_argument("--branches_per_trunk", type=int, default=4,
                        help="Branches per trunk (total N = num_trunks * branches_per_trunk).")
    parser.add_argument("--branch_sigma", type=float, default=0.90,
                        help="Target σ for branch point (default: 0.90, maps to step dynamically).")
    parser.add_argument("--branch_eta", type=float, default=0.10,
                        help="Branch diversity hyperparameter η (default: 0.10).")

    # Gradient guidance args (Phase 3)
    parser.add_argument("--guidance", action="store_true",
                        help="Enable gradient guidance during denoising (Phase 3).")
    parser.add_argument("--guidance_scale", type=float, default=0.001,
                        help="Guidance gradient scale (default: 0.001).")
    parser.add_argument("--guidance_frequency", type=int, default=5,
                        help="Apply guidance every N steps within sigma window (default: 5).")
    parser.add_argument("--guidance_sigma_min", type=float, default=0.08,
                        help="Minimum σ for guidance window (default: 0.08).")
    parser.add_argument("--guidance_sigma_max", type=float, default=0.90,
                        help="Maximum σ for guidance window (default: 0.90).")
    parser.add_argument("--guidance_frames", type=int, default=8,
                        help="Number of frames sampled for guidance (fewer = faster, default: 8).")

    # 4RC model args
    parser.add_argument("--fourrc_model", type=str, required=True,
                        help="Path to 4RC model checkpoint directory.")
    parser.add_argument("--image_size", type=int, default=518,
                        help="4RC input resolution (default: 518).")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Number of keyframes for reward at mid/final checkpoints.")

    # V2 reward weights
    parser.add_argument("--static_weight", type=float, default=0.50,
                        help="R_static weight in total reward.")
    parser.add_argument("--dynamic_weight", type=float, default=0.30,
                        help="R_dynamic weight in total reward.")
    parser.add_argument("--motion_weight", type=float, default=0.20,
                        help="R_motion weight in total reward.")

    # Dynamic mask
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.01,
                        help="Dynamic mask threshold as fraction of scene_scale.")

    # R_static
    parser.add_argument("--tau_reproj", type=float, default=0.05,
                        help="Reprojection error temperature.")
    parser.add_argument("--occlusion_margin", type=float, default=1.05,
                        help="Occlusion depth margin multiplier.")

    # R_dynamic
    parser.add_argument("--tau_accel", type=float, default=0.02,
                        help="Acceleration penalty temperature.")
    parser.add_argument("--tau_speed", type=float, default=1.5,
                        help="Extreme speed penalty temperature.")
    parser.add_argument("--max_sample_pixels", type=int, default=1000,
                        help="Max pixels for dynamic trajectory analysis.")

    # R_motion
    parser.add_argument("--tau_cam", type=float, default=0.02,
                        help="Camera acceleration temperature.")
    parser.add_argument("--tau_rot", type=float, default=0.05,
                        help="Rotation acceleration temperature.")
    parser.add_argument("--min_motion", type=float, default=0.005,
                        help="Minimum motion for gate activation.")
    parser.add_argument("--tau_motion", type=float, default=0.02,
                        help="Motion gate sigmoid temperature.")

    # Valid mask
    parser.add_argument("--conf_valid_quantile", type=float, default=0.20,
                        help="Confidence quantile for valid mask.")

    # Model offloading (DiT <-> 4RC)
    parser.add_argument("--no_model_offload", action="store_true",
                        help="Disable DiT/4RC alternating offload (needs more VRAM).")

    # Offline scoring args
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory with .pt/.mp4 videos (for --mode score).")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/geo_reward_bon_v2",
                        help="Output directory for results.")

    return parser.parse_args()


def build_recon_config(args):
    cfg = ReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        dynamic_threshold_ratio=args.dynamic_threshold_ratio,
        tau_reproj=args.tau_reproj,
        occlusion_margin=args.occlusion_margin,
        tau_accel=args.tau_accel,
        tau_speed=args.tau_speed,
        max_sample_pixels=args.max_sample_pixels,
        tau_cam=args.tau_cam,
        tau_rot=args.tau_rot,
        min_motion=args.min_motion,
        tau_motion=args.tau_motion,
        conf_valid_quantile=args.conf_valid_quantile,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )
    # Override guidance params if guidance mode is enabled
    if hasattr(args, 'guidance') and args.guidance:
        cfg.guidance_scale = args.guidance_scale
        cfg.guidance_frequency = args.guidance_frequency
        cfg.sigma_min = args.guidance_sigma_min
        cfg.sigma_max = args.guidance_sigma_max
    return cfg


def load_4rc_model(model_path, device="cuda"):
    """Load 4RC (Arc) model from checkpoint path."""
    from arc.models.arc.arc import Arc

    logger.info(f"Loading 4RC model from: {model_path}")
    model = Arc.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def _vae_setup_for_guidance(wan_i2v, vae_device, vae_device_2):
    """Return a callable that splits VAE decoder to guidance devices."""
    if vae_device is None:
        return None

    def _setup():
        if vae_device_2 is not None:
            wan_i2v.vae.split_decoder_to_devices(vae_device, vae_device_2)
        elif hasattr(wan_i2v.vae, 'model'):
            wan_i2v.vae.model.to(vae_device)
        else:
            wan_i2v.vae.to(vae_device)
        wan_i2v.vae.mean = wan_i2v.vae.mean.to(vae_device)
        wan_i2v.vae.std = wan_i2v.vae.std.to(vae_device)
        wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]

    return _setup


def _vae_restore_after_guidance(wan_i2v, vae_device_2):
    """Return a callable that restores VAE to cuda:0 for next case."""
    def _restore():
        dit_device = torch.device("cuda:0")
        if hasattr(wan_i2v.vae, 'model'):
            wan_i2v.vae.model.to(dit_device)
            wan_i2v.vae.model.decoder.split_layer_idx = None
            wan_i2v.vae.model.decoder.device_2 = None
            wan_i2v.vae.mean = wan_i2v.vae.mean.to(dit_device)
            wan_i2v.vae.std = wan_i2v.vae.std.to(dit_device)
            wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]
        else:
            wan_i2v.vae.to(dit_device)

    return _restore


def run_bon(args):
    """Full Best-of-N pipeline with V2 reward: generate candidates and select best."""
    assert args.ckpt_dir is not None, "--ckpt_dir is required for BoN mode."
    assert args.image is not None, "--image is required for BoN mode."
    assert args.prompt is not None, "--prompt is required for BoN mode."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_recon_config(args)

    logger.info("Loading 4RC model...")
    fourrc_model = load_4rc_model(args.fourrc_model, device="cpu")

    recon_reward = ReconstructionReward(model=fourrc_model, device=str(device), cfg=cfg)

    logger.info("Loading Wan2.2 I2V pipeline...")
    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    img = Image.open(args.image).convert("RGB")
    logger.info(f"Input image: {args.image} ({img.size[0]}x{img.size[1]})")
    logger.info(f"Prompt: {args.prompt}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(args.image).stem
    case_dir = os.path.join(args.output_dir, f"{image_stem}_{timestamp}")
    os.makedirs(case_dir, exist_ok=True)

    use_progressive = not args.no_progressive
    offload_models = not args.no_model_offload

    # Validate guidance compatibility
    if args.guidance and use_progressive and not args.tree_branching:
        logger.warning(
            "--guidance is incompatible with progressive elimination (without tree_branching). "
            "Gradient guidance only works with sequential BoN (--no_progressive), "
            "tree_branching, or single generation (N=1). Ignoring --guidance flag."
        )
        args.guidance = False

    # Apply tree_branching+guidance optimized defaults (user can still override via CLI)
    if args.guidance and args.tree_branching:
        if not any(a.startswith('--guidance_frequency') for a in sys.argv):
            args.guidance_frequency = 3
        if not any(a.startswith('--guidance_sigma_max') for a in sys.argv):
            args.guidance_sigma_max = 0.83

    if args.tree_branching:
        N = args.num_trunks * args.branches_per_trunk

        if args.guidance:
            # 4-GPU resident mode for Tree Branching + Guidance
            if torch.cuda.device_count() >= 4:
                vae_device = torch.device("cuda:1")
                vae_device_2 = torch.device("cuda:2")
                fourrc_device = torch.device("cuda:3")
                logger.info("4-GPU guidance: DiT cuda:0, VAE-front cuda:1, "
                            "VAE-back cuda:2, 4RC cuda:3")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            elif torch.cuda.device_count() >= 3:
                vae_device = torch.device("cuda:1")
                vae_device_2 = None
                fourrc_device = torch.device("cuda:2")
                logger.info("3-GPU guidance: DiT cuda:0, VAE cuda:1, 4RC cuda:2")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            elif torch.cuda.device_count() >= 2:
                vae_device = torch.device("cuda:1")
                vae_device_2 = None
                fourrc_device = torch.device("cuda:1")
                logger.info("2-GPU guidance: DiT cuda:0, VAE+4RC cuda:1")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            else:
                vae_device = None
                vae_device_2 = None
                fourrc_device = None

            guidance_obj = GeometricGuidance(
                model_4rc=fourrc_model,
                vae=wan_i2v.vae,
                cfg=cfg,
                guidance_frames=args.guidance_frames,
                vae_device=vae_device,
                fourrc_device=fourrc_device,
                vae_device_2=vae_device_2,
            )

            # Multi-GPU: no model offload needed
            tree_offload = (vae_device is None)

            bon = GeoRewardBoNTreeBranchingGuided(
                wan_i2v=wan_i2v,
                recon_reward=recon_reward,
                guidance=guidance_obj,
                num_trunks=args.num_trunks,
                branches_per_trunk=args.branches_per_trunk,
                branch_sigma=args.branch_sigma,
                branch_eta=args.branch_eta,
                max_frames=args.max_frames,
                sigma_checkpoints=args.sigma_checkpoints,
                elimination_ratio=args.elimination_ratio,
                min_survivors=args.min_survivors,
                score_epsilon=args.score_epsilon,
                early_max_frames=args.early_max_frames,
                offload_models=tree_offload,
                vae_setup_fn=_vae_setup_for_guidance(wan_i2v, vae_device, vae_device_2),
                vae_restore_fn=_vae_restore_after_guidance(wan_i2v, vae_device_2),
            )

            logger.info(
                f"Mode: Tree Branching + Guidance BoN V2 "
                f"(trunks={args.num_trunks}, branches={args.branches_per_trunk}, "
                f"N={N}, branch_sigma={args.branch_sigma}, eta={args.branch_eta}, "
                f"guidance_scale={args.guidance_scale}, freq={args.guidance_frequency}, "
                f"sigma=[{args.guidance_sigma_min}, {args.guidance_sigma_max}])")

        else:
            tree_offload = offload_models
            bon = GeoRewardBoNTreeBranching(
                wan_i2v=wan_i2v,
                recon_reward=recon_reward,
                num_trunks=args.num_trunks,
                branches_per_trunk=args.branches_per_trunk,
                branch_sigma=args.branch_sigma,
                branch_eta=args.branch_eta,
                max_frames=args.max_frames,
                sigma_checkpoints=args.sigma_checkpoints,
                elimination_ratio=args.elimination_ratio,
                min_survivors=args.min_survivors,
                score_epsilon=args.score_epsilon,
                early_max_frames=args.early_max_frames,
                offload_models=offload_models,
            )
            logger.info(
                f"Mode: Tree Branching BoN V2 "
                f"(trunks={args.num_trunks}, branches={args.branches_per_trunk}, "
                f"N={N}, branch_sigma={args.branch_sigma}, eta={args.branch_eta}, "
                f"σ checkpoints: {args.sigma_checkpoints}, offload: {offload_models})")

        def _save_fn(tensor, path):
            save_video(
                tensor=tensor[None],
                save_file=path,
                fps=wan_cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        t0 = time.time()
        # In multi-GPU guidance mode, don't offload DiT during denoising
        dit_offload = not args.no_offload_model
        if args.guidance and vae_device is not None:
            dit_offload = False
        best_video, result_log, best_seed = bon.generate(
            prompt=args.prompt,
            image=img,
            N=N,
            frame_num=args.frame_num,
            seed_base=args.seed_base,
            output_dir=case_dir if args.save_intermediate else None,
            save_fn=_save_fn,
            max_area=MAX_AREA_CONFIGS[args.size],
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            offload_model=dit_offload,
        )
        total_time = time.time() - t0

        if best_video is not None:
            best_path = os.path.join(case_dir, f"seed_{best_seed}_BEST.mp4")
            _save_fn(best_video, best_path)

        result_log["prompt"] = args.prompt
        result_log["image"] = os.path.abspath(args.image)
        result_log["total_time_sec"] = total_time
        result_log["config"] = {
            "reward_version": "v2_4rc",
            "tree_branching": True,
            "guidance": args.guidance,
            "num_trunks": args.num_trunks,
            "branches_per_trunk": args.branches_per_trunk,
            "branch_sigma": args.branch_sigma,
            "branch_eta": args.branch_eta,
            "sigma_checkpoints": args.sigma_checkpoints,
            "elimination_ratio": args.elimination_ratio,
            "min_survivors": args.min_survivors,
            "score_epsilon": args.score_epsilon,
            "early_max_frames": args.early_max_frames,
            "fourrc_model": args.fourrc_model,
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "offload_models": tree_offload,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
            "size": args.size,
            "frame_num": args.frame_num,
            "sampling_steps": args.sampling_steps,
            "guide_scale": args.guide_scale,
            "seed_base": args.seed_base,
        }
        if args.guidance:
            result_log["config"]["guidance_scale"] = args.guidance_scale
            result_log["config"]["guidance_frequency"] = args.guidance_frequency
            result_log["config"]["guidance_sigma_min"] = args.guidance_sigma_min
            result_log["config"]["guidance_sigma_max"] = args.guidance_sigma_max
            result_log["config"]["guidance_frames"] = args.guidance_frames

        log_path = os.path.join(case_dir, "rewards.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result_log, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {case_dir}")

        return best_video, result_log

    elif use_progressive:
        logger.info(f"Mode: Progressive Elimination BoN V2 "
                    f"(σ checkpoints: {args.sigma_checkpoints}, "
                    f"elimination_ratio: {args.elimination_ratio}, "
                    f"offload: {offload_models})")

        bon = GeoRewardBoNProgressiveV2(
            wan_i2v=wan_i2v,
            recon_reward=recon_reward,
            max_frames=args.max_frames,
            sigma_checkpoints=args.sigma_checkpoints,
            elimination_ratio=args.elimination_ratio,
            min_survivors=args.min_survivors,
            score_epsilon=args.score_epsilon,
            early_max_frames=args.early_max_frames,
            offload_models=offload_models,
        )

        def _save_fn(tensor, path):
            save_video(
                tensor=tensor[None],
                save_file=path,
                fps=wan_cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        t0 = time.time()
        best_video, result_log, best_seed = bon.generate(
            prompt=args.prompt,
            image=img,
            N=args.N,
            frame_num=args.frame_num,
            seed_base=args.seed_base,
            output_dir=case_dir if args.save_intermediate else None,
            save_fn=_save_fn,
            max_area=MAX_AREA_CONFIGS[args.size],
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            offload_model=not args.no_offload_model,
        )
        total_time = time.time() - t0

        if best_video is not None:
            best_path = os.path.join(case_dir, f"seed_{best_seed}_BEST.mp4")
            _save_fn(best_video, best_path)

        result_log["prompt"] = args.prompt
        result_log["image"] = os.path.abspath(args.image)
        result_log["total_time_sec"] = total_time
        result_log["config"] = {
            "reward_version": "v2_4rc",
            "progressive": True,
            "sigma_checkpoints": args.sigma_checkpoints,
            "elimination_ratio": args.elimination_ratio,
            "min_survivors": args.min_survivors,
            "score_epsilon": args.score_epsilon,
            "early_max_frames": args.early_max_frames,
            "fourrc_model": args.fourrc_model,
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "offload_models": offload_models,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
            "dynamic_threshold_ratio": args.dynamic_threshold_ratio,
            "tau_reproj": args.tau_reproj,
            "tau_accel": args.tau_accel,
            "tau_speed": args.tau_speed,
            "tau_cam": args.tau_cam,
            "tau_rot": args.tau_rot,
            "min_motion": args.min_motion,
            "conf_valid_quantile": args.conf_valid_quantile,
            "size": args.size,
            "frame_num": args.frame_num,
            "sampling_steps": args.sampling_steps,
            "guide_scale": args.guide_scale,
            "seed_base": args.seed_base,
        }

        log_path = os.path.join(case_dir, "rewards.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result_log, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {case_dir}")

        return best_video, result_log

    else:
        if args.guidance:
            logger.info(f"Mode: Guided Sequential BoN V2 "
                        f"(N={args.N}, guidance_scale={args.guidance_scale}, "
                        f"guidance_freq={args.guidance_frequency}, "
                        f"sigma_window=[{args.guidance_sigma_min}, {args.guidance_sigma_max}], "
                        f"offload: {offload_models})")
        else:
            logger.info("Mode: Sequential BoN V2 (no progressive elimination)")

        from geo_reward.utils import sample_frames as _sample_frames
        indices = _sample_frames(args.frame_num, args.max_frames)

        import random
        seed_base = args.seed_base if args.seed_base is not None else random.randint(0, 2**31 - 1)

        candidates = []
        rewards = []

        # Set up guidance if enabled
        guidance_obj = None
        dual_gpu_guidance = False
        tri_gpu_guidance = False
        quad_gpu_guidance = False
        if args.guidance:
            if torch.cuda.device_count() >= 4:
                # 4-GPU: DiT on cuda:0, VAE front on cuda:1, VAE back on cuda:2, 4RC on cuda:3
                quad_gpu_guidance = True
                tri_gpu_guidance = True
                dual_gpu_guidance = True
                vae_device = torch.device("cuda:1")
                vae_device_2 = torch.device("cuda:2")
                fourrc_device = torch.device("cuda:3")
                logger.info(f"  4-GPU guidance: DiT cuda:0, VAE-front cuda:1, VAE-back cuda:2, 4RC cuda:3")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            elif torch.cuda.device_count() >= 3:
                # 3-GPU: DiT on cuda:0, VAE on cuda:1, 4RC on cuda:2
                tri_gpu_guidance = True
                dual_gpu_guidance = True
                vae_device = torch.device("cuda:1")
                vae_device_2 = None
                fourrc_device = torch.device("cuda:2")
                logger.info(f"  3-GPU guidance: DiT on cuda:0, VAE on cuda:1, 4RC on cuda:2")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            elif torch.cuda.device_count() >= 2:
                # 2-GPU: DiT on cuda:0, 4RC+VAE on cuda:1
                dual_gpu_guidance = True
                vae_device = torch.device("cuda:1")
                vae_device_2 = None
                fourrc_device = torch.device("cuda:1")
                logger.info(f"  Dual-GPU guidance: DiT on cuda:0, 4RC+VAE on cuda:1")
                fourrc_model.to(fourrc_device)
                recon_reward.device = str(fourrc_device)
            else:
                vae_device = None
                vae_device_2 = None
                fourrc_device = None

            guidance_obj = GeometricGuidance(
                model_4rc=fourrc_model,
                vae=wan_i2v.vae,
                cfg=cfg,
                guidance_frames=args.guidance_frames,
                vae_device=vae_device,
                fourrc_device=fourrc_device,
                vae_device_2=vae_device_2,
            )

        for i in range(args.N):
            seed = seed_base + i

            t0 = time.time()

            if guidance_obj is not None:
                # Guided generation: use prepare_progressive + denoise_with_guidance
                # In dual-GPU mode, DiT has full GPU — no need to offload high/low models
                dit_offload = (not args.no_offload_model) and (not dual_gpu_guidance)
                state = wan_i2v.prepare_progressive(
                    input_prompt=args.prompt,
                    img=img,
                    seeds=[seed],
                    frame_num=args.frame_num,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    offload_model=dit_offload,
                )
                total_steps = len(state['timesteps'])

                if dual_gpu_guidance:
                    # Move VAE to vae_device now (after encode is done)
                    if quad_gpu_guidance:
                        # 4-GPU: split decoder across cuda:1 and cuda:2
                        wan_i2v.vae.split_decoder_to_devices(vae_device, vae_device_2)
                        wan_i2v.vae.mean = wan_i2v.vae.mean.to(vae_device)
                        wan_i2v.vae.std = wan_i2v.vae.std.to(vae_device)
                        wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]
                    elif hasattr(wan_i2v.vae, 'model'):
                        wan_i2v.vae.model.to(vae_device)
                        wan_i2v.vae.mean = wan_i2v.vae.mean.to(vae_device)
                        wan_i2v.vae.std = wan_i2v.vae.std.to(vae_device)
                        wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]
                    else:
                        wan_i2v.vae.to(vae_device)

                    # Multi-GPU: no offload/reload needed, models stay on their GPUs
                    wan_i2v.denoise_candidates_with_guidance(
                        state, [0], 0, total_steps,
                        guidance=guidance_obj,
                        guidance_offload_dit=None,
                        guidance_reload_dit=None,
                    )
                else:
                    # Single-GPU fallback: offload/reload between DiT and 4RC+VAE
                    def _guidance_offload():
                        wan_i2v.low_noise_model.cpu()
                        wan_i2v.high_noise_model.cpu()
                        torch.cuda.empty_cache()
                        fourrc_model.cuda()
                        if hasattr(wan_i2v.vae, 'model'):
                            wan_i2v.vae.model.cuda()
                        else:
                            wan_i2v.vae.cuda()

                    def _guidance_reload():
                        fourrc_model.cpu()
                        if hasattr(wan_i2v.vae, 'model'):
                            wan_i2v.vae.model.cpu()
                        else:
                            wan_i2v.vae.cpu()
                        torch.cuda.empty_cache()
                        device = wan_i2v.device
                        if next(wan_i2v.low_noise_model.parameters()).device.type != device.type:
                            wan_i2v.low_noise_model.to(device)
                        if next(wan_i2v.high_noise_model.parameters()).device.type != device.type:
                            wan_i2v.high_noise_model.to(device)

                    wan_i2v.denoise_candidates_with_guidance(
                        state, [0], 0, total_steps,
                        guidance=guidance_obj,
                        guidance_offload_dit=_guidance_offload,
                        guidance_reload_dit=_guidance_reload,
                    )

                # Final VAE decode: in multi-GPU mode, VAE is on vae_device
                latent_for_decode = state['candidates'][0]['latent']
                if dual_gpu_guidance:
                    latent_for_decode = latent_for_decode.to(vae_device)
                video = wan_i2v.decode_latent(latent_for_decode)

                # Move VAE back to cuda:0 for next candidate's prepare_progressive
                if dual_gpu_guidance:
                    dit_device = torch.device("cuda:0")
                    if hasattr(wan_i2v.vae, 'model'):
                        wan_i2v.vae.model.to(dit_device)
                        wan_i2v.vae.model.decoder.split_layer_idx = None
                        wan_i2v.vae.model.decoder.device_2 = None
                        wan_i2v.vae.mean = wan_i2v.vae.mean.to(dit_device)
                        wan_i2v.vae.std = wan_i2v.vae.std.to(dit_device)
                        wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]
                    else:
                        wan_i2v.vae.to(dit_device)

                wan_i2v.cleanup_progressive(
                    state, offload_model=state.get('offload_model', True))
            else:
                # Standard generation without guidance
                video = wan_i2v.generate(
                    input_prompt=args.prompt,
                    img=img,
                    frame_num=args.frame_num,
                    seed=seed,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    offload_model=not args.no_offload_model,
                )

            gen_time = time.time() - t0

            if video is None:
                logger.warning(f"Candidate {i+1}/{args.N}: generation returned None, skipping.")
                continue

            candidates.append(video)

            if offload_models and not dual_gpu_guidance:
                wan_i2v.low_noise_model.cpu()
                wan_i2v.high_noise_model.cpu()
                torch.cuda.empty_cache()
                fourrc_model.cuda()

            frames_pil = wan_output_to_pil(video)
            sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]

            t1 = time.time()
            r = recon_reward.compute_reward(sampled)
            reward_time = time.time() - t1
            rewards.append(r)

            if offload_models and not dual_gpu_guidance:
                fourrc_model.cpu()
                torch.cuda.empty_cache()

            logger.info(f"  Candidate {i+1}/{args.N} (seed={seed}): "
                        f"total={r['total']:.4f} "
                        f"(R_static={r['R_static']:.4f}, R_dynamic={r['R_dynamic']:.4f}, "
                        f"R_motion={r['R_motion']:.4f}, G_anchor={r['G_anchor']:.2f}) "
                        f"[gen={gen_time:.1f}s, reward={reward_time:.1f}s]")

        if not candidates:
            raise RuntimeError("No valid candidates generated.")

        import numpy as np
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
        logger.info(f"\n[BoN V2] Selected candidate {best_idx+1}/{len(candidates)} "
                    f"with reward {rewards[best_idx]['total']:.4f}")

        os.makedirs(case_dir, exist_ok=True)
        ranked_indices = sorted(range(len(rewards)),
                                key=lambda i: rewards[i]["total"], reverse=True)
        for rank, orig_idx in enumerate(ranked_indices):
            reward_val = rewards[orig_idx]["total"]
            suffix = "_BEST" if orig_idx == best_idx else ""
            filename = f"candidate_{rank+1:02d}_r{reward_val:.4f}{suffix}.mp4"
            video_path = os.path.join(case_dir, filename)
            save_video(
                tensor=candidates[orig_idx][None],
                save_file=video_path,
                fps=wan_cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        results = {
            "prompt": args.prompt,
            "image": os.path.abspath(args.image),
            "N": args.N,
            "best_rank": 1,
            "best_original_idx": best_idx,
            "best_reward": rewards[best_idx]["total"],
            "config": {
                "reward_version": "v2_4rc",
                "progressive": False,
                "guidance": args.guidance,
                "guidance_scale": args.guidance_scale if args.guidance else None,
                "guidance_frequency": args.guidance_frequency if args.guidance else None,
                "guidance_sigma_min": args.guidance_sigma_min if args.guidance else None,
                "guidance_sigma_max": args.guidance_sigma_max if args.guidance else None,
                "guidance_frames": args.guidance_frames if args.guidance else None,
                "fourrc_model": args.fourrc_model,
                "image_size": args.image_size,
                "max_frames": args.max_frames,
                "offload_models": offload_models,
                "static_weight": args.static_weight,
                "dynamic_weight": args.dynamic_weight,
                "motion_weight": args.motion_weight,
                "dynamic_threshold_ratio": args.dynamic_threshold_ratio,
                "tau_reproj": args.tau_reproj,
                "tau_accel": args.tau_accel,
                "tau_speed": args.tau_speed,
                "tau_cam": args.tau_cam,
                "tau_rot": args.tau_rot,
                "min_motion": args.min_motion,
                "conf_valid_quantile": args.conf_valid_quantile,
            },
            "candidates": [
                {"rank": rank + 1, "original_idx": i, "reward": rewards[i],
                 "is_best": i == best_idx}
                for rank, i in enumerate(ranked_indices)
            ],
        }
        log_path = os.path.join(case_dir, "rewards.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {case_dir}")

        return candidates[best_idx], rewards


def run_score(args):
    """Score pre-generated videos offline with V2 reward."""
    assert args.video_dir is not None, "--video_dir is required for score mode."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_recon_config(args)

    fourrc_model = load_4rc_model(args.fourrc_model, device=str(device))
    recon_reward = ReconstructionReward(model=fourrc_model, device=str(device), cfg=cfg)

    video_dir = Path(args.video_dir)
    pt_files = sorted(video_dir.glob("*.pt"))
    mp4_files = sorted(video_dir.glob("*.mp4"))
    all_files = pt_files + mp4_files

    if not all_files:
        logger.error(f"No .pt or .mp4 files found in {args.video_dir}")
        return

    logger.info(f"Found {len(all_files)} videos to score.")

    indices = sample_frames(args.frame_num, args.max_frames)
    rewards = []

    for i, vf in enumerate(all_files):
        if vf.suffix == ".pt":
            tensor = torch.load(vf, map_location="cpu")
            frames_pil = wan_output_to_pil(tensor)
        else:
            import imageio.v3 as iio
            try:
                frames_np = iio.imread(str(vf), plugin="pyav")
            except ImportError:
                frames_np = iio.imread(str(vf), plugin="FFMPEG")
            frames_pil = [Image.fromarray(f) for f in frames_np]

        sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]
        r = recon_reward.compute_reward(sampled)
        rewards.append(r)

        logger.info(f"  [{i+1}/{len(all_files)}] {vf.name}: total={r['total']:.4f} "
                    f"(R_static={r['R_static']:.4f}, R_dynamic={r['R_dynamic']:.4f}, "
                    f"R_motion={r['R_motion']:.4f}, G_anchor={r['G_anchor']:.2f})")

    best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
    logger.info(f"\nBest video: {all_files[best_idx].name} "
                f"(reward={rewards[best_idx]['total']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "video_dir": str(args.video_dir),
        "reward_version": "v2_4rc",
        "scores": [{"file": f.name, **r} for f, r in zip(all_files, rewards)],
        "best_file": all_files[best_idx].name,
        "best_reward": rewards[best_idx],
        "config": {
            "fourrc_model": args.fourrc_model,
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
            "dynamic_threshold_ratio": args.dynamic_threshold_ratio,
            "tau_reproj": args.tau_reproj,
            "tau_accel": args.tau_accel,
            "tau_speed": args.tau_speed,
            "tau_cam": args.tau_cam,
            "tau_rot": args.tau_rot,
            "min_motion": args.min_motion,
            "conf_valid_quantile": args.conf_valid_quantile,
        },
    }
    log_path = os.path.join(
        args.output_dir, f"scores_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Scores saved to: {log_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "bon":
        run_bon(args)
    elif args.mode == "score":
        run_score(args)
