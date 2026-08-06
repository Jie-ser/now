"""
Batch runner for GeoReward V2 (4RC) Best-of-N progressive elimination.

Runs a contiguous range of test cases with Wan2.2 and 4RC models loaded once.

Usage:
    python run_bon_batch_v2.py \
        --start 1 --end 10 \
        --ckpt_dir /path/to/wan2.2/checkpoints \
        --fourrc_model /path/to/4RC \
        --t5_cpu
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Wan2.2"))
sys.path.insert(0, str(ROOT / "4RC-main"))

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video
from geo_reward import ReconstructionReward, ReconRewardConfig, GeoRewardBoNProgressiveV2


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for GeoReward V2 BoN")

    # Batch range
    parser.add_argument("--start", type=int, required=True, help="First test index, inclusive")
    parser.add_argument("--end", type=int, required=True, help="Last test index, inclusive")

    # Input
    parser.add_argument("--input_dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--prompts", type=Path, default=ROOT / "batch_prompts.json")
    parser.add_argument("--name_prefix", default="test",
                        help="Input filename and prompt-key prefix.")

    # Wan2.2 model
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--size", default="480*832", choices=list(MAX_AREA_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=81,
                        help="Number of output frames (must be 4n+1).")
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=3.0,
                        help="Noise schedule shift (3.0 for 480p recommended).")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--N", type=int, default=8)

    # 4RC model
    parser.add_argument("--fourrc_model", type=str, required=True,
                        help="Path to 4RC model checkpoint directory.")
    parser.add_argument("--image_size", type=int, default=518,
                        help="4RC input resolution.")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Keyframes for reward at mid/final checkpoints.")

    # Progressive elimination
    parser.add_argument("--no_progressive", action="store_true",
                        help="Disable progressive elimination (sequential BoN).")
    parser.add_argument("--sigma_checkpoints", type=float, nargs="+",
                        default=[0.83, 0.63],
                        help="σ thresholds for checkpoints.")
    parser.add_argument("--elimination_ratio", type=float, default=0.5,
                        help="Fraction to eliminate at each checkpoint.")
    parser.add_argument("--min_survivors", type=int, default=2,
                        help="Minimum survivors at any checkpoint.")
    parser.add_argument("--score_epsilon", type=float, default=0.02,
                        help="Score indistinguishability threshold.")
    parser.add_argument("--early_max_frames", type=int, default=12,
                        help="Frames to sample at early checkpoint.")

    # V2 reward weights
    parser.add_argument("--static_weight", type=float, default=0.40,
                        help="R_static weight in total reward.")
    parser.add_argument("--dynamic_weight", type=float, default=0.40,
                        help="R_dynamic weight in total reward.")
    parser.add_argument("--motion_weight", type=float, default=0.20,
                        help="R_motion weight in total reward.")

    # Dynamic mask
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.01,
                        help="Dynamic mask threshold as fraction of scene_scale.")

    # R_static
    parser.add_argument("--tau_reproj", type=float, default=0.10,
                        help="Reprojection error temperature.")
    parser.add_argument("--occlusion_margin", type=float, default=1.05,
                        help="Occlusion depth margin multiplier.")

    # R_dynamic
    parser.add_argument("--tau_accel", type=float, default=0.05,
                        help="Acceleration penalty temperature.")
    parser.add_argument("--tau_speed", type=float, default=3.0,
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
    parser.add_argument("--tau_motion", type=float, default=0.005,
                        help="Motion gate sigmoid temperature.")

    # Valid mask
    parser.add_argument("--conf_valid_quantile", type=float, default=0.20,
                        help="Confidence quantile for valid mask.")

    # Model offloading
    parser.add_argument("--no_model_offload", action="store_true",
                        help="Disable DiT/4RC alternating offload.")

    # Output
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "geo_reward_bon_v2")

    return parser.parse_args()


def build_recon_config(args):
    return ReconRewardConfig(
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


def load_4rc_model(model_path, device="cpu"):
    """Load 4RC (Arc) model from checkpoint path."""
    from arc.arc import Arc

    logger.info(f"Loading 4RC model from: {model_path}")
    model = Arc.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def find_input_image(input_dir, name):
    """Return the supported image file for a test name."""
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = input_dir / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(
        path for path in input_dir.glob(f"{name}*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Input image not found for {name}; expected one of "
        f"{input_dir / (name + '.png')}, {input_dir / (name + '.jpg')}, "
        f"or {input_dir / (name + '.jpeg')}"
    )


def main():
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise ValueError("Require 1 <= start <= end.")

    with args.prompts.open(encoding="utf-8") as f:
        prompts = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan2.2 I2V + 4RC.")

    cfg = build_recon_config(args)
    offload_models = not args.no_model_offload

    logger.info("Loading 4RC model (initially on CPU for offload)...")
    fourrc_model = load_4rc_model(args.fourrc_model, device="cpu")
    recon_reward = ReconstructionReward(model=fourrc_model, device=str(device), cfg=cfg)

    logger.info("Loading Wan2.2 I2V once...")
    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    use_progressive = not args.no_progressive

    if use_progressive:
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

    for index in range(args.start, args.end + 1):
        name = f"{args.name_prefix}{index:04d}"
        image_path = find_input_image(args.input_dir, name)
        prompt = prompts.get(name)
        if not prompt:
            raise KeyError(f"Prompt not found for {name} in {args.prompts}")

        logger.info("===== %s (%d/%d) =====", name, index, args.end)

        case_dir = args.output_dir / f"{image_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        case_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        if use_progressive:
            best_video, result_log, best_seed = bon.generate(
                prompt=prompt,
                image=Image.open(image_path).convert("RGB"),
                N=args.N,
                frame_num=args.frame_num,
                seed_base=None,
                output_dir=str(case_dir),
                save_fn=_save_fn,
                max_area=MAX_AREA_CONFIGS[args.size],
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                offload_model=True,
            )
            elapsed = time.time() - t0

            if best_video is not None:
                best_path = str(case_dir / f"seed_{best_seed}_BEST.mp4")
                _save_fn(best_video, best_path)

            result_log["prompt"] = prompt
            result_log["image"] = str(image_path.resolve())
            result_log["total_time_sec"] = elapsed
            result_log["config"] = {
                "reward_version": "v2_4rc",
                "progressive": True,
                "fourrc_model": args.fourrc_model,
                "image_size": args.image_size,
                "offload_models": offload_models,
                "static_weight": args.static_weight,
                "dynamic_weight": args.dynamic_weight,
                "motion_weight": args.motion_weight,
            }

            with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
                json.dump(result_log, f, indent=2, ensure_ascii=False)
            logger.info("Saved results to %s (%.1fs)", case_dir, elapsed)

        else:
            from geo_reward.utils import sample_frames as _sample_frames
            indices = _sample_frames(args.frame_num, args.max_frames)

            import random
            seed_base = random.randint(0, 2**31 - 1)
            candidates = []
            rewards = []

            for i in range(args.N):
                seed = seed_base + i
                video = wan_i2v.generate(
                    input_prompt=prompt,
                    img=Image.open(image_path).convert("RGB"),
                    frame_num=args.frame_num,
                    seed=seed,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    offload_model=True,
                )
                if video is None:
                    continue

                candidates.append(video)

                if offload_models:
                    wan_i2v.model.cpu()
                    torch.cuda.empty_cache()
                    fourrc_model.cuda()

                from geo_reward.utils import wan_output_to_pil as _to_pil
                frames_pil = _to_pil(video)
                sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]
                r = recon_reward.compute_reward(sampled)
                rewards.append(r)

                if offload_models:
                    fourrc_model.cpu()
                    torch.cuda.empty_cache()
                    wan_i2v.model.cuda()

                logger.info(f"  Candidate {i+1}/{args.N} (seed={seed}): "
                            f"total={r['total']:.4f}")

            elapsed = time.time() - t0

            if not candidates:
                logger.error(f"No valid candidates for {name}, skipping.")
                continue

            best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
            ranked_indices = sorted(range(len(rewards)),
                                    key=lambda i: rewards[i]["total"], reverse=True)

            for rank, orig_idx in enumerate(ranked_indices):
                reward_val = rewards[orig_idx]["total"]
                suffix = "_BEST" if orig_idx == best_idx else ""
                filename = f"candidate_{rank+1:02d}_r{reward_val:.4f}{suffix}.mp4"
                _save_fn(candidates[orig_idx], str(case_dir / filename))

            results = {
                "prompt": prompt,
                "image": str(image_path.resolve()),
                "reward_version": "v2_4rc",
                "progressive": False,
                "best_idx": best_idx,
                "best_reward": rewards[best_idx],
                "total_time_sec": elapsed,
                "candidates": [
                    {"rank": rank + 1, "original_idx": i, "reward": rewards[i],
                     "is_best": i == best_idx}
                    for rank, i in enumerate(ranked_indices)
                ],
            }
            with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Saved results to %s (%.1fs)", case_dir, elapsed)


if __name__ == "__main__":
    main()
