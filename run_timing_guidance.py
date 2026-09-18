"""
All-in-one timing benchmark for gradient guidance experiments.

Loads models ONCE, runs warmup, then sequentially runs all experiments.
Requires 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_timing_guidance.py
"""

import os
import sys
import json
import time
import torch
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Wan2.2"))

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
from geo_reward import ReconstructionReward, ReconRewardConfig
from geo_reward.bon_pipeline import (
    GeoRewardBoNTreeBranching,
    GeoRewardBoNTreeBranchingGuided,
)
from geo_reward.utils import wan_output_to_pil, sample_frames
from geo_reward.guidance import GeometricGuidance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ======================== CONFIG ========================
CKPT_DIR = "/pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B"
FOURRC_MODEL = "/pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC"
IMAGE_PATH = "/pfs/mayuema/spj/now/inputs/jishi/test_zhenji0014.jpg"
PROMPT = (
    "Use the input image as the exact first frame. A brown drawer box is in "
    "the right foreground with the upper drawer partly open. The left black "
    "robot arm starts from its current pose on the left platform, extends "
    "toward the open drawer front, pushes the drawer inward smoothly until "
    "fully closed and flush, then lifts slightly and returns to its initial "
    "standby area. The right manipulator stays in the background without "
    "interaction. Camera must stay locked and static: no zoom, no pan, no "
    "tilt, no dolly, no camera shift."
)
OUTPUT_BASE = "/pfs/mayuema/spj/now/outputs/jishi/Treebranch_guidance"
SIZE = "480*832"
SAMPLE_SHIFT = 5.0
FRAME_NUM = 81
SAMPLING_STEPS = 40
GUIDE_SCALE = 5.0

# Guidance params
GUIDANCE_SCALE = 0.001
GUIDANCE_FREQUENCY = 5
GUIDANCE_SIGMA_MIN = 0.08
GUIDANCE_SIGMA_MAX = 0.90
GUIDANCE_FRAMES = 8

# Tree Branching guidance uses tighter sigma_max
TREE_GUIDANCE_FREQUENCY = 3
TREE_GUIDANCE_SIGMA_MAX = 0.83

# Reward params
SIGMA_CHECKPOINTS = [0.83, 0.63]
BRANCH_SIGMA = 0.90
BRANCH_ETA = 0.10
MAX_FRAMES = 20
EARLY_MAX_FRAMES = 12
IMAGE_SIZE = 518
# ========================================================


def make_save_fn(wan_cfg):
    def _save_fn(tensor, path):
        save_video(
            tensor=tensor[None],
            save_file=path,
            fps=wan_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    return _save_fn


def build_cfg(guidance_sigma_max=GUIDANCE_SIGMA_MAX):
    return ReconRewardConfig(
        static_weight=0.50,
        dynamic_weight=0.30,
        motion_weight=0.20,
        tau_reproj=0.05,
        tau_accel=0.02,
        tau_speed=1.5,
        tau_cam=0.02,
        tau_rot=0.05,
        min_motion=0.005,
        conf_valid_quantile=0.20,
        max_frames=MAX_FRAMES,
        image_size=IMAGE_SIZE,
        guidance_scale=GUIDANCE_SCALE,
        guidance_frequency=GUIDANCE_FREQUENCY,
        sigma_min=GUIDANCE_SIGMA_MIN,
        sigma_max=guidance_sigma_max,
    )


def vae_setup(wan_i2v, vae_device, vae_device_2):
    if vae_device_2 is not None:
        wan_i2v.vae.split_decoder_to_devices(vae_device, vae_device_2)
    elif hasattr(wan_i2v.vae, 'model'):
        wan_i2v.vae.model.to(vae_device)
    else:
        wan_i2v.vae.to(vae_device)
    wan_i2v.vae.mean = wan_i2v.vae.mean.to(vae_device)
    wan_i2v.vae.std = wan_i2v.vae.std.to(vae_device)
    wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]


def vae_restore(wan_i2v):
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


def run_sequential_guidance(wan_i2v, recon_reward, fourrc_model, guidance_obj,
                            img, N, output_dir, wan_cfg, indices,
                            vae_device, vae_device_2, quad_gpu):
    """Run no_progressive + guidance for N candidates, return timing + results."""
    os.makedirs(output_dir, exist_ok=True)
    save_fn = make_save_fn(wan_cfg)
    seed_base = torch.randint(0, 2**31, (1,)).item()

    candidates = []
    rewards = []
    candidate_timings = []
    t_total_start = time.time()

    for i in range(N):
        seed = seed_base + i
        t0 = time.time()

        state = wan_i2v.prepare_progressive(
            input_prompt=PROMPT,
            img=img,
            seeds=[seed],
            frame_num=FRAME_NUM,
            max_area=MAX_AREA_CONFIGS[SIZE],
            shift=SAMPLE_SHIFT,
            sample_solver="unipc",
            sampling_steps=SAMPLING_STEPS,
            guide_scale=GUIDE_SCALE,
            offload_model=False,
        )
        total_steps = len(state['timesteps'])

        vae_setup(wan_i2v, vae_device, vae_device_2)

        wan_i2v.denoise_candidates_with_guidance(
            state, [0], 0, total_steps,
            guidance=guidance_obj,
            guidance_offload_dit=None,
            guidance_reload_dit=None,
        )

        latent = state['candidates'][0]['latent'].to(vae_device)
        video = wan_i2v.decode_latent(latent)

        vae_restore(wan_i2v)
        wan_i2v.cleanup_progressive(
            state, offload_model=state.get('offload_model', True))

        gen_time = time.time() - t0

        if video is None:
            logger.warning(f"Candidate {i+1}/{N}: returned None, skipping.")
            continue

        candidates.append(video)

        t1 = time.time()
        frames_pil = wan_output_to_pil(video)
        sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]
        r = recon_reward.compute_reward(sampled)
        reward_time = time.time() - t1
        rewards.append(r)

        candidate_timings.append({
            "seed": seed,
            "generate_sec": gen_time,
            "reward_sec": reward_time,
        })

        logger.info(
            f"  Candidate {i+1}/{N} (seed={seed}): total={r['total']:.4f} "
            f"[gen={gen_time:.1f}s, reward={reward_time:.1f}s]")

    total_elapsed = time.time() - t_total_start

    if not candidates:
        logger.error("No valid candidates generated.")
        return

    best_idx = max(range(len(rewards)), key=lambda j: rewards[j]["total"])
    ranked = sorted(range(len(rewards)),
                    key=lambda j: rewards[j]["total"], reverse=True)

    for rank, orig_idx in enumerate(ranked):
        rv = rewards[orig_idx]["total"]
        suffix = "_BEST" if orig_idx == best_idx else ""
        fname = f"candidate_{rank+1:02d}_r{rv:.4f}{suffix}.mp4"
        save_fn(candidates[orig_idx], os.path.join(output_dir, fname))

    results = {
        "mode": "sequential_guidance",
        "prompt": PROMPT,
        "image": IMAGE_PATH,
        "N": N,
        "best_idx": best_idx,
        "best_reward": rewards[best_idx]["total"],
        "config": {
            "guidance": True,
            "guidance_scale": GUIDANCE_SCALE,
            "guidance_frequency": GUIDANCE_FREQUENCY,
            "guidance_sigma_min": GUIDANCE_SIGMA_MIN,
            "guidance_sigma_max": GUIDANCE_SIGMA_MAX,
            "guidance_frames": GUIDANCE_FRAMES,
        },
        "candidates": [
            {"rank": rk + 1, "original_idx": j, "reward": rewards[j],
             "is_best": j == best_idx}
            for rk, j in enumerate(ranked)
        ],
        "timing": {
            "total_sec": total_elapsed,
            "per_candidate": candidate_timings,
            "avg_generate_sec": (
                sum(c["generate_sec"] for c in candidate_timings)
                / len(candidate_timings) if candidate_timings else 0.0
            ),
            "avg_reward_sec": (
                sum(c["reward_sec"] for c in candidate_timings)
                / len(candidate_timings) if candidate_timings else 0.0
            ),
        },
    }

    with open(os.path.join(output_dir, "rewards.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {output_dir} ({total_elapsed:.1f}s)")


def run_tree_guidance(wan_i2v, recon_reward, fourrc_model, guidance_obj,
                      img, num_trunks, branches_per_trunk,
                      output_dir, wan_cfg,
                      vae_device, vae_device_2):
    """Run TreeBranching + guidance, return timing + results."""
    os.makedirs(output_dir, exist_ok=True)
    save_fn = make_save_fn(wan_cfg)

    bon = GeoRewardBoNTreeBranchingGuided(
        wan_i2v=wan_i2v,
        recon_reward=recon_reward,
        guidance=guidance_obj,
        num_trunks=num_trunks,
        branches_per_trunk=branches_per_trunk,
        branch_sigma=BRANCH_SIGMA,
        branch_eta=BRANCH_ETA,
        max_frames=MAX_FRAMES,
        sigma_checkpoints=SIGMA_CHECKPOINTS,
        elimination_ratio=0.5,
        min_survivors=2,
        score_epsilon=0.02,
        early_max_frames=EARLY_MAX_FRAMES,
        offload_models=False,
        vae_setup_fn=lambda: vae_setup(wan_i2v, vae_device, vae_device_2),
        vae_restore_fn=lambda: vae_restore(wan_i2v),
    )

    N = num_trunks * branches_per_trunk
    t0 = time.time()
    best_video, result_log, best_seed = bon.generate(
        prompt=PROMPT,
        image=img,
        N=N,
        frame_num=FRAME_NUM,
        seed_base=None,
        output_dir=output_dir,
        save_fn=save_fn,
        max_area=MAX_AREA_CONFIGS[SIZE],
        shift=SAMPLE_SHIFT,
        sample_solver="unipc",
        sampling_steps=SAMPLING_STEPS,
        guide_scale=GUIDE_SCALE,
        offload_model=False,
    )
    total_time = time.time() - t0

    if best_video is not None:
        save_fn(best_video, os.path.join(output_dir, f"seed_{best_seed}_BEST.mp4"))

    result_log["prompt"] = PROMPT
    result_log["image"] = IMAGE_PATH
    result_log["total_time_sec"] = total_time
    result_log["config"] = {
        "tree_branching": True,
        "guidance": True,
        "num_trunks": num_trunks,
        "branches_per_trunk": branches_per_trunk,
        "branch_sigma": BRANCH_SIGMA,
        "branch_eta": BRANCH_ETA,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_frequency": TREE_GUIDANCE_FREQUENCY,
        "guidance_sigma_min": GUIDANCE_SIGMA_MIN,
        "guidance_sigma_max": TREE_GUIDANCE_SIGMA_MAX,
        "guidance_frames": GUIDANCE_FRAMES,
        "sigma_checkpoints": SIGMA_CHECKPOINTS,
    }

    with open(os.path.join(output_dir, "rewards.json"), "w", encoding="utf-8") as f:
        json.dump(result_log, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {output_dir} ({total_time:.1f}s)")


def main():
    assert torch.cuda.device_count() >= 4, \
        f"Need 4 GPUs, found {torch.cuda.device_count()}"

    vae_device = torch.device("cuda:1")
    vae_device_2 = torch.device("cuda:2")
    fourrc_device = torch.device("cuda:3")

    # ========== Load models once ==========
    logger.info("=" * 60)
    logger.info("Loading models (one-time)...")
    logger.info("=" * 60)

    from arc.models.arc.arc import Arc
    fourrc_model = Arc.from_pretrained(FOURRC_MODEL)
    fourrc_model = fourrc_model.to(fourrc_device).eval()

    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=0,
        rank=0,
        t5_cpu=True,
    )

    cfg_seq = build_cfg(guidance_sigma_max=GUIDANCE_SIGMA_MAX)
    recon_reward_seq = ReconstructionReward(
        model=fourrc_model, device=str(fourrc_device), cfg=cfg_seq)

    cfg_tree = build_cfg(guidance_sigma_max=TREE_GUIDANCE_SIGMA_MAX)
    recon_reward_tree = ReconstructionReward(
        model=fourrc_model, device=str(fourrc_device), cfg=cfg_tree)

    guidance_seq = GeometricGuidance(
        model_4rc=fourrc_model,
        vae=wan_i2v.vae,
        cfg=cfg_seq,
        guidance_frames=GUIDANCE_FRAMES,
        vae_device=vae_device,
        fourrc_device=fourrc_device,
        vae_device_2=vae_device_2,
    )

    guidance_tree = GeometricGuidance(
        model_4rc=fourrc_model,
        vae=wan_i2v.vae,
        cfg=cfg_tree,
        guidance_frames=GUIDANCE_FRAMES,
        vae_device=vae_device,
        fourrc_device=fourrc_device,
        vae_device_2=vae_device_2,
    )

    img = Image.open(IMAGE_PATH).convert("RGB")
    indices = sample_frames(FRAME_NUM, MAX_FRAMES)
    logger.info(f"Image: {IMAGE_PATH} ({img.size[0]}x{img.size[1]})")
    logger.info(f"4-GPU layout: DiT cuda:0, VAE-front cuda:1, "
                f"VAE-back cuda:2, 4RC cuda:3")
    logger.info("Models loaded.\n")

    # ========== Warmup ==========
    logger.info("=" * 60)
    logger.info("WARMUP: sequential guidance N=1 (result discarded)")
    logger.info("=" * 60)
    warmup_dir = os.path.join(OUTPUT_BASE, "_warmup")
    run_sequential_guidance(
        wan_i2v, recon_reward_seq, fourrc_model, guidance_seq,
        img, N=1, output_dir=warmup_dir, wan_cfg=wan_cfg, indices=indices,
        vae_device=vae_device, vae_device_2=vae_device_2, quad_gpu=True,
    )
    logger.info("Warmup complete.\n")

    # ========== Experiment list ==========
    experiments = [
        # (name, type, params)
        ("seq_guidance_N2", "sequential", {"N": 2}),
        ("seq_guidance_N4", "sequential", {"N": 4}),
        ("tree_guidance_N4_2x2", "tree", {"num_trunks": 2, "branches_per_trunk": 2}),
        ("tree_guidance_N8_2x4", "tree", {"num_trunks": 2, "branches_per_trunk": 4}),
        ("tree_guidance_N16_2x8", "tree", {"num_trunks": 2, "branches_per_trunk": 8}),
    ]

    for exp_name, exp_type, params in experiments:
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT: {exp_name}")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(OUTPUT_BASE, f"{exp_name}_{timestamp}")

        if exp_type == "sequential":
            run_sequential_guidance(
                wan_i2v, recon_reward_seq, fourrc_model, guidance_seq,
                img, N=params["N"], output_dir=exp_dir,
                wan_cfg=wan_cfg, indices=indices,
                vae_device=vae_device, vae_device_2=vae_device_2,
                quad_gpu=True,
            )
        elif exp_type == "tree":
            run_tree_guidance(
                wan_i2v, recon_reward_tree, fourrc_model, guidance_tree,
                img,
                num_trunks=params["num_trunks"],
                branches_per_trunk=params["branches_per_trunk"],
                output_dir=exp_dir, wan_cfg=wan_cfg,
                vae_device=vae_device, vae_device_2=vae_device_2,
            )

        logger.info(f"Done: {exp_name}\n")

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
