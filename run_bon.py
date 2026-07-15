"""
GeoReward Best-of-N: Generate videos with Wan2.2 I2V and select the best
using DA3 geometry-based reward.

Usage:
    python run_bon.py \
        --ckpt_dir /path/to/wan2.2/checkpoints \
        --image /path/to/first_frame.png \
        --prompt "pick up the red cube and place it on the left" \
        --N 8 \
        --size 480*832

For offline scoring of pre-generated videos:
    python run_bon.py --mode score --video_dir /path/to/videos/
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

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

from geo_reward import DA3GeoReward, GeoRewardBoN
from geo_reward.utils import wan_output_to_da3_input, sample_frames


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GeoReward Best-of-N Pipeline")

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
    parser.add_argument("--offload_model", action="store_true", default=True,
                        help="Offload inactive DiT model to CPU.")

    # BoN args
    parser.add_argument("--N", type=int, default=8,
                        help="Number of candidates for Best-of-N.")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="Base seed (candidates use seed_base+i).")

    # DA3 reward args
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="DA3 model name on HuggingFace Hub or local path.")
    parser.add_argument("--process_res", type=int, default=504,
                        help="DA3 processing resolution.")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Number of keyframes to sample for reward.")
    parser.add_argument("--reward_stride", type=int, default=2,
                        help="Frame stride for projection consistency.")

    # Offline scoring args
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory with pre-generated .pt video tensors (for --mode score).")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/geo_reward_bon",
                        help="Output directory for results.")

    return parser.parse_args()


def run_bon(args):
    """Full Best-of-N pipeline: generate candidates and select best."""
    assert args.ckpt_dir is not None, "--ckpt_dir is required for BoN mode."
    assert args.image is not None, "--image is required for BoN mode."
    assert args.prompt is not None, "--prompt is required for BoN mode."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DA3 reward
    logger.info(f"Loading DA3 model: {args.da3_model}")
    da3_reward = DA3GeoReward(
        model_name=args.da3_model,
        device=str(device),
        process_res=args.process_res,
    )

    # Initialize Wan2.2 I2V
    logger.info("Loading Wan2.2 I2V pipeline...")
    cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    # Load input image
    img = Image.open(args.image).convert("RGB")
    logger.info(f"Input image: {args.image} ({img.size[0]}x{img.size[1]})")
    logger.info(f"Prompt: {args.prompt}")

    # Create BoN pipeline
    bon = GeoRewardBoN(
        wan_i2v=wan_i2v,
        da3_reward=da3_reward,
        max_frames=args.max_frames,
    )

    # Generate and select
    t0 = time.time()
    all_candidates, all_rewards, best_idx = bon.generate(
        prompt=args.prompt,
        image=img,
        N=args.N,
        frame_num=args.frame_num,
        seed_base=args.seed_base,
        reward_stride=args.reward_stride,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        offload_model=args.offload_model,
    )
    total_time = time.time() - t0

    # Build case folder: image_stem + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(args.image).stem
    case_dir = os.path.join(args.output_dir, f"{image_stem}_{timestamp}")
    os.makedirs(case_dir, exist_ok=True)

    # Rank candidates by reward (descending)
    ranked_indices = sorted(range(len(all_rewards)),
                            key=lambda i: all_rewards[i]["total"], reverse=True)

    # Save all candidate videos
    for rank, orig_idx in enumerate(ranked_indices):
        reward_val = all_rewards[orig_idx]["total"]
        suffix = "_BEST" if orig_idx == best_idx else ""
        filename = f"candidate_{rank+1:02d}_r{reward_val:.4f}{suffix}.mp4"
        video_path = os.path.join(case_dir, filename)
        save_video(
            tensor=all_candidates[orig_idx][None],
            save_file=video_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    logger.info(f"All {len(all_candidates)} candidate videos saved to: {case_dir}")

    # Save rewards log
    results = {
        "prompt": args.prompt,
        "image": os.path.abspath(args.image),
        "N": args.N,
        "best_rank": 1,
        "best_original_idx": best_idx,
        "best_reward": all_rewards[best_idx]["total"],
        "total_time_sec": total_time,
        "candidates": [
            {
                "rank": rank + 1,
                "original_idx": orig_idx,
                "reward": all_rewards[orig_idx],
                "is_best": orig_idx == best_idx,
            }
            for rank, orig_idx in enumerate(ranked_indices)
        ],
        "config": {
            "da3_model": args.da3_model,
            "process_res": args.process_res,
            "max_frames": args.max_frames,
            "reward_stride": args.reward_stride,
            "size": args.size,
            "frame_num": args.frame_num,
            "sampling_steps": args.sampling_steps,
            "guide_scale": args.guide_scale,
            "seed_base": args.seed_base,
        }
    }
    log_path = os.path.join(case_dir, "rewards.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Rewards log saved to: {log_path}")

    return all_candidates[best_idx], all_rewards


def run_score(args):
    """Score pre-generated videos offline."""
    assert args.video_dir is not None, "--video_dir is required for score mode."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading DA3 model: {args.da3_model}")
    da3_reward = DA3GeoReward(
        model_name=args.da3_model,
        device=str(device),
        process_res=args.process_res,
    )

    from geo_reward.bon_pipeline import GeoRewardBoNOffline
    scorer = GeoRewardBoNOffline(
        da3_reward=da3_reward,
        max_frames=args.max_frames,
    )

    # Load video tensors (.pt files)
    video_dir = Path(args.video_dir)
    pt_files = sorted(video_dir.glob("*.pt"))
    if not pt_files:
        logger.error(f"No .pt files found in {args.video_dir}")
        return

    logger.info(f"Found {len(pt_files)} video tensors to score.")
    video_tensors = [torch.load(f, map_location="cpu") for f in pt_files]

    rewards = scorer.score_videos(
        video_tensors,
        frame_num=args.frame_num,
        reward_stride=args.reward_stride,
    )

    # Report results
    best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
    logger.info(f"\nBest video: {pt_files[best_idx].name} "
                f"(reward={rewards[best_idx]['total']:.4f})")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "video_dir": str(args.video_dir),
        "scores": [{"file": f.name, **r} for f, r in zip(pt_files, rewards)],
        "best_file": pt_files[best_idx].name,
        "best_reward": rewards[best_idx],
    }
    log_path = os.path.join(args.output_dir, f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Scores saved to: {log_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "bon":
        run_bon(args)
    elif args.mode == "score":
        run_score(args)
