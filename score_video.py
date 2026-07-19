"""
Score a single video file using DA3 GeoReward.

Supports .mp4 video files and .pt tensor files (Wan2.2 output format).

Usage:
    python score_video.py --video path/to/video.mp4
    python score_video.py --video path/to/video.pt --output_dir results/ --keep_ratio 0.8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from geo_reward import DA3GeoReward
from geo_reward.utils import wan_output_to_da3_input, sample_frames


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Score a video with DA3 GeoReward")

    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file (.mp4 or .pt)")
    parser.add_argument("--output_dir", type=str, default="outputs/scores",
                        help="Directory to save result JSON.")
    parser.add_argument("--da3_model", type=str,
                        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="DA3 model name or local path.")
    parser.add_argument("--process_res", type=int, default=504,
                        help="DA3 processing resolution.")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Number of keyframes to sample.")
    parser.add_argument("--reward_stride", type=int, default=2,
                        help="Frame stride for projection consistency.")
    parser.add_argument("--keep_ratio", type=float, default=0.7,
                        help="Truncated mean keep ratio (0-1).")

    return parser.parse_args()


def load_video_as_frames(video_path):
    """
    Load a video file and return a list of PIL Images.

    Supports:
        .pt  - Wan2.2 output tensor (3, T, H, W) in [-1, 1]
        .mp4 - Standard video file decoded with imageio
    """
    ext = Path(video_path).suffix.lower()

    if ext == ".pt":
        tensor = torch.load(video_path, map_location="cpu")
        if tensor.dim() == 4 and tensor.shape[0] == 3:
            return wan_output_to_da3_input(tensor)
        raise ValueError(f"Unexpected .pt tensor shape: {tensor.shape}, expected (3, T, H, W)")

    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        import imageio.v3 as iio
        frames_np = iio.imread(video_path, plugin="pyav")
        frames = [Image.fromarray(frame) for frame in frames_np]
        return frames

    raise ValueError(f"Unsupported video format: {ext}. Use .mp4 or .pt")


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading video: {video_path}")
    all_frames = load_video_as_frames(str(video_path))
    logger.info(f"Loaded {len(all_frames)} frames")

    indices = sample_frames(len(all_frames), args.max_frames)
    sampled_frames = [all_frames[i] for i in indices if i < len(all_frames)]
    logger.info(f"Sampled {len(sampled_frames)} keyframes for scoring")

    logger.info(f"Loading DA3 model: {args.da3_model}")
    da3_reward = DA3GeoReward(
        model_name=args.da3_model,
        device=device,
        process_res=args.process_res,
    )

    logger.info("Computing reward...")
    reward = da3_reward.compute_reward(
        sampled_frames,
        stride=args.reward_stride,
        keep_ratio=args.keep_ratio,
    )

    logger.info(f"Results: total={reward['total']:.4f} "
                f"(proj={reward['proj']:.4f}, anchor={reward['anchor']:.4f}, "
                f"conf={reward['conf']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    result_name = f"{video_path.stem}_cal_result.json"
    result_path = os.path.join(args.output_dir, result_name)

    result = {
        "video": str(video_path.resolve()),
        "reward": {
            "total": reward["total"],
            "proj": reward["proj"],
            "anchor": reward["anchor"],
            "conf": reward["conf"],
        },
        "config": {
            "da3_model": args.da3_model,
            "process_res": args.process_res,
            "max_frames": args.max_frames,
            "reward_stride": args.reward_stride,
            "keep_ratio": args.keep_ratio,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
