"""
Score a single video file using DA3 GeoReward V1.

Supports .mp4 video files and .pt tensor files (Wan2.2 output format).

Usage:
    python score_video.py --video path/to/video.mp4
    python score_video.py --video path/to/video.pt --output_dir results/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from geo_reward import DA3GeoReward, GeometryRewardConfig
from geo_reward.utils import wan_output_to_da3_input, sample_frames


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Score a video with DA3 GeoReward V1")

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

    # V1 reward config
    parser.add_argument("--image_diff_threshold", type=float, default=15.0)
    parser.add_argument("--image_vote_ratio", type=float, default=0.30)
    parser.add_argument("--depth_change_threshold", type=float, default=0.05)
    parser.add_argument("--min_motion_threshold", type=float, default=0.01)
    parser.add_argument("--tau_smooth", type=float, default=2.0)
    parser.add_argument("--tau_shape", type=float, default=0.10)
    parser.add_argument("--motion_shape_weight", type=float, default=0.70)
    parser.add_argument("--motion_smooth_weight", type=float, default=0.30)

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
        try:
            frames_np = iio.imread(video_path, plugin="pyav")
        except ImportError:
            frames_np = iio.imread(video_path, plugin="FFMPEG")
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

    cfg = GeometryRewardConfig(
        image_diff_threshold=args.image_diff_threshold,
        image_vote_ratio=args.image_vote_ratio,
        depth_change_threshold=args.depth_change_threshold,
        min_motion_threshold=args.min_motion_threshold,
        tau_smooth=args.tau_smooth,
        tau_shape=args.tau_shape,
        motion_shape_weight=args.motion_shape_weight,
        motion_smooth_weight=args.motion_smooth_weight,
    )

    logger.info(f"Loading DA3 model: {args.da3_model}")
    da3_reward = DA3GeoReward(
        model_name=args.da3_model,
        device=device,
        process_res=args.process_res,
        cfg=cfg,
    )

    logger.info("Computing reward...")
    reward = da3_reward.compute_reward(sampled_frames)

    logger.info(f"Results: total={reward['total']:.4f} "
                f"(scene={reward['scene']:.4f}, motion={reward['motion']:.4f}, "
                f"gate={reward['motion_gate']:.2f}, shape={reward['shape']:.4f}, "
                f"smooth={reward['smoothness']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    result_name = f"{video_path.stem}_cal_result.json"
    result_path = os.path.join(args.output_dir, result_name)

    result = {
        "video": str(video_path.resolve()),
        "reward": reward,
        "config": {
            "da3_model": args.da3_model,
            "process_res": args.process_res,
            "max_frames": args.max_frames,
            "image_diff_threshold": args.image_diff_threshold,
            "image_vote_ratio": args.image_vote_ratio,
            "depth_change_threshold": args.depth_change_threshold,
            "min_motion_threshold": args.min_motion_threshold,
            "tau_smooth": args.tau_smooth,
            "tau_shape": args.tau_shape,
            "motion_shape_weight": args.motion_shape_weight,
            "motion_smooth_weight": args.motion_smooth_weight,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
