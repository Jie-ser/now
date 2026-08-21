"""
Score a single video file using 4RC GeoReward V2.

Supports .mp4 video files and .pt tensor files (Wan2.2 output format).

Usage:
    python score_video_v2.py --video path/to/video.mp4 --fourrc_model /path/to/4RC
    python score_video_v2.py --video path/to/video.pt --output_dir results/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from geo_reward import ReconstructionReward, ReconRewardConfig
from geo_reward.utils import wan_output_to_pil, sample_frames


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Score a video with 4RC GeoReward V2")

    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file (.mp4 or .pt)")
    parser.add_argument("--output_dir", type=str, default="outputs/scores_v2",
                        help="Directory to save result JSON.")

    # 4RC model
    parser.add_argument("--fourrc_model", type=str, required=True,
                        help="Path to 4RC model checkpoint directory.")
    parser.add_argument("--image_size", type=int, default=518,
                        help="4RC input resolution (default: 518).")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Number of keyframes to sample.")

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
                        help="Max pixels sampled for dynamic trajectory analysis.")

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
                        help="Confidence quantile for valid mask (Q20 = keep top 80%%).")

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


def load_4rc_model(model_path, device="cuda"):
    """Load 4RC (Arc) model from checkpoint path."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "4RC-main", "4RC-main"))
    from arc.models.arc.arc import Arc

    logger.info(f"Loading 4RC model from: {model_path}")
    model = Arc.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def load_video_as_frames(video_path):
    """Load a video file and return a list of PIL Images."""
    ext = Path(video_path).suffix.lower()

    if ext == ".pt":
        tensor = torch.load(video_path, map_location="cpu")
        if tensor.dim() == 4 and tensor.shape[0] == 3:
            return wan_output_to_pil(tensor)
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

    cfg = build_recon_config(args)
    model = load_4rc_model(args.fourrc_model, device=device)

    recon_reward = ReconstructionReward(model=model, device=device, cfg=cfg)

    logger.info("Computing V2 reward...")
    reward = recon_reward.compute_reward(sampled_frames)

    logger.info(f"Results: total={reward['total']:.4f} "
                f"(R_static={reward['R_static']:.4f}, "
                f"R_dynamic={reward['R_dynamic']:.4f}, "
                f"R_motion={reward['R_motion']:.4f}, "
                f"G_anchor={reward['G_anchor']:.2f}, "
                f"dynamic_ratio={reward['dynamic_ratio']:.3f})")

    os.makedirs(args.output_dir, exist_ok=True)
    result_name = f"{video_path.stem}_v2_result.json"
    result_path = os.path.join(args.output_dir, result_name)

    result = {
        "video": str(video_path.resolve()),
        "reward": reward,
        "config": {
            "fourrc_model": args.fourrc_model,
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
            "dynamic_threshold_ratio": args.dynamic_threshold_ratio,
            "tau_reproj": args.tau_reproj,
            "occlusion_margin": args.occlusion_margin,
            "tau_accel": args.tau_accel,
            "tau_speed": args.tau_speed,
            "max_sample_pixels": args.max_sample_pixels,
            "tau_cam": args.tau_cam,
            "tau_rot": args.tau_rot,
            "min_motion": args.min_motion,
            "tau_motion": args.tau_motion,
            "conf_valid_quantile": args.conf_valid_quantile,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
