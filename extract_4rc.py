"""
Extract full 4RC outputs from a video file.

Saves pts, track, extrinsics, intrinsics, conf, conf_track as .npz.

Usage:
    python extract_4rc.py \
        --video path/to/video.mp4 \
        --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC

    # Batch: all mp4 files in a directory
    python extract_4rc.py \
        --video_dir path/to/videos/ \
        --fourrc_model /pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC

    # Custom frame sampling
    python extract_4rc.py \
        --video path/to/video.mp4 \
        --fourrc_model /path/to/4RC \
        --max_frames 20 --image_size 518
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "4RC-main", "4RC-main"))

from geo_reward.fourrc_adapter import frames_to_views, run_4rc_inference
from geo_reward.utils import sample_frames
from score_video_v2 import load_video_as_frames, load_4rc_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract(video_path, model, device, max_frames=20, image_size=518):
    """Run 4RC on a video and return the raw output dict (tensors on CPU)."""
    all_frames = load_video_as_frames(str(video_path))
    logger.info(f"  Loaded {len(all_frames)} frames")

    indices = sample_frames(len(all_frames), max_frames)
    sampled = [all_frames[i] for i in indices if i < len(all_frames)]
    logger.info(f"  Sampled {len(sampled)} keyframes: {indices[:6]}{'...' if len(indices) > 6 else ''}")

    views = frames_to_views(sampled, target_size=image_size)
    result = run_4rc_inference(model, views, device=device)

    out = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        else:
            out[k] = v
    out["frame_indices"] = np.array(indices, dtype=np.int64)
    return out


def save_npz(result, save_path):
    """Save 4RC output dict as .npz (all tensors converted to numpy)."""
    arrays = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            arrays[k] = v.float().numpy()
        elif isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            arrays[k] = np.array(v)
    np.savez_compressed(save_path, **arrays)
    logger.info(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract 4RC outputs from video(s)")
    parser.add_argument("--video", type=str, default=None, help="Single video path (.mp4/.pt)")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory of videos (batch mode)")
    parser.add_argument("--fourrc_model", type=str, required=True, help="4RC checkpoint path")
    parser.add_argument("--output_dir", type=str, default="outputs/4rc_extracts", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=20, help="Max keyframes to sample")
    parser.add_argument("--image_size", type=int, default=518, help="4RC input resolution")
    args = parser.parse_args()

    if args.video is None and args.video_dir is None:
        parser.error("Provide --video or --video_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_4rc_model(args.fourrc_model, device=device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video:
        videos = [Path(args.video)]
    else:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".pt"}
        videos = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in exts)
        logger.info(f"Found {len(videos)} videos in {args.video_dir}")

    for i, vp in enumerate(videos):
        logger.info(f"[{i+1}/{len(videos)}] Processing: {vp.name}")
        result = extract(vp, model, device, args.max_frames, args.image_size)

        save_name = f"{vp.stem}_4rc.npz"
        save_path = os.path.join(args.output_dir, save_name)
        save_npz(result, save_path)

        pts_shape = result["pts"].shape
        logger.info(f"  pts={pts_shape}, track={result['track'].shape}, "
                    f"extrinsic={result['extrinsic'].shape}, intrinsic={result['intrinsic'].shape}, "
                    f"conf={result['conf'].shape}, conf_track={result['conf_track'].shape}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
