"""Run a contiguous image range with Wan and DA3 loaded once per GPU process."""

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

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video
from geo_reward import DA3GeoReward, GeoRewardBoN, GeoRewardBoNProgressive, GeometryRewardConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for GeoReward V1 BoN")
    parser.add_argument("--start", type=int, required=True, help="First test index, inclusive")
    parser.add_argument("--end", type=int, required=True, help="Last test index, inclusive")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--da3_model", required=True)
    parser.add_argument("--input_dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--prompts", type=Path, default=ROOT / "batch_prompts.json")
    parser.add_argument(
        "--name_prefix",
        default="test",
        help="Input filename and prompt-key prefix, e.g. 'test' or 'test_real'.",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "geo_reward_bon")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--size", default="480*832", choices=MAX_AREA_CONFIGS.keys())
    parser.add_argument("--sample_shift", type=float, default=3.0)
    parser.add_argument("--t5_cpu", action="store_true")

    # V1 reward config
    parser.add_argument("--image_diff_threshold", type=float, default=15.0)
    parser.add_argument("--image_vote_ratio", type=float, default=0.30)
    parser.add_argument("--depth_change_threshold", type=float, default=0.05)
    parser.add_argument("--min_motion_threshold", type=float, default=0.01)
    parser.add_argument("--tau_smooth", type=float, default=2.0)
    parser.add_argument("--tau_shape", type=float, default=0.10)
    parser.add_argument("--motion_shape_weight", type=float, default=0.70)
    parser.add_argument("--motion_smooth_weight", type=float, default=0.30)

    # Progressive elimination args
    parser.add_argument("--no_progressive", action="store_true",
                        help="Disable progressive elimination (use original sequential BoN).")
    parser.add_argument("--sigma_checkpoints", type=float, nargs="+",
                        default=[0.83, 0.63],
                        help="σ thresholds for checkpoints (default: 0.83 0.63).")
    parser.add_argument("--elimination_ratio", type=float, default=0.5,
                        help="Fraction of candidates to eliminate at each checkpoint (default: 0.5).")
    parser.add_argument("--min_survivors", type=int, default=2,
                        help="Minimum number of survivors at any checkpoint (default: 2).")
    parser.add_argument("--score_epsilon", type=float, default=0.02,
                        help="Score indistinguishability threshold for safety keep (default: 0.02).")
    parser.add_argument("--early_max_frames", type=int, default=12,
                        help="Number of frames to sample for DA3 at early checkpoint (default: 12).")

    return parser.parse_args()


def save_case(candidates, rewards, best_idx, image_path, prompt, output_dir, cfg, args, elapsed):
    case_dir = output_dir / f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ranked_indices = sorted(range(len(rewards)), key=lambda i: rewards[i]["total"], reverse=True)
    for rank, original_idx in enumerate(ranked_indices, start=1):
        reward = rewards[original_idx]["total"]
        suffix = "_BEST" if original_idx == best_idx else ""
        save_video(
            tensor=candidates[original_idx][None],
            save_file=str(case_dir / f"candidate_{rank:02d}_r{reward:.4f}{suffix}.mp4"),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    result = {
        "prompt": prompt,
        "image": str(image_path.resolve()),
        "N": args.N,
        "best_rank": 1,
        "best_original_idx": best_idx,
        "best_reward": rewards[best_idx]["total"],
        "total_time_sec": elapsed,
        "candidates": [
            {"rank": rank, "original_idx": i, "reward": rewards[i], "is_best": i == best_idx}
            for rank, i in enumerate(ranked_indices, start=1)
        ],
        "config": {
            "reward_version": "v1",
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
    with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d candidates to %s", len(candidates), case_dir)


def find_input_image(input_dir, name):
    """Return the supported image file for a test name, regardless of suffix."""
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
        raise RuntimeError("CUDA is required for Wan2.2 I2V.")

    reward_cfg = GeometryRewardConfig(
        image_diff_threshold=args.image_diff_threshold,
        image_vote_ratio=args.image_vote_ratio,
        depth_change_threshold=args.depth_change_threshold,
        min_motion_threshold=args.min_motion_threshold,
        tau_smooth=args.tau_smooth,
        tau_shape=args.tau_shape,
        motion_shape_weight=args.motion_shape_weight,
        motion_smooth_weight=args.motion_smooth_weight,
    )

    logger.info("Loading DA3 once on %s...", torch.cuda.get_device_name(0))
    da3_reward = DA3GeoReward(
        model_name=args.da3_model,
        device=str(device),
        process_res=504,
        cfg=reward_cfg,
    )

    logger.info("Loading Wan2.2 I2V once...")
    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0, t5_cpu=args.t5_cpu
    )
    bon = GeoRewardBoN(wan_i2v=wan_i2v, da3_reward=da3_reward, max_frames=20)

    use_progressive = not args.no_progressive

    if use_progressive:
        bon_prog = GeoRewardBoNProgressive(
            wan_i2v=wan_i2v,
            da3_reward=da3_reward,
            max_frames=20,
            sigma_checkpoints=args.sigma_checkpoints,
            elimination_ratio=args.elimination_ratio,
            min_survivors=args.min_survivors,
            score_epsilon=args.score_epsilon,
            early_max_frames=args.early_max_frames,
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
        t0 = time.time()

        case_dir = args.output_dir / f"{image_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        case_dir.mkdir(parents=True, exist_ok=True)

        if use_progressive:
            best_video, result_log, best_seed = bon_prog.generate(
                prompt=prompt,
                image=Image.open(image_path).convert("RGB"),
                N=args.N,
                frame_num=81,
                seed_base=None,
                output_dir=str(case_dir),
                save_fn=_save_fn,
                max_area=MAX_AREA_CONFIGS[args.size],
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=40,
                guide_scale=5.0,
                offload_model=True,
            )
            elapsed = time.time() - t0
            if best_video is not None:
                best_path = str(case_dir / f"seed_{best_seed}_BEST.mp4")
                _save_fn(best_video, best_path)
            result_log["prompt"] = prompt
            result_log["image"] = str(image_path.resolve())
            result_log["total_time_sec"] = elapsed
            with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
                json.dump(result_log, f, indent=2, ensure_ascii=False)
            logger.info("Saved results to %s", case_dir)
        else:
            candidates, rewards, best_idx = bon.generate(
                prompt=prompt,
                image=Image.open(image_path).convert("RGB"),
                N=args.N,
                frame_num=81,
                seed_base=None,
                max_area=MAX_AREA_CONFIGS[args.size],
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=40,
                guide_scale=5.0,
                offload_model=True,
            )
            save_case(candidates, rewards, best_idx, image_path, prompt,
                      args.output_dir, wan_cfg, args, time.time() - t0)


if __name__ == "__main__":
    main()
