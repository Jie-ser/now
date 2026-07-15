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
from geo_reward import DA3GeoReward, GeoRewardBoN


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for GeoReward BoN")
    parser.add_argument("--start", type=int, required=True, help="First test index, inclusive")
    parser.add_argument("--end", type=int, required=True, help="Last test index, inclusive")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--da3_model", required=True)
    parser.add_argument("--input_dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--prompts", type=Path, default=ROOT / "batch_prompts.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "geo_reward_bon")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--size", default="480*832", choices=MAX_AREA_CONFIGS.keys())
    parser.add_argument("--sample_shift", type=float, default=3.0)
    parser.add_argument("--t5_cpu", action="store_true")
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
    }
    with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d candidates to %s", len(candidates), case_dir)


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
    logger.info("Loading DA3 once on %s...", torch.cuda.get_device_name(0))
    da3_reward = DA3GeoReward(model_name=args.da3_model, device=str(device), process_res=504)
    logger.info("Loading Wan2.2 I2V once...")
    cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0, t5_cpu=args.t5_cpu
    )
    bon = GeoRewardBoN(wan_i2v=wan_i2v, da3_reward=da3_reward, max_frames=20)

    for index in range(args.start, args.end + 1):
        name = f"test{index:04d}"
        image_path = args.input_dir / f"{name}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        prompt = prompts.get(name)
        if not prompt:
            raise KeyError(f"Prompt not found for {name} in {args.prompts}")
        logger.info("===== %s (%d/%d) =====", name, index, args.end)
        t0 = time.time()
        candidates, rewards, best_idx = bon.generate(
            prompt=prompt,
            image=Image.open(image_path).convert("RGB"),
            N=args.N,
            frame_num=81,
            seed_base=None,
            reward_stride=2,
            max_area=MAX_AREA_CONFIGS[args.size],
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=40,
            guide_scale=5.0,
            offload_model=True,
        )
        save_case(candidates, rewards, best_idx, image_path, prompt, args.output_dir, cfg, args, time.time() - t0)


if __name__ == "__main__":
    main()
