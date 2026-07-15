"""Run a contiguous range of image/prompt pairs through GeoReward BoN.

Run this script from the repository root.  GPU selection is deliberately
controlled outside the script with CUDA_VISIBLE_DEVICES, so run_bon.py keeps
using its logical cuda:0 while the shell selects the physical GPU.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for run_bon.py")
    parser.add_argument("--start", type=int, required=True, help="First test index, inclusive")
    parser.add_argument("--end", type=int, required=True, help="Last test index, inclusive")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--da3_model", required=True)
    parser.add_argument("--input_dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--prompts", type=Path, default=ROOT / "batch_prompts.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "geo_reward_bon")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--size", default="480*832")
    parser.add_argument("--sample_shift", type=float, default=3.0)
    parser.add_argument("--t5_cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise ValueError("Require 1 <= start <= end.")

    with args.prompts.open(encoding="utf-8") as f:
        prompts = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, args.end + 1):
        name = f"test{index:04d}"
        image = args.input_dir / f"{name}.png"
        if not image.is_file():
            raise FileNotFoundError(f"Input image not found: {image}")
        if name not in prompts:
            raise KeyError(f"Prompt not found for {name} in {args.prompts}")

        command = [
            sys.executable,
            str(ROOT / "run_bon.py"),
            "--ckpt_dir", args.ckpt_dir,
            "--image", str(image),
            "--prompt", prompts[name],
            "--N", str(args.N),
            "--size", args.size,
            "--sample_shift", str(args.sample_shift),
            "--da3_model", args.da3_model,
            "--output_dir", str(args.output_dir),
        ]
        if args.t5_cpu:
            command.append("--t5_cpu")

        print(f"\n===== {name} ({index}/{args.end}) =====", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
