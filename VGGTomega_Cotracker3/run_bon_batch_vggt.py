"""
批量 Best-of-N CLI（VGGT-Omega + CoTracker3 后端）。

批量处理多个测试用例，Wan2.2 模型只加载一次。
VGGT-Omega 和 CoTracker3 延迟加载，与 DiT 交替占用 GPU。

用法：
    python VGGT_Cotracker3/run_bon_batch_vggt.py \
        --start 1 --end 24 \
        --ckpt_dir /path/to/Wan2.2-I2V-A14B \
        --vggt_model /path/to/vggt_omega_checkpoint.pth \
        --input_dir /path/to/inputs/inputs_real \
        --prompts batch_prompts_real.json \
        --name_prefix test_real \
        --N 8 --t5_cpu --sample_shift 5.0
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Wan2.2"))
sys.path.insert(0, str(ROOT))

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video

from VGGT_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGT_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量 GeoReward VGGT+CoTracker3 Best-of-N Pipeline"
    )

    # 批次范围
    parser.add_argument("--start", type=int, default=None,
                        help="起始测试索引（含）。")
    parser.add_argument("--end", type=int, default=None,
                        help="结束测试索引（含）。")

    # 输入
    parser.add_argument("--input_dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--prompts", type=Path, default=ROOT / "batch_prompts.json")
    parser.add_argument("--name_prefix", default="test",
                        help="输入文件名和 prompt key 的前缀。")
    parser.add_argument("--name_list", action="store_true",
                        help="遍历 prompts JSON 中所有 key（忽略 --start/--end/--name_prefix）。")
    parser.add_argument("--name_list_start", type=int, default=0,
                        help="排序 key 列表的起始索引（0-based，含）。")
    parser.add_argument("--name_list_end", type=int, default=None,
                        help="排序 key 列表的结束索引（不含）。None = 全部。")

    # Wan2.2 模型
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--size", default="480*832",
                        choices=list(MAX_AREA_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=81,
                        help="输出帧数（必须是 4n+1）。")
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=5.0,
                        help="噪声调度 shift。")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--N", type=int, default=8)

    # VGGT-Omega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGT-Omega 权重路径。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。None 则用 torch.hub。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20,
                        help="评分用帧数。")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="CoTracker3 每批追踪点数。")
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGT-Omega 输入分辨率。")

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50)
    parser.add_argument("--dynamic_weight", type=float, default=0.30)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # Dynamic mask
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.01)

    # R_static
    parser.add_argument("--tau_reproj", type=float, default=0.05)
    parser.add_argument("--occlusion_margin", type=float, default=1.05)

    # R_dynamic
    parser.add_argument("--tau_accel", type=float, default=0.02)
    parser.add_argument("--tau_speed", type=float, default=1.5)
    parser.add_argument("--max_sample_pixels", type=int, default=1000)

    # R_motion
    parser.add_argument("--tau_cam", type=float, default=0.02)
    parser.add_argument("--tau_rot", type=float, default=0.05)
    parser.add_argument("--min_motion", type=float, default=0.005)
    parser.add_argument("--tau_motion", type=float, default=0.02)

    # 输出
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "outputs" / "geo_reward_bon_vggt")

    return parser.parse_args()


def build_reward_config(args):
    """从命令行参数构建 VGGTReconRewardConfig。"""
    return VGGTReconRewardConfig(
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
        max_frames=args.num_frames_for_reward,
        image_size=args.image_resolution,
        chunk_size=args.chunk_size,
    )


def find_input_image(input_dir, name):
    """查找测试用例的输入图片。"""
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
        f"输入图片未找到: {name}; 期望 "
        f"{input_dir / (name + '.png')}, {input_dir / (name + '.jpg')}, "
        f"或 {input_dir / (name + '.jpeg')}"
    )


def main():
    args = parse_args()

    # 读取 prompt 文件
    with args.prompts.open(encoding="utf-8") as f:
        prompts = json.load(f)

    # 确定要处理的用例序列
    if args.name_list:
        all_keys = sorted(prompts.keys())
        end_idx = args.name_list_end if args.name_list_end is not None else len(all_keys)
        name_sequence = all_keys[args.name_list_start:end_idx]
        if not name_sequence:
            raise ValueError(f"No keys in range [{args.name_list_start}:{end_idx}] "
                             f"(total keys: {len(all_keys)})")
        logger.info(f"name_list 模式: {len(name_sequence)} 个用例 "
                    f"({name_sequence[0]} .. {name_sequence[-1]})")
    else:
        if args.start is None or args.end is None:
            raise ValueError("--start 和 --end 是必需的（除非使用 --name_list）。")
        if args.start < 1 or args.end < args.start:
            raise ValueError("要求 1 <= start <= end。")
        name_sequence = [f"{args.name_prefix}{i:04d}" for i in range(args.start, args.end + 1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan2.2 I2V + VGGT-Omega + CoTracker3.")

    cfg = build_reward_config(args)

    # 加载 Wan2.2（只加载一次）
    logger.info("加载 Wan2.2 I2V pipeline（只加载一次）...")
    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    # 构建 BoN pipeline（VGGT/CoTracker3 延迟加载）
    bon = GeoRewardBoNVGGT(
        wan_i2v=wan_i2v,
        vggt_model_path=args.vggt_model,
        cotracker_model_path=args.cotracker_model,
        N=args.N,
        num_frames_for_reward=args.num_frames_for_reward,
        reward_config=cfg,
        chunk_size=args.chunk_size,
        device=str(device),
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

    # 逐个用例处理
    total_cases = len(name_sequence)
    for case_idx, name in enumerate(name_sequence):
        image_path = find_input_image(args.input_dir, name)
        prompt = prompts.get(name)
        if not prompt:
            raise KeyError(f"Prompt not found for {name} in {args.prompts}")

        logger.info("===== %s (%d/%d) =====", name, case_idx + 1, total_cases)

        case_dir = args.output_dir / f"{image_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        case_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        img_pil = Image.open(image_path).convert("RGB")

        result = bon.generate(
            image=img_pil,
            prompt=prompt,
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
        )

        elapsed = time.time() - t0

        # 保存结果日志
        results_log = {
            "prompt": prompt,
            "image": str(image_path.resolve()),
            "reward_version": "vggt_v2",
            "N": args.N,
            "best_seed": result["best_seed"],
            "best_score": result["best_score"],
            "total_time_sec": elapsed,
            "config": {
                "vggt_model": args.vggt_model,
                "cotracker_model": args.cotracker_model,
                "image_resolution": args.image_resolution,
                "chunk_size": args.chunk_size,
                "num_frames_for_reward": args.num_frames_for_reward,
                "static_weight": args.static_weight,
                "dynamic_weight": args.dynamic_weight,
                "motion_weight": args.motion_weight,
            },
            "candidates": [
                {
                    "seed": result["seeds"][i],
                    "score": result["all_scores"][i],
                    "details": result["all_details"][i],
                    "is_best": result["seeds"][i] == result["best_seed"],
                }
                for i in range(len(result["seeds"]))
            ],
        }

        with (case_dir / "rewards.json").open("w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2, ensure_ascii=False)
        logger.info("结果保存至 %s (%.1fs)", case_dir, elapsed)


if __name__ == "__main__":
    main()
