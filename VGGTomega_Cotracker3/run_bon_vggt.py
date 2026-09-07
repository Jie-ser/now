"""
单条 Best-of-N CLI（VGGT-Omega + CoTracker3 后端）。

用法：
    python VGGT_Cotracker3/run_bon_vggt.py \
        --ckpt_dir /path/to/Wan2.2-I2V-A14B \
        --vggt_model /path/to/vggt_omega_checkpoint.pth \
        --image /path/to/first_frame.png \
        --prompt "robot arm picks up the red block" \
        --N 8 --size 480*832 --sample_shift 5.0 --t5_cpu
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

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Wan2.2"))
sys.path.insert(0, str(ROOT))

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

from VGGT_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
from VGGT_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoReward VGGT+CoTracker3 Best-of-N Pipeline (单条)"
    )

    # Wan2.2 生成参数
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Wan2.2 checkpoint 目录。")
    parser.add_argument("--image", type=str, required=True,
                        help="首帧图片路径。")
    parser.add_argument("--prompt", type=str, required=True,
                        help="动作指令文本。")
    parser.add_argument("--size", type=str, default="480*832",
                        choices=["720*1280", "1280*720", "480*832", "832*480"],
                        help="输出分辨率。")
    parser.add_argument("--frame_num", type=int, default=81,
                        help="输出帧数（必须是 4n+1）。")
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_shift", type=float, default=5.0,
                        help="噪声调度 shift（480p 推荐 3.0-5.0）。")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--t5_cpu", action="store_true",
                        help="T5 放 CPU 以节省显存。")

    # BoN 参数
    parser.add_argument("--N", type=int, default=8,
                        help="候选数。")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="基础种子（候选使用 seed_base+i）。")

    # VGGT-Omega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGT-Omega 权重路径（.pth 文件）。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。None 则用 torch.hub 自动下载。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20,
                        help="评分用帧数。")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="CoTracker3 每批追踪点数。")
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGT-Omega 输入分辨率（长边参考）。")

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50,
                        help="R_static 权重。")
    parser.add_argument("--dynamic_weight", type=float, default=0.30,
                        help="R_dynamic 权重。")
    parser.add_argument("--motion_weight", type=float, default=0.20,
                        help="R_motion 权重。")

    # Dynamic mask
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.01,
                        help="动态 mask 阈值（scene_scale 的比例）。")

    # R_static
    parser.add_argument("--tau_reproj", type=float, default=0.05,
                        help="重投影误差温度。")
    parser.add_argument("--occlusion_margin", type=float, default=1.05,
                        help="遮挡深度余量乘子。")

    # R_dynamic
    parser.add_argument("--tau_accel", type=float, default=0.02,
                        help="加速度 penalty 温度。")
    parser.add_argument("--tau_speed", type=float, default=1.5,
                        help="极端速度 penalty 温度。")
    parser.add_argument("--max_sample_pixels", type=int, default=1000,
                        help="动态轨迹分析最大像素数。")

    # R_motion
    parser.add_argument("--tau_cam", type=float, default=0.02,
                        help="相机加速度温度。")
    parser.add_argument("--tau_rot", type=float, default=0.05,
                        help="旋转加速度温度。")
    parser.add_argument("--min_motion", type=float, default=0.005,
                        help="最低运动量。")
    parser.add_argument("--tau_motion", type=float, default=0.02,
                        help="Motion gate sigmoid 温度。")

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/geo_reward_bon_vggt",
                        help="输出目录。")

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


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan2.2 I2V + VGGT-Omega + CoTracker3.")

    cfg = build_reward_config(args)

    # 加载 Wan2.2
    logger.info("加载 Wan2.2 I2V pipeline...")
    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    # 构建 BoN pipeline
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

    # 读取输入图片
    img = Image.open(args.image).convert("RGB")
    logger.info(f"输入图片: {args.image} ({img.size[0]}x{img.size[1]})")
    logger.info(f"Prompt: {args.prompt}")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(args.image).stem
    case_dir = os.path.join(args.output_dir, f"{image_stem}_{timestamp}")
    os.makedirs(case_dir, exist_ok=True)

    # 保存函数
    def _save_fn(tensor, path):
        save_video(
            tensor=tensor[None],
            save_file=path,
            fps=wan_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    # 执行 BoN
    t0 = time.time()
    result = bon.generate(
        image=img,
        prompt=args.prompt,
        N=args.N,
        frame_num=args.frame_num,
        seed_base=args.seed_base,
        output_dir=case_dir,
        save_fn=_save_fn,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
    )
    total_time = time.time() - t0

    # 保存结果
    results_log = {
        "prompt": args.prompt,
        "image": os.path.abspath(args.image),
        "reward_version": "vggt_v2",
        "N": args.N,
        "best_seed": result["best_seed"],
        "best_score": result["best_score"],
        "total_time_sec": total_time,
        "config": {
            "vggt_model": args.vggt_model,
            "cotracker_model": args.cotracker_model,
            "image_resolution": args.image_resolution,
            "chunk_size": args.chunk_size,
            "num_frames_for_reward": args.num_frames_for_reward,
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
            "size": args.size,
            "frame_num": args.frame_num,
            "sampling_steps": args.sampling_steps,
            "guide_scale": args.guide_scale,
            "sample_shift": args.sample_shift,
            "seed_base": args.seed_base,
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

    log_path = os.path.join(case_dir, "rewards.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    logger.info(f"结果保存至: {case_dir}")


if __name__ == "__main__":
    main()
