"""
VGGT-Omega + CoTracker3 替代后端 for GeoReward.

用 VGGT-Omega（几何+相机）+ CoTracker3（dense tracking）组合替代 4RC，
实现相同的 Reward 计算，验证 GeoReward 设计的通用性。

所有 Reward 公式与现有 geo_reward/recon_reward.py 完全一致，
仅替换几何后端和 mask 逻辑（去掉 conf，用 CoTracker3 visibility）。
"""

from .recon_reward_vggt import VGGTReconstructionReward, VGGTReconRewardConfig
from .bon_pipeline_vggt import GeoRewardBoNVGGT
from .vggt_omega_adapter import load_vggt_omega, preprocess_frames, run_vggt_omega_inference
from .cotracker3_adapter import load_cotracker3, frames_to_video_tensor, run_dense_tracking
from .combo_adapter import run_combo_inference
