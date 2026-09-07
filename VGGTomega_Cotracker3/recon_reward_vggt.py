"""
VGGT-Omega + CoTracker3 后端的 Reconstruction Quality Reward。

与 geo_reward/recon_reward.py 使用完全相同的 Reward 公式，差异仅在：
- 几何后端：4RC → VGGT-Omega + CoTracker3 组合
- R_static: 去掉 valid_geo（conf-based mask），所有像素参与
- R_dynamic: 用 CoTracker3 visibility 替代 valid_track（物理遮挡 vs 模型置信度）
- 不包含梯度引导相关功能

Reward 组成:
- R_static: 静态区域跨帧深度重投影一致性
- R_dynamic: 动态区域轨迹加速度 penalty + 覆盖率 + 速度平滑度
- R_motion: 相机平滑度 + motion gate + 瞬移 penalty
- G_anchor: 首帧几何合理性门控

Total: R_total = G_anchor * (0.50*R_static + 0.30*R_dynamic + 0.20*R_motion)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .combo_adapter import run_combo_inference


@dataclass
class VGGTReconRewardConfig:
    """
    配置类，与 ReconRewardConfig 参数完全一致（去掉 conf_valid_quantile 和梯度引导参数）。
    """
    # Weights
    static_weight: float = 0.50
    dynamic_weight: float = 0.30
    motion_weight: float = 0.20

    # Dynamic mask
    dynamic_threshold_ratio: float = 0.01

    # R_static
    tau_reproj: float = 0.05
    occlusion_margin: float = 1.05

    # R_dynamic
    tau_accel: float = 0.02
    tau_speed: float = 1.5
    max_sample_pixels: int = 1000

    # R_motion
    tau_cam: float = 0.02
    tau_rot: float = 0.05
    min_motion: float = 0.005
    tau_motion: float = 0.02

    # Frame sampling
    max_frames: int = 20
    image_size: int = 512  # VGGT-Omega 输入分辨率（长边参考）

    # CoTracker3
    chunk_size: int = 4096  # 每批追踪的像素数


class VGGTReconstructionReward:
    """
    VGGT-Omega + CoTracker3 后端的 Reconstruction Quality Reward。

    使用 VGGT-Omega 获取 depth/camera/pts，CoTracker3 获取 dense tracking，
    计算与 4RC 版本完全一致的 reward 公式。
    """

    def __init__(self, vggt_model=None, cotracker_model=None,
                 device="cuda", cfg=None):
        """
        Args:
            vggt_model: 预加载的 VGGTOmega 实例。可延迟到 compute_reward() 传入。
            cotracker_model: 预加载的 CoTrackerPredictor 实例。可延迟到 compute_reward() 传入。
            device: 推理设备。
            cfg: VGGTReconRewardConfig 实例。
        """
        self.vggt_model = vggt_model
        self.cotracker_model = cotracker_model
        self.device = device
        self.cfg = cfg or VGGTReconRewardConfig()

    def compute_reward(self, frames_pil, vggt_model=None, cotracker_model=None):
        """
        计算 GeoReward。

        所有 checkpoint（early/mid/final）使用完全相同的公式，
        仅输入帧数不同。

        Args:
            frames_pil: List[PIL.Image]（采样后的关键帧）
            vggt_model: 可选 VGGTOmega 模型覆盖
            cotracker_model: 可选 CoTrackerPredictor 模型覆盖

        Returns:
            dict:
              - total: float, 最终 reward [0, 1]
              - version: "vggt_v2"
              - R_static, R_dynamic, R_motion, G_anchor: float
              - scene_scale, dynamic_ratio, visibility_ratio: float
        """
        vggt = vggt_model or self.vggt_model
        cotracker = cotracker_model or self.cotracker_model
        if vggt is None:
            raise ValueError("No VGGT-Omega model provided.")
        if cotracker is None:
            raise ValueError("No CoTracker3 model provided.")

        # 组合推理
        predictions = run_combo_inference(
            vggt, cotracker, frames_pil,
            image_resolution=self.cfg.image_size,
            chunk_size=self.cfg.chunk_size,
            device=self.device,
        )

        pts = predictions["pts"]             # (N, H, W, 3)
        track = predictions["track"]         # (N, H, W, 3)
        extrinsics = predictions["extrinsic"]  # (N, 4, 4)
        intrinsics = predictions["intrinsic"]  # (N, 3, 3)
        visibility = predictions["visibility"]  # (N, H, W) bool

        # 场景尺度
        scene_scale = _compute_scene_scale(pts, extrinsic_frame0=extrinsics[0])

        # 动态/静态分割（与 fourrc_adapter.compute_dynamic_mask 相同逻辑）
        static_mask, dynamic_mask = _compute_dynamic_mask(
            track,
            threshold_ratio=self.cfg.dynamic_threshold_ratio,
            scene_scale=scene_scale,
        )

        # 计算各分量
        r_static = self._compute_R_static(
            pts, extrinsics, intrinsics, static_mask
        )
        r_dynamic = self._compute_R_dynamic(
            track, dynamic_mask, visibility, scene_scale
        )
        r_motion = self._compute_R_motion(
            extrinsics, track, dynamic_mask, scene_scale
        )
        g_anchor = self._compute_anchor_gate(pts[0], static_mask, extrinsics[0])

        r_total = g_anchor * (
            self.cfg.static_weight * r_static
            + self.cfg.dynamic_weight * r_dynamic
            + self.cfg.motion_weight * r_motion
        )

        return {
            "total": float(r_total),
            "version": "vggt_v2",
            "R_static": float(r_static),
            "R_dynamic": float(r_dynamic),
            "R_motion": float(r_motion),
            "G_anchor": float(g_anchor),
            "scene_scale": float(scene_scale),
            "dynamic_ratio": float(dynamic_mask.float().mean()),
            "visibility_ratio": float(visibility.float().mean()),
        }

    # ===== 以下方法从 recon_reward.py 逐行复制，仅修改 mask 逻辑 =====

    def _compute_R_static(self, pts, extrinsics, intrinsics, static_mask):
        """
        静态区域几何一致性：跨帧深度重投影。

        与 recon_reward.py 的 _compute_R_static 完全相同，
        但去掉 valid_geo 过滤（所有静态像素参与）。
        """
        N, H, W, _ = pts.shape
        device = pts.device
        errors_reproj = []
        valid_counts = []

        frame_pairs = self._get_frame_pairs(N, strides=[1, 3, 5])

        for i, j in frame_pairs:
            # 差异：不再用 valid_geo[i] & valid_geo[j]，仅用 static_mask
            mask = static_mask
            if mask.sum() < 100:
                continue

            pts_i = pts[i][mask]  # (K, 3)

            w2c_j = torch.linalg.inv(extrinsics[j])  # (4, 4)
            pts_i_homo = F.pad(pts_i, (0, 1), value=1.0)  # (K, 4)
            pts_i_in_cam_j = (w2c_j @ pts_i_homo.T).T[:, :3]  # (K, 3)

            proj_uv = (intrinsics[j] @ pts_i_in_cam_j.T).T  # (K, 3)
            proj_u = proj_uv[:, 0] / (proj_uv[:, 2] + 1e-8)
            proj_v = proj_uv[:, 1] / (proj_uv[:, 2] + 1e-8)
            proj_depth = pts_i_in_cam_j[:, 2]

            in_bounds = (proj_u >= 0) & (proj_u < W - 1) & (proj_v >= 0) & (proj_v < H - 1)
            positive_depth = proj_depth > 0.01
            valid = in_bounds & positive_depth

            if valid.sum() < 50:
                continue

            pts_j_world = pts[j].reshape(H * W, 3)
            pts_j_homo = F.pad(pts_j_world, (0, 1), value=1.0)
            pts_j_cam = (w2c_j @ pts_j_homo.T).T[:, :3].reshape(H, W, 3)
            depth_map_j_cam = pts_j_cam[..., 2]  # (H, W)

            grid = torch.stack([
                2.0 * proj_u[valid] / (W - 1) - 1.0,
                2.0 * proj_v[valid] / (H - 1) - 1.0,
            ], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, K_valid, 2)

            sampled_depth = F.grid_sample(
                depth_map_j_cam.unsqueeze(0).unsqueeze(0),
                grid, mode='bilinear', align_corners=True
            ).reshape(-1)  # (K_valid,)

            not_occluded = proj_depth[valid] < sampled_depth * self.cfg.occlusion_margin
            valid_sampled = (sampled_depth > 0.01) & not_occluded

            if valid_sampled.sum() < 20:
                continue

            log_error = torch.abs(
                torch.log(proj_depth[valid][valid_sampled] + 1e-8) -
                torch.log(sampled_depth[valid_sampled] + 1e-8)
            )
            errors_reproj.append(log_error.median())
            valid_counts.append(valid_sampled.sum().float() / mask.sum().float())

        if not errors_reproj:
            return torch.tensor(0.5, device=device)

        E_reproj = torch.stack(errors_reproj).mean()
        V_ratio = torch.stack(valid_counts).mean()

        R_static = (
            torch.exp(-E_reproj / self.cfg.tau_reproj)
            * torch.sigmoid((V_ratio - 0.3) / 0.1)
        )
        return R_static.clamp(0.0, 1.0)

    def _compute_R_dynamic(self, track, dynamic_mask, visibility, scene_scale):
        """
        动态区域追踪质量：轨迹分析。

        与 recon_reward.py 的 _compute_R_dynamic 完全相同，
        但用 visibility（CoTracker3）替代 valid_track（4RC conf）。
        """
        N, H, W, _ = track.shape
        device = track.device

        # 差异：用 visibility 替代 valid_track
        combined_mask = dynamic_mask.unsqueeze(0) & visibility  # (N, H, W)

        dynamic_total = dynamic_mask.sum().float()
        if dynamic_total < 50:
            return torch.tensor(0.5, device=device)

        per_frame_coverage = combined_mask.float().sum(dim=(-1, -2)) / dynamic_total
        coverage = per_frame_coverage.mean()

        if N < 3:
            return coverage ** 0.5

        all_valid = combined_mask.all(dim=0) & dynamic_mask  # (H, W)
        valid_indices = all_valid.nonzero()  # (M, 2)

        if valid_indices.shape[0] < 10:
            return coverage ** 0.5

        K = min(self.cfg.max_sample_pixels, valid_indices.shape[0])
        # 确定性均匀步长采样（行优先顺序提供空间覆盖）
        step = valid_indices.shape[0] / K
        selected_idx = (torch.arange(K, device=device) * step).long()
        selected = valid_indices[selected_idx]

        trajectories = track[:, selected[:, 0], selected[:, 1], :]  # (N, K, 3)

        # 加速度（二阶有限差分）
        accel = trajectories[2:] - 2 * trajectories[1:-1] + trajectories[:-2]
        accel_magnitude = torch.norm(accel, dim=-1) / scene_scale  # (N-2, K)
        E_accel = accel_magnitude.median()

        # 速度过量（p95/median 比率）
        velocity = trajectories[1:] - trajectories[:-1]  # (N-1, K, 3)
        speed = torch.norm(velocity, dim=-1)  # (N-1, K)

        speed_flat = speed.flatten()
        if speed_flat.numel() > 0:
            p95 = torch.quantile(speed_flat, 0.95)
            median_speed = speed_flat.median()
            speed_excess = (p95 / (median_speed + 1e-8)) - 1.0
            E_speed = speed_excess.clamp(min=0.0)
        else:
            E_speed = torch.tensor(0.0, device=device)

        R_dynamic = (
            coverage ** 0.5
            * torch.exp(-E_accel / self.cfg.tau_accel)
            * torch.exp(-E_speed / self.cfg.tau_speed)
        )
        return R_dynamic.clamp(0.0, 1.0)

    def _compute_R_motion(self, extrinsics, track, dynamic_mask, scene_scale):
        """
        相机和运动质量评估。

        与 recon_reward.py 的 _compute_R_motion 完全相同。
        """
        N = extrinsics.shape[0]
        device = extrinsics.device

        # 相机平移加速度
        cam_positions = extrinsics[:, :3, 3]  # (N, 3)
        if N >= 3:
            cam_accel = cam_positions[2:] - 2 * cam_positions[1:-1] + cam_positions[:-2]
            E_cam_accel = torch.norm(cam_accel, dim=-1).median() / scene_scale
        else:
            E_cam_accel = torch.tensor(0.0, device=device)

        # 相机旋转加速度
        cam_rotations = extrinsics[:, :3, :3]  # (N, 3, 3)
        if N >= 3:
            rot_diffs = []
            for t in range(N - 1):
                R_rel = cam_rotations[t + 1] @ cam_rotations[t].T
                angle = torch.acos(
                    torch.clamp((R_rel.trace() - 1) / 2, -1.0, 1.0)
                )
                rot_diffs.append(angle)
            rot_diffs = torch.stack(rot_diffs)
            if len(rot_diffs) >= 2:
                rot_accel = torch.abs(rot_diffs[1:] - rot_diffs[:-1])
                E_rot_accel = rot_accel.median()
            else:
                E_rot_accel = torch.tensor(0.0, device=device)
        else:
            E_rot_accel = torch.tensor(0.0, device=device)

        # Motion gate
        displacement_per_frame = torch.norm(track, dim=-1)  # (N, H, W)
        max_displacement = displacement_per_frame.max(dim=0).values  # (H, W)
        if dynamic_mask.sum() > 0:
            dynamic_motion = max_displacement[dynamic_mask].median()
        else:
            dynamic_motion = max_displacement.quantile(0.90)

        gate = torch.sigmoid((dynamic_motion - self.cfg.min_motion) / self.cfg.tau_motion)

        # 瞬移 penalty
        if dynamic_mask.sum() > 0 and N > 1:
            frame_displacements = torch.norm(track[1:] - track[:-1], dim=-1)  # (N-1, H, W)
            dynamic_frame_disp = frame_displacements[:, dynamic_mask]
            if dynamic_frame_disp.numel() > 0:
                teleport_ratio = (dynamic_frame_disp > 0.5 * scene_scale).float().mean()
                E_teleport = teleport_ratio
            else:
                E_teleport = torch.tensor(0.0, device=device)
        else:
            E_teleport = torch.tensor(0.0, device=device)

        R_motion = (
            gate
            * torch.exp(-E_cam_accel / self.cfg.tau_cam)
            * torch.exp(-E_rot_accel / self.cfg.tau_rot)
            * (1.0 - E_teleport)
        )
        return R_motion.clamp(0.0, 1.0)

    def _compute_anchor_gate(self, pts_frame0, static_mask, extrinsic_frame0=None):
        """
        首帧几何合理性门控。

        与 recon_reward.py 的 _compute_anchor_gate 完全相同。
        """
        device = pts_frame0.device

        # 变换世界点到 camera-0 坐标系以获取正确的深度
        if extrinsic_frame0 is not None:
            w2c = torch.linalg.inv(extrinsic_frame0)  # (4, 4)
            H, W = pts_frame0.shape[:2]
            pts_flat = pts_frame0.reshape(-1, 3)
            pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
            pts_cam = (w2c @ pts_homo.T).T[:, :3].reshape(H, W, 3)
            depth_map = pts_cam[..., 2]
        else:
            depth_map = pts_frame0[..., 2]

        if static_mask.sum() < 50:
            depth = depth_map.flatten()
        else:
            depth = depth_map[static_mask]

        valid_depth = (depth > 0.01) & (depth < 100.0) & depth.isfinite()
        anchor_validity = valid_depth.float().mean()

        G_anchor = torch.sigmoid((anchor_validity - 0.8) / 0.05)
        return G_anchor

    @staticmethod
    def _get_frame_pairs(N, strides=None):
        """生成帧对用于重投影分析。与 recon_reward.py 完全相同。"""
        if strides is None:
            strides = [1, 3, 5]
        pairs = []
        for stride in strides:
            for i in range(N - stride):
                pairs.append((i, i + stride))
        return pairs


# ===== 辅助函数（从 fourrc_adapter.py 复制，避免依赖 4RC） =====

def _compute_scene_scale(pts, static_mask=None, extrinsic_frame0=None):
    """
    从首帧深度计算场景尺度（静态区域 depth 中位数）。

    与 fourrc_adapter.compute_scene_scale 完全相同。
    """
    pts_frame0 = pts[0]  # (H, W, 3)

    if extrinsic_frame0 is not None:
        w2c = torch.linalg.inv(extrinsic_frame0)  # (4, 4)
        H, W = pts_frame0.shape[:2]
        pts_flat = pts_frame0.reshape(-1, 3)
        pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
        pts_cam = (w2c @ pts_homo.T).T[:, :3].reshape(H, W, 3)
        depth_frame0 = pts_cam[..., 2]
    else:
        depth_frame0 = pts_frame0[..., 2]

    if static_mask is not None and static_mask.sum() > 50:
        depths = depth_frame0[static_mask]
    else:
        depths = depth_frame0.flatten()

    valid = depths[depths.isfinite() & (depths > 0.01)]
    if valid.numel() > 0:
        scale = valid.median().item()
    else:
        finite_depths = depths[depths.isfinite()]
        if finite_depths.numel() > 0:
            scale = finite_depths.abs().median().item()
        else:
            scale = 1.0

    return max(scale, 1e-6)


def _compute_dynamic_mask(track, threshold_ratio=0.01, scene_scale=None):
    """
    从 track 的逐帧最大位移推断 static/dynamic mask。

    与 fourrc_adapter.compute_dynamic_mask 完全相同。
    """
    if scene_scale is None:
        raise ValueError(
            "scene_scale is required for _compute_dynamic_mask."
        )

    displacement_per_frame = torch.norm(track, dim=-1)  # (N, H, W)
    max_displacement = displacement_per_frame.max(dim=0).values  # (H, W)

    threshold = threshold_ratio * scene_scale
    dynamic_mask = max_displacement > threshold
    static_mask = ~dynamic_mask

    return static_mask, dynamic_mask
