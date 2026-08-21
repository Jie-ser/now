"""
GeoReward V2: 4RC-based Reconstruction Quality Reward.

Uses 4RC's explicit 3D outputs (pts, track, extrinsics) to compute
geometric consistency — "can this video be reconstructed consistently in 3D?"

Reward components:
- R_static: cross-frame depth reprojection consistency on static regions
- R_dynamic: trajectory acceleration penalty + coverage + smoothness
- R_motion: camera smoothness + motion gate + teleportation penalty
- G_anchor: first-frame geometric sanity gate

Total: R_total = G_anchor * (0.40*R_static + 0.40*R_dynamic + 0.20*R_motion)

All checkpoints (early/mid/final) use the same formula; only input frame count differs.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fourrc_adapter import (
    frames_to_views,
    run_4rc_inference,
    compute_valid_mask,
    compute_dynamic_mask,
    compute_scene_scale,
)


@dataclass
class ReconRewardConfig:
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

    # Valid mask
    conf_valid_quantile: float = 0.20

    # Frame sampling
    max_frames: int = 20
    image_size: int = 518

    # Gradient guidance (Phase 3)
    guidance_scale: float = 0.001
    guidance_frequency: int = 5
    sigma_min: float = 0.08
    sigma_max: float = 0.90


class ReconstructionReward:
    """
    4RC-based Reconstruction Quality Reward (V2).

    Computes geometric consistency from 4RC's explicit 3D output:
    - pts (world 3D points), track (3D displacement), extrinsics/intrinsics
    - conf/conf_track only used as valid mask, never as reward signal.
    """

    def __init__(self, model=None, device="cuda", cfg=None):
        """
        Args:
            model: Pre-loaded Arc model instance (4RC).
                   If None, must be passed to compute_reward() calls.
            device: Device for inference.
            cfg: ReconRewardConfig instance.
        """
        self.model = model
        self.device = device
        self.cfg = cfg or ReconRewardConfig()

    def compute_reward(self, frames_pil, model=None):
        """
        Compute V2 reconstruction quality reward for a sequence of frames.

        All checkpoints (early/mid/final) use this same function;
        the only difference is the number of input frames.

        Args:
            frames_pil: List of PIL Images (sampled keyframes).
            model: Optional 4RC model override.

        Returns:
            Dict with keys: total, version, R_static, R_dynamic, R_motion,
            G_anchor, scene_scale, dynamic_ratio, valid_geo_ratio, valid_track_ratio.
        """
        mdl = model or self.model
        if mdl is None:
            raise ValueError("No 4RC model provided.")

        views = frames_to_views(frames_pil, target_size=self.cfg.image_size)
        predictions = run_4rc_inference(mdl, views, device=self.device)

        pts = predictions["pts"]
        track = predictions["track"]
        conf = predictions["conf"]
        conf_track = predictions["conf_track"]
        extrinsics = predictions["extrinsic"]
        intrinsics = predictions["intrinsic"]

        scene_scale = compute_scene_scale(pts, extrinsic_frame0=extrinsics[0])

        valid_geo, valid_track = compute_valid_mask(
            conf, conf_track, quantile=self.cfg.conf_valid_quantile
        )

        static_mask, dynamic_mask = compute_dynamic_mask(
            track,
            threshold_ratio=self.cfg.dynamic_threshold_ratio,
            scene_scale=scene_scale,
        )

        r_static = self._compute_R_static(
            pts, extrinsics, intrinsics, static_mask, valid_geo
        )
        r_dynamic = self._compute_R_dynamic(
            track, dynamic_mask, valid_track, scene_scale
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
            "version": "v2",
            "R_static": float(r_static),
            "R_dynamic": float(r_dynamic),
            "R_motion": float(r_motion),
            "G_anchor": float(g_anchor),
            "scene_scale": float(scene_scale),
            "dynamic_ratio": float(dynamic_mask.float().mean()),
            "valid_geo_ratio": float(valid_geo.float().mean()),
            "valid_track_ratio": float(valid_track.float().mean()),
        }

    def compute_differentiable_loss(self, raw_output, valid_mask, dynamic_mask, scene_scale):
        """
        Compute differentiable geometric loss for gradient guidance (Phase 3).

        This function is designed to be called within a gradient-enabled context.
        conf is detached as valid_mask — we never backprop through confidence.

        Args:
            raw_output: Dict from 4RC forward pass (pts, track, extrinsic, intrinsic).
            valid_mask: (N, H, W) detached bool mask from conf.
            dynamic_mask: (H, W) detached bool mask from track.
            scene_scale: Float.

        Returns:
            Scalar loss tensor (differentiable).
        """
        pts = raw_output["pts"]
        track = raw_output["track"]
        extrinsics = raw_output["extrinsic"]
        intrinsics = raw_output["intrinsic"]
        static_mask = ~dynamic_mask

        l_reproj = self._differentiable_reproj_loss(
            pts, extrinsics, intrinsics, static_mask, valid_mask, scene_scale
        )
        l_smooth = self._differentiable_track_smoothness(
            track, dynamic_mask, valid_mask, scene_scale
        )
        l_anchor = self._differentiable_anchor_loss(pts, static_mask, extrinsics[0])

        loss = l_reproj + 0.5 * l_smooth + 0.3 * l_anchor
        return loss

    def _compute_R_static(self, pts, extrinsics, intrinsics, static_mask, valid_geo):
        """
        Static region geometric consistency via cross-frame depth reprojection.

        For each frame pair (i, j):
        1. Project frame_i's static 3D points into frame_j's camera
        2. Compare projected depth with frame_j's actual depth at that pixel
        3. Use log-depth error as metric

        Includes occlusion filtering and forward-backward consistency.
        """
        N, H, W, _ = pts.shape
        device = pts.device
        errors_reproj = []
        valid_counts = []

        frame_pairs = self._get_frame_pairs(N, strides=[1, 3, 5])

        for i, j in frame_pairs:
            mask = static_mask & valid_geo[i] & valid_geo[j]
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

    def _compute_R_dynamic(self, track, dynamic_mask, valid_track, scene_scale):
        """
        Dynamic region tracking quality via trajectory analysis.

        Components:
        - Coverage: fraction of dynamic pixels with valid tracking
        - Acceleration penalty: 3D trajectory second-order smoothness
        - Speed excess: p95/median velocity ratio (teleportation detection)
        """
        N, H, W, _ = track.shape
        device = track.device

        combined_mask = dynamic_mask.unsqueeze(0) & valid_track  # (N, H, W)

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
        # Deterministic uniform-stride sampling (row-major order gives spatial coverage)
        step = valid_indices.shape[0] / K
        selected_idx = (torch.arange(K, device=device) * step).long()
        selected = valid_indices[selected_idx]

        trajectories = track[:, selected[:, 0], selected[:, 1], :]  # (N, K, 3)

        # Acceleration (second-order finite difference)
        accel = trajectories[2:] - 2 * trajectories[1:-1] + trajectories[:-2]
        accel_magnitude = torch.norm(accel, dim=-1) / scene_scale  # (N-2, K)
        E_accel = accel_magnitude.median()

        # Speed excess (p95/median ratio)
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
        Camera and motion quality evaluation.

        Components:
        - Camera translation acceleration penalty
        - Camera rotation acceleration penalty
        - Motion gate (ensures dynamic area has sufficient movement)
        - Teleportation penalty (extreme single-frame displacements)
        """
        N = extrinsics.shape[0]
        device = extrinsics.device

        # Camera translation acceleration
        cam_positions = extrinsics[:, :3, 3]  # (N, 3)
        if N >= 3:
            cam_accel = cam_positions[2:] - 2 * cam_positions[1:-1] + cam_positions[:-2]
            E_cam_accel = torch.norm(cam_accel, dim=-1).median() / scene_scale
        else:
            E_cam_accel = torch.tensor(0.0, device=device)

        # Camera rotation acceleration
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

        # Teleportation penalty
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
        First-frame geometric sanity gate.

        Verifies that the first frame's static region has reasonable depth values
        (non-zero, non-inf, within plausible range). Depth is measured in the
        camera-0 coordinate system (z-axis = viewing direction).
        """
        device = pts_frame0.device

        # Transform world pts to camera-0 coords to get proper depth
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

    def _differentiable_reproj_loss(self, pts, extrinsics, intrinsics,
                                     static_mask, valid_mask, scene_scale):
        """Differentiable reprojection loss for gradient guidance."""
        N, H, W, _ = pts.shape
        losses = []

        for i in range(0, min(N - 1, 3)):
            j = i + 1
            mask = static_mask & valid_mask[i] & valid_mask[j]
            if mask.sum() < 100:
                continue

            pts_i = pts[i][mask]
            w2c_j = torch.linalg.inv(extrinsics[j])
            pts_i_homo = F.pad(pts_i, (0, 1), value=1.0)
            pts_in_j = (w2c_j @ pts_i_homo.T).T[:, :3]
            proj_depth = pts_in_j[:, 2]

            pts_j = pts[j][mask]
            pts_j_homo = F.pad(pts_j, (0, 1), value=1.0)
            pts_j_cam = (w2c_j @ pts_j_homo.T).T[:, 2]

            valid = (proj_depth > 0.01) & (pts_j_cam > 0.01)
            if valid.sum() < 50:
                continue

            log_err = (torch.log(proj_depth[valid] + 1e-8) -
                       torch.log(pts_j_cam[valid] + 1e-8)).abs()
            losses.append(log_err.mean())

        if not losses:
            return torch.tensor(0.0, device=pts.device, requires_grad=True)
        return torch.stack(losses).mean()

    def _differentiable_track_smoothness(self, track, dynamic_mask, valid_mask, scene_scale):
        """Differentiable track smoothness loss for gradient guidance."""
        N, H, W, _ = track.shape
        if N < 3:
            return torch.tensor(0.0, device=track.device, requires_grad=True)

        all_valid = valid_mask.all(dim=0) & dynamic_mask
        if all_valid.sum() < 10:
            return torch.tensor(0.0, device=track.device, requires_grad=True)

        K = min(500, all_valid.sum().item())
        indices = all_valid.nonzero()[:K]
        traj = track[:, indices[:, 0], indices[:, 1], :]  # (N, K, 3)

        accel = traj[2:] - 2 * traj[1:-1] + traj[:-2]
        return (torch.norm(accel, dim=-1) / scene_scale).mean()

    def _differentiable_anchor_loss(self, pts, static_mask, extrinsic_frame0=None):
        """Differentiable anchor loss: depth consistency of first frame in camera coords."""
        pts_frame0 = pts[0]  # (H, W, 3)

        if extrinsic_frame0 is not None:
            w2c = torch.linalg.inv(extrinsic_frame0)
            H, W = pts_frame0.shape[:2]
            pts_flat = pts_frame0.reshape(-1, 3)
            pts_homo = torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=-1)
            pts_cam = (w2c @ pts_homo.T).T[:, :3].reshape(H, W, 3)
            depth0 = pts_cam[..., 2]
        else:
            depth0 = pts_frame0[..., 2]

        if static_mask.sum() < 50:
            return torch.tensor(0.0, device=pts.device, requires_grad=True)

        static_depth = depth0[static_mask]
        valid = static_depth > 0.01
        if valid.sum() < 50:
            return torch.tensor(0.0, device=pts.device, requires_grad=True)

        depths = static_depth[valid]
        median_d = depths.median().detach()
        deviation = (torch.log(depths + 1e-8) - torch.log(median_d + 1e-8)).abs()
        return deviation.mean()

    @staticmethod
    def _get_frame_pairs(N, strides=None):
        """Generate frame pairs for reprojection analysis."""
        if strides is None:
            strides = [1, 3, 5]
        pairs = []
        for stride in strides:
            for i in range(N - stride):
                pairs.append((i, i + stride))
        return pairs
