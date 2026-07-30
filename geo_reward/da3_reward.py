"""
DA3 Geometry Reward V1: Scene + Motion consistency.

Uses Depth Anything 3's explicit geometric outputs (depth, camera poses, confidence)
to score video physical consistency via:
- R_scene: bi-directional projection consistency on static regions only
- R_motion: motion_gate * (0.7 * R_shape + 0.3 * R_smoothness) on dynamic regions

Total: R_total = motion_gate * (0.30 * R_scene + 0.70 * motion_quality)
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from depth_anything_3.api import DepthAnything3

from .region_masks import build_static_dynamic_masks
from .motion_reward import compute_motion_reward, depth_to_world_points


@dataclass
class GeometryRewardConfig:
    # Region masks
    image_diff_threshold: float = 15.0
    image_vote_ratio: float = 0.30
    depth_change_threshold: float = 0.05
    min_component_area_ratio: float = 0.002
    morph_kernel_size: int = 3

    # Motion
    min_motion_threshold: float = 0.01
    smooth_quantile: float = 0.95
    tau_smooth: float = 2.0
    tau_shape: float = 0.10
    motion_shape_weight: float = 0.70
    motion_smooth_weight: float = 0.30

    # Scene
    tau_scene_proj: float = 0.05
    tau_scene_anchor: float = 0.05
    scene_proj_weight: float = 0.60
    scene_anchor_weight: float = 0.40
    scene_keep_ratio: float = 0.90

    # V1 total
    total_scene_weight: float = 0.30
    total_motion_weight: float = 0.70

    # Projection stride
    proj_stride: int = 4


def prepare_da3_aligned_images(frames_pil, pred):
    """
    Get RGB images aligned to DA3 output resolution.

    Uses pred.processed_images if available, otherwise resizes PIL frames
    to match depth map resolution.
    """
    if hasattr(pred, "processed_images") and pred.processed_images is not None:
        return pred.processed_images.astype(np.uint8)

    h, w = pred.depth.shape[-2:]
    images = []
    for frame in frames_pil:
        resized = frame.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        images.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(images, axis=0)


def robust_region_mean(values, keep_ratio=0.90):
    """
    Trimmed mean on finite values: keep the lowest keep_ratio fraction.

    Used for gentle outlier removal within the static region (boundary noise).
    """
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    values = np.sort(values)
    n_keep = max(1, int(len(values) * keep_ratio))
    return float(values[:n_keep].mean())


class DA3GeoReward:
    """
    Geometry-based reward V1: Scene + Motion.

    Computes:
    - R_scene: static background projection + anchor consistency (0-1)
    - R_motion: motion_gate * (shape + smoothness) for dynamic regions (0-1)
    - R_total: motion_gate * (0.30 * R_scene + 0.70 * motion_quality)
    """

    def __init__(self, model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1", device="cuda",
                 process_res=504, cfg=None):
        self.model = DepthAnything3.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.process_res = process_res
        self.cfg = cfg or GeometryRewardConfig()

    def compute_reward(self, frames_pil, task_spec=None):
        """
        Compute V1 geometry reward for a sequence of frames.

        Args:
            frames_pil: List of PIL Images (sampled keyframes, ~20 frames).
            task_spec: Optional task metadata (for V2 interaction, unused in V1).

        Returns:
            Dict with keys: total, version, scene, motion, motion_gate,
            shape, smoothness, motion_energy, scene_error, shape_error,
            smoothness_error, static_ratio, dynamic_ratio, unknown_ratio.
        """
        return self._compute_reward_v1(frames_pil)

    def _compute_reward_v1(self, frames_pil):
        pred = self.model.inference(frames_pil, process_res=self.process_res)

        depths = pred.depth           # (N, H, W)
        extrinsics = pred.extrinsics  # (N, 4, 4) or (N, 3, 4)
        intrinsics = pred.intrinsics  # (N, 3, 3)
        conf = pred.conf              # (N, H, W)

        if extrinsics is None or intrinsics is None:
            return self._fallback_result()

        images = prepare_da3_aligned_images(frames_pil, pred)

        masks = build_static_dynamic_masks(
            images=images,
            depths=depths,
            conf=conf,
            image_diff_threshold=self.cfg.image_diff_threshold,
            image_vote_ratio=self.cfg.image_vote_ratio,
            depth_change_threshold=self.cfg.depth_change_threshold,
            min_component_area_ratio=self.cfg.min_component_area_ratio,
            morph_kernel_size=self.cfg.morph_kernel_size,
        )

        scene = self._compute_scene_reward(
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            conf=conf,
            static_mask=masks["global_static"],
        )

        world_maps, point_clouds, scene_scale = self._build_dynamic_geometry(
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_dynamic_masks=masks["frame_dynamic"],
            global_static_mask=masks["global_static"],
        )

        motion = compute_motion_reward(
            dynamic_point_clouds=point_clouds,
            scene_scale=scene_scale,
            min_motion_threshold=self.cfg.min_motion_threshold,
            tau_smooth=self.cfg.tau_smooth,
            tau_shape=self.cfg.tau_shape,
            shape_weight=self.cfg.motion_shape_weight,
            smooth_weight=self.cfg.motion_smooth_weight,
        )

        motion_quality = (
            self.cfg.motion_shape_weight * motion["shape"]
            + self.cfg.motion_smooth_weight * motion["smoothness"]
        )

        total = motion["gate"] * (
            self.cfg.total_scene_weight * scene["reward"]
            + self.cfg.total_motion_weight * motion_quality
        )

        return {
            "total": float(total),
            "version": "v1",
            "scene": float(scene["reward"]),
            "motion": float(motion["reward"]),
            "motion_gate": float(motion["gate"]),
            "shape": float(motion["shape"]),
            "smoothness": float(motion["smoothness"]),
            "motion_energy": float(motion["motion_energy"]),
            "scene_proj_error": scene.get("proj_error"),
            "scene_anchor_error": scene.get("anchor_error"),
            "shape_error": motion["shape_error"],
            "smoothness_error": motion["smoothness_error"],
            "static_ratio": float(masks["global_static"].mean()),
            "dynamic_ratio": float(masks["global_dynamic"].mean()),
            "unknown_ratio": float(masks["global_unknown"].mean()),
        }

    def compute_reward_early(self, frames_pil):
        """
        Simplified reward for early checkpoint (high-noise pred_x0).

        Only uses motion_gate and R_scene — skips R_shape/R_smoothness which
        are unstable on noisy intermediate frames.

        Formula: early_score = motion_gate * (0.4 * R_scene + 0.6 * motion_gate)
        """
        pred = self.model.inference(frames_pil, process_res=self.process_res)

        depths = pred.depth
        extrinsics = pred.extrinsics
        intrinsics = pred.intrinsics
        conf = pred.conf

        if extrinsics is None or intrinsics is None:
            return self._fallback_result()

        images = prepare_da3_aligned_images(frames_pil, pred)

        masks = build_static_dynamic_masks(
            images=images,
            depths=depths,
            conf=conf,
            image_diff_threshold=self.cfg.image_diff_threshold,
            image_vote_ratio=self.cfg.image_vote_ratio,
            depth_change_threshold=self.cfg.depth_change_threshold,
            min_component_area_ratio=self.cfg.min_component_area_ratio,
            morph_kernel_size=self.cfg.morph_kernel_size,
        )

        scene = self._compute_scene_reward(
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            conf=conf,
            static_mask=masks["global_static"],
        )

        _, point_clouds, scene_scale = self._build_dynamic_geometry(
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_dynamic_masks=masks["frame_dynamic"],
            global_static_mask=masks["global_static"],
        )

        from .motion_reward import robust_centroid
        centroids = []
        for pc in point_clouds:
            if pc is not None and len(pc) > 0:
                c = robust_centroid(pc)
                if c is not None:
                    centroids.append(c)

        if len(centroids) >= 2 and scene_scale > 0:
            centroids_arr = np.array(centroids)
            displacements = np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)
            speeds = displacements / scene_scale
            median_speed = float(np.median(speeds))
            motion_gate = min(1.0, median_speed / (self.cfg.min_motion_threshold + 1e-8))
        else:
            motion_gate = 0.0

        total = motion_gate * (0.4 * scene["reward"] + 0.6 * motion_gate)

        return {
            "total": float(total),
            "version": "v1_early",
            "scene": float(scene["reward"]),
            "motion_gate": float(motion_gate),
            "motion": 0.0,
            "shape": 0.0,
            "smoothness": 0.0,
            "motion_energy": float(motion_gate),
            "scene_proj_error": scene.get("proj_error"),
            "scene_anchor_error": scene.get("anchor_error"),
            "shape_error": None,
            "smoothness_error": None,
            "static_ratio": float(masks["global_static"].mean()),
            "dynamic_ratio": float(masks["global_dynamic"].mean()),
            "unknown_ratio": float(masks["global_unknown"].mean()),
        }

    def _compute_scene_reward(self, depths, extrinsics, intrinsics, conf, static_mask):
        """
        Compute R_scene: projection + anchor consistency on static regions only.

        Returns dict with reward, proj_error, anchor_error.
        """
        r_scene_proj, proj_error = self._static_projection_consistency(
            depths, extrinsics, intrinsics, conf, static_mask
        )
        r_scene_anchor, anchor_error = self._static_anchor_consistency(
            depths, extrinsics, intrinsics, conf, static_mask
        )

        reward = (
            self.cfg.scene_proj_weight * r_scene_proj
            + self.cfg.scene_anchor_weight * r_scene_anchor
        )

        return {
            "reward": float(reward),
            "proj": float(r_scene_proj),
            "anchor": float(r_scene_anchor),
            "proj_error": proj_error,
            "anchor_error": anchor_error,
        }

    def _static_projection_consistency(self, depths, extrinsics, intrinsics, conf, static_mask):
        """
        Bi-directional projection consistency evaluated only on static pixels.

        Maps error to [0, 1] via exp(-E / tau_scene_proj).
        """
        N, H, W = depths.shape
        stride = self.cfg.proj_stride
        total_error = 0.0
        count = 0

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels_flat = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=0)

        static_flat = static_mask.reshape(-1)

        for t in range(0, N - stride, stride):
            s = t + stride

            err_forward = self._project_error_static(
                depths[t], depths[s],
                extrinsics[t], extrinsics[s],
                intrinsics[t], intrinsics[s],
                conf, t, s,
                pixels_flat, H, W,
                static_mask, static_flat,
            )

            err_backward = self._project_error_static(
                depths[s], depths[t],
                extrinsics[s], extrinsics[t],
                intrinsics[s], intrinsics[t],
                conf, s, t,
                pixels_flat, H, W,
                static_mask, static_flat,
            )

            if err_forward is not None and err_backward is not None:
                total_error += (err_forward + err_backward) / 2
                count += 1
            elif err_forward is not None:
                total_error += err_forward
                count += 1
            elif err_backward is not None:
                total_error += err_backward
                count += 1

        if count == 0:
            return 0.0, None

        mean_error = total_error / count
        r_proj = float(np.exp(-mean_error / self.cfg.tau_scene_proj))
        return r_proj, float(mean_error)

    def _project_error_static(self, depth_src, depth_tgt, ext_src, ext_tgt,
                              intr_src, intr_tgt, conf, src_idx, tgt_idx,
                              pixels_flat, H, W, static_mask, static_flat):
        """
        Single-direction projection error, filtered to static regions.

        Returns trimmed mean error (scalar) or None.
        """
        K_src_inv = np.linalg.inv(intr_src)
        rays = K_src_inv @ pixels_flat
        pts_cam_src = rays * depth_src.reshape(-1)[None, :]

        R_src = ext_src[:3, :3]
        t_src = ext_src[:3, 3]
        pts_world = R_src.T @ (pts_cam_src - t_src[:, None])

        R_tgt = ext_tgt[:3, :3]
        t_tgt = ext_tgt[:3, 3]
        pts_cam_tgt = R_tgt @ pts_world + t_tgt[:, None]

        proj = intr_tgt @ pts_cam_tgt
        px = proj[0] / (proj[2] + 1e-8)
        py = proj[1] / (proj[2] + 1e-8)
        depth_projected = pts_cam_tgt[2]

        depth_sampled = self._bilinear_sample(depth_tgt, px, py)

        # Static mask filtering: source pixel must be static
        # Target landing position must also be in static region
        target_static = self._bilinear_sample(
            static_mask.astype(np.float32), px, py
        ) > 0.5

        # Confidence filtering via quantile
        conf_valid = np.ones(H * W, dtype=bool)
        if conf is not None:
            conf_sampled = self._bilinear_sample(conf[tgt_idx], px, py)
            conf_threshold = np.quantile(conf[tgt_idx][np.isfinite(conf[tgt_idx])], 0.5)
            conf_valid = conf_sampled > conf_threshold

        valid = (
            (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
            & (depth_projected > 1e-3) & (depth_sampled > 1e-3)
            & static_flat & target_static & conf_valid
        )

        if valid.sum() < 100:
            return None

        ratio = depth_projected[valid] / depth_sampled[valid]
        scale = np.median(ratio)
        if scale < 1e-6:
            return None
        aligned = depth_projected[valid] / scale

        log_err = np.abs(np.log(aligned / (depth_sampled[valid] + 1e-8) + 1e-8))

        return robust_region_mean(log_err, keep_ratio=self.cfg.scene_keep_ratio)

    def _static_anchor_consistency(self, depths, extrinsics, intrinsics, conf, static_mask):
        """
        First-frame anchor consistency on static mask only.

        Maps error to [0, 1] via exp(-E / tau_scene_anchor).
        """
        N, H, W = depths.shape
        if N < 2:
            return 0.0, None

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels_flat = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=0)

        static_flat = static_mask.reshape(-1)

        K0_inv = np.linalg.inv(intrinsics[0])
        rays_0 = K0_inv @ pixels_flat
        pts_cam_0 = rays_0 * depths[0].reshape(-1)[None, :]
        R_0 = extrinsics[0, :3, :3]
        t_0 = extrinsics[0, :3, 3]
        pts_world_0 = R_0.T @ (pts_cam_0 - t_0[:, None])

        errors = []
        sample_step = max(1, N // 10)
        for t in range(1, N, sample_step):
            R_t = extrinsics[t, :3, :3]
            t_t = extrinsics[t, :3, 3]
            pts_cam_t = R_t @ pts_world_0 + t_t[:, None]

            proj_t = intrinsics[t] @ pts_cam_t
            px = proj_t[0] / (proj_t[2] + 1e-8)
            py = proj_t[1] / (proj_t[2] + 1e-8)
            depth_proj = pts_cam_t[2]

            depth_actual = self._bilinear_sample(depths[t], px, py)

            target_static = self._bilinear_sample(
                static_mask.astype(np.float32), px, py
            ) > 0.5

            valid = (
                (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
                & (depth_proj > 1e-3) & (depth_actual > 1e-3)
                & static_flat & target_static
            )

            if valid.sum() < 100:
                continue

            ratio = depth_proj[valid] / depth_actual[valid]
            scale = np.median(ratio)
            if scale < 1e-6:
                continue
            deviation = np.abs(np.log(ratio / scale + 1e-8))

            err = robust_region_mean(deviation, keep_ratio=self.cfg.scene_keep_ratio)
            if err is not None:
                errors.append(err)

        if len(errors) == 0:
            return 0.0, None

        mean_error = float(np.mean(errors))
        r_anchor = float(np.exp(-mean_error / self.cfg.tau_scene_anchor))
        return r_anchor, mean_error

    def _build_dynamic_geometry(self, depths, extrinsics, intrinsics,
                                frame_dynamic_masks, global_static_mask):
        """
        Build per-frame dynamic point clouds in world coordinates.

        Returns:
            world_maps: list of (H, W, 3) arrays
            point_clouds: list of (M_t, 3) arrays (dynamic points only)
            scene_scale: float (median depth of first frame's static region)
        """
        N = len(depths)
        world_maps = []
        point_clouds = []

        for t in range(N):
            world = depth_to_world_points(
                depths[t],
                extrinsics[t],
                intrinsics[t],
            )
            world_maps.append(world)

            valid = (
                frame_dynamic_masks[t]
                & np.isfinite(world).all(axis=-1)
                & (depths[t] > 1e-6)
            )
            point_clouds.append(world[valid])

        # Scene scale: median depth in first frame's static region
        static_depths = depths[0][global_static_mask]
        if len(static_depths) > 0:
            scene_scale = max(float(np.median(static_depths)), 1e-6)
        else:
            scene_scale = max(float(np.median(depths[0])), 1e-6)

        return world_maps, point_clouds, scene_scale

    def _fallback_result(self):
        """Return zero-reward result when DA3 outputs are unavailable."""
        return {
            "total": 0.0,
            "version": "v1",
            "scene": 0.0,
            "motion": 0.0,
            "motion_gate": 0.0,
            "shape": 0.0,
            "smoothness": 0.0,
            "motion_energy": 0.0,
            "scene_proj_error": None,
            "scene_anchor_error": None,
            "shape_error": None,
            "smoothness_error": None,
            "static_ratio": 0.0,
            "dynamic_ratio": 0.0,
            "unknown_ratio": 0.0,
        }

    @staticmethod
    def _bilinear_sample(image, x, y):
        """Bilinear interpolation sampling on a 2D numpy array."""
        H, W = image.shape
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        x0c = np.clip(x0, 0, W - 1)
        x1c = np.clip(x1, 0, W - 1)
        y0c = np.clip(y0, 0, H - 1)
        y1c = np.clip(y1, 0, H - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return (wa * image[y0c, x0c] + wb * image[y0c, x1c]
                + wc * image[y1c, x0c] + wd * image[y1c, x1c])
