"""
DA3 Geometry Reward: Bi-directional Depth Projection Consistency.

Uses Depth Anything 3's explicit geometric outputs (depth, camera poses, confidence)
to score video physical consistency via cross-frame 3D projection.
"""

import numpy as np
from depth_anything_3.api import DepthAnything3


class DA3GeoReward:
    """
    Geometry-based reward using Depth Anything 3.

    Computes three reward components:
    - Projection consistency (50%): bi-directional depth reprojection error
    - Anchor consistency (35%): first-frame 3D structure stability
    - Confidence score (15%): DA3 prediction confidence
    """

    def __init__(self, model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1", device="cuda",
                 process_res=504):
        self.model = DepthAnything3.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.process_res = process_res

    def compute_reward(self, frames_pil, stride=4, keep_ratio=0.7):
        """
        Compute geometry reward for a sequence of frames.

        Args:
            frames_pil: List of PIL Images (sampled keyframes, ~20 frames).
            stride: Frame interval for projection consistency check.
            keep_ratio: Fraction of pixels to keep after sorting by error.

        Returns:
            Dict with keys: "total", "proj", "anchor", "conf"
        """
        pred = self.model.inference(frames_pil, process_res=self.process_res)

        depths = pred.depth           # (N, H, W)
        extrinsics = pred.extrinsics  # (N, 3, 4) world-to-camera [R|t]
        intrinsics = pred.intrinsics  # (N, 3, 3)
        conf = pred.conf              # (N, H, W)

        if extrinsics is None or intrinsics is None:
            return {"total": 0.0, "proj": 0.0, "anchor": 0.0, "conf": float(conf.mean()) if conf is not None else 0.0}

        r_proj = self._projection_consistency(depths, extrinsics, intrinsics, conf, stride, keep_ratio)
        r_anchor = self._anchor_consistency(depths, extrinsics, intrinsics, conf)
        r_conf = self._confidence_score(conf)

        total = 0.50 * r_proj + 0.35 * r_anchor + 0.15 * r_conf
        return {"total": total, "proj": r_proj, "anchor": r_anchor, "conf": r_conf}

    def _projection_consistency(self, depths, extrinsics, intrinsics, conf, stride,
                                keep_ratio=0.7):
        """
        Bi-directional depth projection consistency.

        For frame pairs (t, s=t+stride):
          1. Forward (t->s): unproject t to 3D, project into s, compare depth
          2. Backward (s->t): unproject s to 3D, project into t, compare depth
          3. Average the truncated mean errors from both directions

        keep_ratio: fraction of pixels to keep after sorting by error (ascending).
                    Implicitly excludes moving regions (robot arm) whose projection
                    errors are large.
        """
        N, H, W = depths.shape
        total_error = 0.0
        count = 0

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels_flat = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=0)  # (3, H*W)

        for t in range(0, N - stride, stride):
            s = t + stride

            err_forward = self._project_and_compare(
                depths[t], depths[s],
                extrinsics[t], extrinsics[s],
                intrinsics[t], intrinsics[s],
                conf[s], pixels_flat, H, W, keep_ratio
            )

            err_backward = self._project_and_compare(
                depths[s], depths[t],
                extrinsics[s], extrinsics[t],
                intrinsics[s], intrinsics[t],
                conf[t], pixels_flat, H, W, keep_ratio
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
            return 0.0
        return -total_error / count

    def _project_and_compare(self, depth_src, depth_tgt, ext_src, ext_tgt,
                             intr_src, intr_tgt, conf_tgt, pixels_flat, H, W,
                             keep_ratio=0.7):
        """
        Single-direction projection comparison: unproject src pixels to 3D,
        project into tgt frame, compare projected depth with tgt actual depth.

        Returns truncated mean error (scalar), or None if insufficient valid pixels.
        """
        K_src_inv = np.linalg.inv(intr_src)  # (3, 3)
        rays = K_src_inv @ pixels_flat  # (3, H*W)
        pts_cam_src = rays * depth_src.reshape(-1)[None, :]  # (3, H*W)

        # Camera -> world: extrinsics is [R|t] (3x4), P_cam = R @ P_world + t
        R_src = ext_src[:3, :3]
        t_src = ext_src[:3, 3]
        pts_world = R_src.T @ (pts_cam_src - t_src[:, None])  # (3, H*W)

        # World -> target camera
        R_tgt = ext_tgt[:3, :3]
        t_tgt = ext_tgt[:3, 3]
        pts_cam_tgt = R_tgt @ pts_world + t_tgt[:, None]  # (3, H*W)

        # Project to target pixel coordinates
        proj = intr_tgt @ pts_cam_tgt  # (3, H*W)
        px = proj[0] / (proj[2] + 1e-8)
        py = proj[1] / (proj[2] + 1e-8)
        depth_projected = pts_cam_tgt[2]

        # Bilinear sample from target depth/conf maps
        depth_sampled = self._bilinear_sample(depth_tgt, px, py)
        conf_sampled = self._bilinear_sample(conf_tgt, px, py) if conf_tgt is not None else np.ones_like(px)

        # Valid pixel mask
        valid = (
            (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
            & (depth_projected > 1e-3) & (depth_sampled > 1e-3)
            & (conf_sampled > 0.3)
        )

        if valid.sum() < 100:
            return None

        # Scale alignment (median ratio)
        ratio = depth_projected[valid] / depth_sampled[valid]
        scale = np.median(ratio)
        if scale < 1e-6:
            return None
        aligned = depth_projected[valid] / scale

        # Log-ratio error
        log_err = np.abs(np.log(aligned / (depth_sampled[valid] + 1e-8) + 1e-8))

        # Truncated mean: keep only the lowest-error fraction of pixels
        sorted_err = np.sort(log_err)
        n_keep = int(len(sorted_err) * keep_ratio)
        if n_keep < 10:
            return None
        truncated_err = sorted_err[:n_keep].mean()

        return truncated_err

    def _anchor_consistency(self, depths, extrinsics, intrinsics, conf):
        """
        First-frame anchor consistency: static regions should maintain
        consistent 3D structure relative to the first frame.
        """
        N, H, W = depths.shape
        if N < 2:
            return 0.0

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels_flat = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=0)  # (3, H*W)

        # First frame 3D point cloud in world coordinates
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

            valid = (
                (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
                & (depth_proj > 1e-3) & (depth_actual > 1e-3)
            )

            if valid.sum() < 100:
                continue

            ratio = depth_proj[valid] / depth_actual[valid]
            scale = np.median(ratio)
            if scale < 1e-6:
                continue
            deviation = np.abs(np.log(ratio / scale + 1e-8))

            # Take the 70% least-deviating pixels (assumed static region)
            sorted_dev = np.sort(deviation)
            n_static = int(len(sorted_dev) * 0.7)
            if n_static > 0:
                errors.append(sorted_dev[:n_static].mean())

        if len(errors) == 0:
            return 0.0
        return -np.mean(errors)

    def _confidence_score(self, conf):
        """DA3 confidence mean as a quality indicator."""
        if conf is None:
            return 0.5
        return float(conf.mean())

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
