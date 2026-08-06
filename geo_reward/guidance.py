"""
Gradient guidance module for GeoReward V2 (Phase 3).

Applies differentiable geometric loss as guidance during the Wan2.2 denoising
process. The gradient target is explicit geometric consistency — NOT confidence.

Architecture:
  1. Extract pred_x0 at each guidance step
  2. VAE decode (differentiable path)
  3. 4RC forward (no @torch.no_grad wrapper)
  4. Compute geometric loss (conf detached as valid mask only)
  5. Backpropagate through latent → modify v_pred

KNOWN LIMITATIONS (must be resolved before activation):
  - Requires a custom `vae.decode_differentiable()` method that does NOT wrap
    in torch.no_grad(). Wan2.2's standard decode uses no_grad and returns a list,
    which breaks both gradient flow and the expected (B,C,T,H,W) tensor interface.
  - Not yet integrated into GeoRewardBoNProgressiveV2 or Wan's denoising loop.
  - This module is a standalone framework awaiting Phase 1/2 validation.

Note: This module requires Phase 1/2 validation to confirm reward signal
effectiveness before activation.
"""

import warnings

import torch
import torch.nn.functional as F

from .fourrc_adapter import compute_valid_mask, compute_dynamic_mask, compute_scene_scale
from .recon_reward import ReconRewardConfig, ReconstructionReward


class GeometricGuidance:
    """
    Gradient-based geometric guidance for denoising.

    Modifies v_pred at selected steps to steer latents toward
    geometrically consistent video generation.
    """

    def __init__(self, model_4rc, vae=None, cfg=None, recon_reward=None):
        """
        Args:
            model_4rc: 4RC (Arc) model, must allow gradient flow.
            vae: Wan2.2 VAE decoder (must support differentiable decode).
            cfg: ReconRewardConfig with guidance parameters.
            recon_reward: ReconstructionReward instance for computing
                          the full differentiable loss. If None, one is
                          created internally with the given cfg.
        """
        self.model_4rc = model_4rc
        self.vae = vae
        self.cfg = cfg or ReconRewardConfig()
        self.recon_reward = recon_reward or ReconstructionReward(
            model=model_4rc, cfg=self.cfg
        )
        self.step_count = 0

    def should_guide(self, sigma_t):
        """Check whether guidance should be applied at this noise level and step."""
        if not (self.cfg.sigma_min < sigma_t < self.cfg.sigma_max):
            return False
        if self.step_count % self.cfg.guidance_frequency != 0:
            return False
        return True

    def guided_denoise_step(self, latent, v_pred, sigma_t, step_idx):
        """
        Apply geometric guidance to v_pred if conditions are met.

        Args:
            latent: Current noisy latent (B, C, T, H, W).
            v_pred: Model's velocity prediction.
            sigma_t: Current noise level.
            step_idx: Current denoising step index.

        Returns:
            Modified v_pred with geometric guidance gradient applied.
        """
        self.step_count = step_idx

        if not self.should_guide(sigma_t):
            return v_pred

        if self.vae is None or self.model_4rc is None:
            return v_pred

        x0_hat = (latent - sigma_t * v_pred).detach().requires_grad_(True)

        video = self._differentiable_decode(x0_hat)
        if video is None:
            return v_pred

        frames = self._sample_fixed_frames(video, n=8)
        views = self._prepare_views_differentiable(frames)

        raw_output = self.model_4rc(views, force_no_output_conversion=True)

        # Raw forward output keys (with force_no_output_conversion=True):
        # "depth" (B,N,H,W), "depth_conf" (B,N,H,W),
        # "track" (B,N,H,W,3), "conf_track" (B,N,H,W),
        # "pose_enc" — raw camera encoding (NOT yet decoded to extrinsics/intrinsics)
        # Note: "pts", "extrinsics_token", "intrinsics_token" do NOT exist —
        # those are computed in _postprocess_output which we skipped.

        # Decode pose_enc → extrinsics/intrinsics (differentiable)
        from arc.models.arc.utils.transform import pose_encoding_to_extri_intri, affine_inverse
        depth = raw_output["depth"]  # (B, N, H, W)
        B_raw, N_raw, H_raw, W_raw = depth.shape
        pose_enc = raw_output["pose_enc"]
        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H_raw, W_raw))
        w2c = affine_inverse(c2w)  # (B, N, 4, 4) — world-to-camera

        # Compute world points from depth + camera params (differentiable)
        # Unproject: pixel (u,v) → camera ray → scale by depth → world coords
        device = depth.device
        u_grid, v_grid = torch.meshgrid(
            torch.arange(W_raw, device=device, dtype=depth.dtype),
            torch.arange(H_raw, device=device, dtype=depth.dtype),
            indexing='xy',
        )
        ones = torch.ones_like(u_grid)
        pixels = torch.stack([u_grid, v_grid, ones], dim=-1)  # (H, W, 3)

        # For simplicity, use batch=0 only
        depth_b = depth[0]  # (N, H, W)
        w2c_b = w2c[0]      # (N, 4, 4)
        ixt_b = ixt[0]      # (N, 3, 3)
        track_raw = raw_output.get("track")
        if track_raw is not None:
            track_b = track_raw[0]  # (N, H, W, 3)
        else:
            track_b = None

        # Unproject to world points per frame
        pts_list = []
        for t in range(N_raw):
            K_inv = torch.linalg.inv(ixt_b[t])  # (3, 3)
            rays = (K_inv @ pixels.reshape(-1, 3).T).T  # (H*W, 3)
            pts_cam = rays * depth_b[t].reshape(-1, 1)  # (H*W, 3)
            # cam → world: p_world = R^T @ (p_cam - t) for w2c = [R|t]
            # Or: p_world = c2w @ [p_cam; 1]
            c2w_t = torch.linalg.inv(w2c_b[t])  # (4, 4)
            pts_homo = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=-1)
            pts_world = (c2w_t @ pts_homo.T).T[:, :3]  # (H*W, 3)
            pts_list.append(pts_world.reshape(H_raw, W_raw, 3))

        pts_tensor = torch.stack(pts_list)  # (N, H, W, 3)
        ext_tensor = w2c_b  # (N, 4, 4) — but compute_differentiable_loss expects c2w
        # Convert back to c2w for consistency with ReconstructionReward
        c2w_tensor = c2w[0]  # (N, 4, 4)
        int_tensor = ixt_b   # (N, 3, 3)

        with torch.no_grad():
            depth_conf = raw_output.get("depth_conf")
            conf_track_raw = raw_output.get("conf_track")

            if depth_conf is not None and depth_conf.dim() == 4:
                depth_conf = depth_conf[0]  # (N, H, W)
            if conf_track_raw is not None and conf_track_raw.dim() == 4:
                conf_track_raw = conf_track_raw[0]

            if depth_conf is not None and conf_track_raw is not None:
                valid_geo, _ = compute_valid_mask(depth_conf, conf_track_raw)
            else:
                valid_geo = torch.ones(N_raw, H_raw, W_raw, dtype=torch.bool, device=device)

            if track_b is not None:
                scene_scale = depth_b[0].median().clamp(min=1e-6).item()
                _, dyn_mask = compute_dynamic_mask(track_b, scene_scale=scene_scale)
            else:
                dyn_mask = torch.zeros(H_raw, W_raw, dtype=torch.bool, device=device)
                scene_scale = 1.0

        structured_output = {
            "pts": pts_tensor,
            "track": track_b if track_b is not None else torch.zeros(N_raw, H_raw, W_raw, 3, device=device),
            "extrinsic": c2w_tensor,
            "intrinsic": int_tensor,
        }

        loss = self.recon_reward.compute_differentiable_loss(
            structured_output, valid_geo, dyn_mask, scene_scale
        )

        try:
            grad = torch.autograd.grad(loss, x0_hat, retain_graph=False)[0]
        except RuntimeError as e:
            warnings.warn(
                f"Gradient guidance failed (likely broken grad chain): {e}. "
                "Ensure VAE has decode_differentiable() that preserves gradients.",
                stacklevel=2,
            )
            return v_pred

        scaling_t = 1.0 - sigma_t ** 2
        norm_ratio = v_pred.norm(2) / (grad.norm(2) + 1e-8)
        v_guided = v_pred + self.cfg.guidance_scale * norm_ratio * scaling_t * grad

        return v_guided

    def _differentiable_decode(self, x0_hat):
        """
        Differentiable VAE decode.

        Requires vae.decode_differentiable() — a custom path that does NOT
        use torch.no_grad(). Standard Wan2.2 VAE decode wraps in no_grad
        and will break the gradient chain from x0_hat through to the loss.
        """
        try:
            if hasattr(self.vae, 'decode_differentiable'):
                return self.vae.decode_differentiable(x0_hat)
            else:
                warnings.warn(
                    "VAE has no decode_differentiable() method. "
                    "Standard decode() likely breaks gradient flow. "
                    "Gradient guidance will be a no-op until this is implemented.",
                    stacklevel=2,
                )
                return self.vae.decode(x0_hat)
        except Exception as e:
            warnings.warn(
                f"Differentiable VAE decode failed: {e}. "
                "Gradient guidance skipped for this step.",
                stacklevel=2,
            )
            return None

    def _sample_fixed_frames(self, video, n=8):
        """Sample n uniformly spaced frames from decoded video tensor."""
        # video: (B, 3, T, H, W) or (3, T, H, W)
        if video.dim() == 5:
            video = video[0]
        T = video.shape[1]
        indices = torch.linspace(0, T - 1, n).long()
        return video[:, indices]  # (3, n, H, W)

    def _prepare_views_differentiable(self, frames):
        """
        Prepare frames as 4RC views while maintaining gradient flow.

        Args:
            frames: (3, N, H, W) tensor in [-1, 1].

        Returns:
            List of view dicts with 'img' as differentiable tensors.
        """
        N = frames.shape[1]
        views = []
        for i in range(N):
            frame = frames[:, i]  # (3, H, W)
            H, W = frame.shape[1], frame.shape[2]
            views.append({
                "img": frame.unsqueeze(0),  # (1, 3, H, W)
                "true_shape": torch.tensor([[H, W]], dtype=torch.int32),
                "idx": i,
                "instance": str(i),
            })
        return views

