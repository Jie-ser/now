"""
Gradient guidance module for GeoReward V2 (Phase 3).

Applies differentiable geometric loss as guidance during the Wan2.2 denoising
process. The gradient target is explicit geometric consistency — NOT confidence.

Architecture:
  1. At selected denoising steps, compute pred_x0 from (latent, v_pred, sigma_t)
  2. VAE decode_differentiable(pred_x0) → video frames (gradient flows)
  3. 4RC forward on sampled frames (no @torch.no_grad wrapper)
  4. Compute geometric loss (L_reproj + L_track_smoothness + L_anchor)
     conf is detached as valid mask only — never backprop through confidence
  5. Backpropagate loss → grad w.r.t. pred_x0
  6. Apply WMReward-style normalization to modify v_pred

Integration:
  Called from denoise_candidates_with_guidance() in WanI2V, which invokes
  guidance.guided_v_pred() after computing noise_pred but before scheduler.step().

Memory management:
  Guidance requires 4RC + VAE on GPU simultaneously for the backward pass.
  The pipeline handles offloading DiT before guidance steps and reloading after.
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

    def __init__(self, model_4rc, vae, cfg=None, guidance_frames=8):
        """
        Args:
            model_4rc: 4RC (Arc) model, must allow gradient flow.
            vae: Wan2.2 VAE (Wan2_1_VAE instance with decode_differentiable).
            cfg: ReconRewardConfig with guidance parameters.
            guidance_frames: Number of frames to sample for guidance (fewer = faster).
        """
        self.model_4rc = model_4rc
        self.vae = vae
        self.cfg = cfg or ReconRewardConfig()
        self.guidance_frames = guidance_frames
        self.recon_reward = ReconstructionReward(model=model_4rc, cfg=self.cfg)

    def should_guide(self, sigma_t, step_idx):
        """Check whether guidance should be applied at this noise level and step."""
        if self.cfg.guidance_frequency <= 0:
            return False
        if not (self.cfg.sigma_min < sigma_t < self.cfg.sigma_max):
            return False
        if step_idx % self.cfg.guidance_frequency != 0:
            return False
        return True

    def guided_v_pred(self, latent, v_pred, sigma_t, step_idx):
        """
        Apply geometric guidance to v_pred.

        Called from the denoising loop after CFG but before scheduler.step().

        Args:
            latent: Current noisy latent (C, T, H, W) — single candidate.
            v_pred: Model's velocity prediction (C, T, H, W).
            sigma_t: Current noise level (scalar).
            step_idx: Current denoising step index.

        Returns:
            Modified v_pred with geometric guidance gradient applied,
            or original v_pred if guidance conditions not met or gradient fails.
        """
        if not self.should_guide(sigma_t, step_idx):
            return v_pred

        # Compute pred_x0: x0 = x_t - sigma_t * v_pred (flow matching formula)
        # Detach from the denoising graph — we build a fresh computational graph
        # from x0_hat through VAE decode → 4RC → loss
        x0_hat = (latent - sigma_t * v_pred).detach().requires_grad_(True)

        try:
            grad = self._compute_guidance_gradient(x0_hat)
        except Exception as e:
            warnings.warn(
                f"[GeometricGuidance] Gradient computation failed at step {step_idx}: {e}",
                stacklevel=2,
            )
            return v_pred

        if grad is None:
            return v_pred

        # WMReward-style normalization:
        # scale = guidance_scale * (||v_pred|| / ||grad||) * (1 - sigma_t^2)
        scaling_t = 1.0 - sigma_t ** 2
        norm_ratio = v_pred.norm(2) / (grad.norm(2) + 1e-8)
        v_guided = v_pred + self.cfg.guidance_scale * norm_ratio * scaling_t * grad

        return v_guided

    def _compute_guidance_gradient(self, x0_hat):
        """
        Full forward pass: VAE decode → sample frames → 4RC → geometric loss → grad.

        Bypasses 4RC's inference() wrapper (which has @torch.no_grad) and calls
        loss_of_one_batch() directly to preserve the gradient chain.

        Returns:
            Gradient tensor with same shape as x0_hat, or None on failure.
        """
        # VAE decode (differentiable path)
        video = self.vae.decode_differentiable(x0_hat)
        if video is None:
            return None

        # Sample frames uniformly
        # video shape: (3, T, H, W)
        T = video.shape[1]
        n = max(1, min(self.guidance_frames, T))
        indices = torch.linspace(0, T - 1, n).long()
        frames = video[:, indices]  # (3, n, H, W)

        # Prepare 4RC input views (differentiable)
        views = self._prepare_views(frames)

        # 4RC forward — bypass inference() which has @torch.no_grad.
        # Call loss_of_one_batch() directly with enable_grad.
        from arc.dust3r.inference_multiview import loss_of_one_batch
        from arc.dust3r.utils.device import collate_with_cat

        device = next(self.model_4rc.parameters()).device
        batch = collate_with_cat([tuple(views)])

        with torch.enable_grad():
            result = loss_of_one_batch(
                batch, self.model_4rc, None, device, "bf16-mixed",
            )

        preds = result["preds"]
        N_frames = len(preds)

        # Extract outputs (keeping gradient flow through pts and track)
        pts_list = []
        track_list = []
        ext_list = []
        int_list = []
        conf_list = []
        conf_track_list = []

        for i in range(N_frames):
            pred = preds[i]
            pts_list.append(pred["pts"].squeeze(0))
            track_list.append(pred["track"].squeeze(0))
            ext_list.append(pred["extrinsic"])
            int_list.append(pred["intrinsic"])
            conf_list.append(pred["conf"].squeeze(0))
            conf_track_list.append(pred["conf_track"].squeeze(0))

        pts = torch.stack(pts_list)
        track_abs = torch.stack(track_list)
        extrinsics = torch.stack(ext_list)
        intrinsics = torch.stack(int_list)
        conf = torch.stack(conf_list)
        conf_track = torch.stack(conf_track_list)

        # Convert track to relative displacement (same as fourrc_adapter)
        track = track_abs - pts[0].unsqueeze(0)

        # Compute masks (detached — no gradient through conf)
        with torch.no_grad():
            valid_geo, valid_track = compute_valid_mask(
                conf, conf_track, quantile=self.cfg.conf_valid_quantile
            )
            scene_scale = compute_scene_scale(pts, extrinsic_frame0=extrinsics[0])
            _, dynamic_mask = compute_dynamic_mask(
                track.detach(),
                threshold_ratio=self.cfg.dynamic_threshold_ratio,
                scene_scale=scene_scale,
            )

        # Compute differentiable loss
        structured_output = {
            "pts": pts,
            "track": track,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
        }

        loss = self.recon_reward.compute_differentiable_loss(
            structured_output, valid_geo, dynamic_mask, scene_scale
        )

        # Backprop to x0_hat
        grad = torch.autograd.grad(loss, x0_hat, retain_graph=False)[0]
        return grad

    def _prepare_views(self, frames):
        """
        Prepare frames as 4RC views while maintaining gradient flow.

        Args:
            frames: (3, N, H, W) tensor in [-1, 1].

        Returns:
            List of view dicts with 'img' as differentiable tensors.
        """
        import numpy as np

        N = frames.shape[1]
        target_size = max(self.cfg.image_size, 14)
        patch_size = 14

        views = []
        for i in range(N):
            frame = frames[:, i]  # (3, H, W)
            H_in, W_in = frame.shape[1], frame.shape[2]

            # Resize to target size (differentiable bilinear interpolation)
            scale = target_size / max(H_in, W_in)
            new_H = max(patch_size, int(round(H_in * scale)))
            new_W = max(patch_size, int(round(W_in * scale)))
            frame_resized = F.interpolate(
                frame.unsqueeze(0), size=(new_H, new_W),
                mode='bilinear', align_corners=False
            ).squeeze(0)  # (3, new_H, new_W)

            # Crop to patch-aligned size (center crop)
            cx, cy = new_W // 2, new_H // 2
            halfw = ((2 * cx) // patch_size) * patch_size // 2
            halfh = ((2 * cy) // patch_size) * patch_size // 2

            # Guard: ensure at least one patch (should not happen with the clamp above)
            if halfw < patch_size // 2 or halfh < patch_size // 2:
                halfw = max(halfw, patch_size // 2)
                halfh = max(halfh, patch_size // 2)
                # Ensure indices stay in bounds
                halfw = min(halfw, cx, new_W - cx)
                halfh = min(halfh, cy, new_H - cy)

            frame_cropped = frame_resized[
                :,
                cy - halfh: cy + halfh,
                cx - halfw: cx + halfw,
            ]  # (3, H_crop, W_crop)

            H_out, W_out = frame_cropped.shape[1], frame_cropped.shape[2]

            views.append({
                "img": frame_cropped.unsqueeze(0),  # (1, 3, H, W)
                "true_shape": np.int32([[H_out, W_out]]),
                "idx": i,
                "instance": str(i),
            })
        return views
