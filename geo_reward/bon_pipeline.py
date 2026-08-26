"""
Best-of-N sampling pipeline with GeoReward V2 (4RC).

Generates N candidate videos with Wan2.2 I2V, scores each with 4RC
reconstruction quality reward, and selects the geometrically most consistent one.

Includes:
- GeoRewardBoNProgressive: base progressive elimination BoN (σ-based checkpoints)
- GeoRewardBoNProgressiveV2: progressive elimination with 4RC reward + model offloading
- GeoRewardBoNProgressiveV2Guided: adds gradient guidance during denoising
- GeoRewardBoNTreeBranching: tree branching + progressive elimination (shared trunk)
"""

import json
import os
import random
import time

import numpy as np
import torch

from .utils import wan_output_to_pil, sample_frames


class GeoRewardBoNProgressive:
    """
    Base class for progressive elimination Best-of-N pipeline.

    Provides the shared framework: seed management, sigma-based checkpoint
    scheduling, fixed-ratio elimination logic, and video saving utilities.

    Subclasses must implement _generate_prepared() with their specific
    reward scoring logic.
    """

    DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]
    DEFAULT_ELIMINATION_RATIO = 0.5
    DEFAULT_MIN_SURVIVORS = 2
    DEFAULT_SCORE_EPSILON = 0.02
    DEFAULT_EARLY_MAX_FRAMES = 12

    def __init__(
        self,
        wan_i2v,
        frame_indices=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
    ):
        self.wan = wan_i2v
        self.frame_indices = frame_indices
        self.max_frames = max_frames
        self.sigma_checkpoints = sigma_checkpoints or self.DEFAULT_SIGMA_CHECKPOINTS
        self.elimination_ratio = (
            elimination_ratio if elimination_ratio is not None
            else self.DEFAULT_ELIMINATION_RATIO
        )
        self.min_survivors = (
            min_survivors if min_survivors is not None
            else self.DEFAULT_MIN_SURVIVORS
        )
        self.score_epsilon = (
            score_epsilon if score_epsilon is not None
            else self.DEFAULT_SCORE_EPSILON
        )
        self.early_max_frames = (
            early_max_frames if early_max_frames is not None
            else self.DEFAULT_EARLY_MAX_FRAMES
        )

    def generate(
        self,
        prompt,
        image,
        N=8,
        frame_num=81,
        seed_base=None,
        output_dir=None,
        save_fn=None,
        **wan_kwargs,
    ):
        """
        Two-stage fixed-ratio progressive elimination BoN.

        Args:
            prompt: text prompt.
            image: PIL Image (first frame).
            N: number of initial candidates.
            frame_num: video length in frames.
            seed_base: base seed (candidates use seed_base+i).
            output_dir: if set, save all checkpoint and final videos here.
            save_fn: callable(tensor, path) to write a video to disk.
                     If None, videos are saved as .pt files.
            **wan_kwargs: forwarded to WanI2V.prepare_progressive.

        Returns:
            (best_video_tensor, all_rewards_log, best_seed)
        """
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}.")

        validated_sigmas = []
        for value in self.sigma_checkpoints:
            sigma = float(value)
            if not np.isfinite(sigma) or not 0.0 < sigma < 1.0:
                raise ValueError(
                    "Each sigma checkpoint must be finite and strictly "
                    f"between 0 and 1, got {value}."
                )
            validated_sigmas.append(sigma)

        if self.frame_indices is not None:
            indices = self.frame_indices
        else:
            indices = sample_frames(frame_num, self.max_frames)

        early_indices = sample_frames(frame_num, self.early_max_frames)

        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        seeds = [seed_base + i for i in range(N)]

        print(f"[BoNProgressive] Preparing {N} candidates (seeds {seeds[0]}..{seeds[-1]})")
        print(f"[BoNProgressive] Checkpoints: σ={validated_sigmas}, "
              f"elimination_ratio={self.elimination_ratio}, "
              f"min_survivors={self.min_survivors}, "
              f"epsilon={self.score_epsilon}")
        state = self.wan.prepare_progressive(
            input_prompt=prompt,
            img=image,
            seeds=seeds,
            frame_num=frame_num,
            **wan_kwargs,
        )
        try:
            return self._generate_prepared(
                state=state,
                seeds=seeds,
                indices=indices,
                early_indices=early_indices,
                sigma_checkpoints=validated_sigmas,
                output_dir=output_dir,
                save_fn=save_fn,
            )
        finally:
            self.wan.cleanup_progressive(
                state, offload_model=state.get('offload_model', True))

    def _generate_prepared(
        self,
        state,
        seeds,
        indices,
        early_indices,
        sigma_checkpoints,
        output_dir,
        save_fn,
    ):
        """
        Run progressive denoising with reward-based elimination.

        Subclasses must implement this method with their specific reward logic.
        """
        raise NotImplementedError(
            "Subclasses must implement _generate_prepared() with their reward logic."
        )

    def _eliminate(self, scored, seeds, eliminated_at, phase_name, is_early):
        """
        Fixed-ratio elimination: drop bottom elimination_ratio candidates.

        Safety: if the score gap between the last survivor and first eliminated
        is smaller than score_epsilon, keep one extra candidate.
        Always keep at least min_survivors alive.
        """
        if len(scored) <= self.min_survivors:
            return [c for c, _ in scored], []

        totals = np.array([
            float(r.get('total', float('nan'))) for _, r in scored
        ], dtype=np.float64)
        finite = np.isfinite(totals)

        if not finite.any():
            print(
                f"  WARNING: no finite rewards at {phase_name}; "
                "skipping elimination."
            )
            return [c for c, _ in scored], []

        ranked = sorted(
            range(len(scored)),
            key=lambda i: totals[i] if np.isfinite(totals[i]) else -float("inf"),
            reverse=True,
        )

        keep_count = max(
            self.min_survivors,
            len(scored) - int(len(scored) * self.elimination_ratio),
        )

        if keep_count < len(scored):
            last_keep_score = totals[ranked[keep_count - 1]]
            first_elim_score = totals[ranked[keep_count]]
            if (np.isfinite(last_keep_score) and np.isfinite(first_elim_score)
                    and (last_keep_score - first_elim_score) < self.score_epsilon):
                keep_count = min(keep_count + 1, len(scored))

        survivors = []
        eliminated = []
        survivor_set = set(ranked[:keep_count])
        for idx, (cand_idx, _) in enumerate(scored):
            if idx in survivor_set:
                survivors.append(cand_idx)
            else:
                eliminated.append(cand_idx)
                eliminated_at[f"seed_{seeds[cand_idx]}"] = phase_name

        if eliminated:
            elim_seeds = [seeds[c] for c in eliminated]
            surv_seeds = [seeds[c] for c in survivors]
            print(f"  Eliminated {len(eliminated)} candidates: seeds {elim_seeds} "
                  f"(kept {len(survivors)}: seeds {surv_seeds})")

        return survivors, eliminated

    def _save_video(self, video_tensor, seed, phase_name, output_dir, save_fn):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"seed_{seed}_{phase_name}.mp4"
        path = os.path.join(output_dir, filename)
        if save_fn is not None:
            save_fn(video_tensor, path)
        else:
            pt_path = path.replace('.mp4', '.pt')
            torch.save(video_tensor, pt_path)

    def _build_result_log(self, seeds, rewards_log, eliminated_at,
                          best_seed, sigma_checkpoints, elapsed,
                          checkpoint_steps):
        return {
            "mode": "progressive_elimination_v2",
            "sigma_checkpoints": sigma_checkpoints,
            "elimination_ratio": self.elimination_ratio,
            "min_survivors": self.min_survivors,
            "score_epsilon": self.score_epsilon,
            "early_max_frames": self.early_max_frames,
            "checkpoint_plan": [
                {
                    "completed_steps": item["end_step"],
                    "scheduler_step_index": item["scheduler_step_index"],
                    "target_sigma": item["target_sigma"],
                    "actual_sigma": item["actual_sigma"],
                }
                for item in checkpoint_steps
            ],
            "seeds": seeds,
            "best_seed": best_seed,
            "total_time_sec": elapsed,
            "rewards": rewards_log,
            "eliminated_at": eliminated_at,
        }


class GeoRewardBoNProgressiveV2(GeoRewardBoNProgressive):
    """
    Two-stage fixed-ratio progressive elimination using 4RC V2 reward.

    Key features:
    - Uses ReconstructionReward (4RC) for geometric consistency scoring
    - All checkpoints use the SAME reward formula (no simplified early version)
    - Supports model offloading: DiT ↔ 4RC alternation to fit in GPU memory
    - Only input frame count differs between checkpoints (12 early, 20 mid/final)
    """

    def __init__(
        self,
        wan_i2v,
        recon_reward,
        frame_indices=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
        offload_models=True,
    ):
        """
        Args:
            wan_i2v: Wan2.2 I2V model wrapper.
            recon_reward: ReconstructionReward instance (V2).
            offload_models: If True, swap DiT/4RC between CPU/GPU at checkpoints.
        """
        super().__init__(
            wan_i2v=wan_i2v,
            frame_indices=frame_indices,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
        )
        self.recon_reward = recon_reward
        self.offload_models = offload_models

    def _generate_prepared(
        self,
        state,
        seeds,
        indices,
        early_indices,
        sigma_checkpoints,
        output_dir,
        save_fn,
    ):
        """Run progressive denoising with V2 reward and model offloading."""
        total_steps = len(state['timesteps'])
        checkpoint_steps = []
        seen_end_steps = set()

        for sigma_target in sorted(sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is None or not 0 < end_step < total_steps:
                continue
            if end_step in seen_end_steps:
                print(
                    f"[BoNProgressiveV2] Skipping duplicate checkpoint "
                    f"sigma={sigma_target:.4f} at completed step {end_step}."
                )
                continue

            actual_step_idx = end_step - 1
            actual_sigma = float(
                state['candidates'][0]['scheduler'].sigmas[actual_step_idx])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
                "scheduler_step_index": actual_step_idx,
            })
            seen_end_steps.add(end_step)

        checkpoint_steps.sort(key=lambda item: item["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
            "scheduler_step_index": total_steps,
        })

        alive = list(range(len(seeds)))
        rewards_log = {f"seed_{s}": {} for s in seeds}
        eliminated_at = {}
        cur_step = 0
        t_start = time.time()
        best_video = None
        best_cand_idx = None
        best_final_score = -float("inf")

        for ckpt_idx, checkpoint in enumerate(checkpoint_steps):
            end_step = checkpoint["end_step"]
            actual_sigma = checkpoint["actual_sigma"]
            is_final = (end_step == total_steps)
            is_early = (ckpt_idx == 0 and not is_final)
            phase_name = (
                "final_sigma0.00" if is_final
                else f"checkpoint{ckpt_idx + 1}_sigma{actual_sigma:.4f}"
            )

            print(f"\n[BoNProgressiveV2] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive candidates)")

            last_preds, pre_step_latents = self.wan.denoise_candidates(
                state, alive, cur_step, end_step)
            cur_step = end_step

            frame_indices_for_phase = early_indices if is_early else indices

            # Phase 1: Decode all candidates (VAE on GPU, DiT can stay)
            # Move decoded tensors to CPU immediately to free GPU for 4RC
            decoded_videos = {}
            for cand_idx in alive:
                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx, last_preds[cand_idx],
                        pre_step_latents[cand_idx])

                video_tensor = self.wan.decode_latent(latent_to_decode)

                if output_dir is not None:
                    seed = seeds[cand_idx]
                    self._save_video(
                        video_tensor, seed, phase_name, output_dir, save_fn)

                decoded_videos[cand_idx] = video_tensor.cpu()

                del latent_to_decode, video_tensor
                torch.cuda.empty_cache()

            # Phase 2: Offload DiT + VAE, load 4RC for scoring
            if self.offload_models:
                self._offload_dit()
                self._offload_vae()
                self._load_4rc()

            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]
                video_tensor = decoded_videos[cand_idx]

                frames_pil = wan_output_to_pil(video_tensor)
                sampled = [frames_pil[i] for i in frame_indices_for_phase
                           if i < len(frames_pil)]

                # V2: same reward formula for all checkpoints
                r = self.recon_reward.compute_reward(sampled)

                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(R_static={r['R_static']:.4f}, "
                      f"R_dynamic={r['R_dynamic']:.4f}, "
                      f"R_motion={r['R_motion']:.4f}, "
                      f"G_anchor={r['G_anchor']:.2f})")

                if is_final:
                    total = float(r.get('total', float('nan')))
                    selection_score = total if np.isfinite(total) else -float("inf")
                    if best_cand_idx is None or selection_score > best_final_score:
                        if best_video is not None:
                            del best_video
                        best_video = video_tensor
                        best_cand_idx = cand_idx
                        best_final_score = selection_score
                    else:
                        del video_tensor
                else:
                    del video_tensor

                torch.cuda.empty_cache()

            del decoded_videos

            # Phase 3: Offload 4RC, reload DiT + VAE for next denoising phase
            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self._load_dit()
                self._load_vae()

            if is_final:
                break

            alive, _ = self._eliminate(
                scored, seeds, eliminated_at, phase_name, is_early)

        if best_cand_idx is None:
            raise RuntimeError("No final candidate was decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = self._build_result_log(
            seeds, rewards_log, eliminated_at, best_seed,
            sigma_checkpoints, elapsed, checkpoint_steps)
        result_log["reward_type"] = "v2_4rc"

        print(f"\n[BoNProgressiveV2] Best: seed_{best_seed} "
              f"(total={rewards_log[f'seed_{best_seed}']['final_sigma0.00']['total']:.4f}) "
              f"in {elapsed:.1f}s")

        return best_video, result_log, best_seed

    def _offload_dit(self):
        """Move DiT model to CPU to free GPU memory for 4RC."""
        if hasattr(self.wan, 'low_noise_model') and self.wan.low_noise_model is not None:
            self.wan.low_noise_model.cpu()
        if hasattr(self.wan, 'high_noise_model') and self.wan.high_noise_model is not None:
            self.wan.high_noise_model.cpu()
        torch.cuda.empty_cache()

    def _load_dit(self):
        """Ensure DiT is ready for denoising.

        When offload_model=True, denoise_candidates() loads the correct model
        (high/low noise) on demand per step. Pre-loading both here would OOM.
        We only need to make sure they're not stuck on a wrong device — but
        since denoise_candidates handles this, this is intentionally a no-op.
        """
        pass

    def _offload_vae(self):
        """Move VAE to CPU to free GPU memory for 4RC."""
        if hasattr(self.wan, 'vae') and self.wan.vae is not None:
            if hasattr(self.wan.vae, 'model'):
                self.wan.vae.model.cpu()
            else:
                self.wan.vae.cpu()
            torch.cuda.empty_cache()

    def _load_vae(self):
        """Move VAE back to GPU for decoding."""
        if hasattr(self.wan, 'vae') and self.wan.vae is not None:
            if hasattr(self.wan.vae, 'model'):
                self.wan.vae.model.cuda()
            else:
                self.wan.vae.cuda()

    def _offload_4rc(self):
        """Move 4RC model to CPU."""
        if self.recon_reward.model is not None:
            self.recon_reward.model.cpu()
            torch.cuda.empty_cache()

    def _load_4rc(self):
        """Move 4RC model to GPU for scoring."""
        if self.recon_reward.model is not None:
            self.recon_reward.model.cuda()


class GeoRewardBoNProgressiveV2Guided(GeoRewardBoNProgressiveV2):
    """
    Progressive elimination BoN with gradient guidance (Phase 3).

    Extends GeoRewardBoNProgressiveV2 by applying geometric gradient guidance
    during denoising steps within the [sigma_min, sigma_max] window. Guidance
    steers latents toward geometrically consistent video generation using
    differentiable 4RC loss.

    Guidance runs at every guidance_frequency-th step within the sigma window.
    Memory management: at guidance steps, DiT is offloaded to CPU to make room
    for 4RC + VAE backward pass, then reloaded for the next DiT step.
    """

    def __init__(
        self,
        wan_i2v,
        recon_reward,
        guidance,
        frame_indices=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
        offload_models=True,
    ):
        """
        Args:
            wan_i2v: Wan2.2 I2V model wrapper.
            recon_reward: ReconstructionReward instance (V2).
            guidance: GeometricGuidance instance (configured with 4RC + VAE + cfg).
            offload_models: If True, swap DiT/4RC between CPU/GPU at checkpoints.
        """
        super().__init__(
            wan_i2v=wan_i2v,
            recon_reward=recon_reward,
            frame_indices=frame_indices,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.guidance = guidance

    def _generate_prepared(
        self,
        state,
        seeds,
        indices,
        early_indices,
        sigma_checkpoints,
        output_dir,
        save_fn,
    ):
        """Run progressive denoising with V2 reward, model offloading, and gradient guidance."""
        total_steps = len(state['timesteps'])
        checkpoint_steps = []
        seen_end_steps = set()

        for sigma_target in sorted(sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is None or not 0 < end_step < total_steps:
                continue
            if end_step in seen_end_steps:
                print(
                    f"[BoNV2Guided] Skipping duplicate checkpoint "
                    f"sigma={sigma_target:.4f} at completed step {end_step}."
                )
                continue

            actual_step_idx = end_step - 1
            actual_sigma = float(
                state['candidates'][0]['scheduler'].sigmas[actual_step_idx])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
                "scheduler_step_index": actual_step_idx,
            })
            seen_end_steps.add(end_step)

        checkpoint_steps.sort(key=lambda item: item["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
            "scheduler_step_index": total_steps,
        })

        alive = list(range(len(seeds)))
        rewards_log = {f"seed_{s}": {} for s in seeds}
        eliminated_at = {}
        cur_step = 0
        t_start = time.time()
        best_video = None
        best_cand_idx = None
        best_final_score = -float("inf")

        for ckpt_idx, checkpoint in enumerate(checkpoint_steps):
            end_step = checkpoint["end_step"]
            actual_sigma = checkpoint["actual_sigma"]
            is_final = (end_step == total_steps)
            is_early = (ckpt_idx == 0 and not is_final)
            phase_name = (
                "final_sigma0.00" if is_final
                else f"checkpoint{ckpt_idx + 1}_sigma{actual_sigma:.4f}"
            )

            print(f"\n[BoNV2Guided] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive candidates)")

            # Use guided denoising — guidance applies within its sigma window
            last_preds, pre_step_latents = self.wan.denoise_candidates_with_guidance(
                state, alive, cur_step, end_step,
                guidance=self.guidance,
                guidance_offload_dit=self._guidance_offload_dit,
                guidance_reload_dit=self._guidance_reload_dit,
            )
            cur_step = end_step

            frame_indices_for_phase = early_indices if is_early else indices

            # Phase 1: Decode all candidates
            decoded_videos = {}
            for cand_idx in alive:
                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx, last_preds[cand_idx],
                        pre_step_latents[cand_idx])

                video_tensor = self.wan.decode_latent(latent_to_decode)

                if output_dir is not None:
                    seed = seeds[cand_idx]
                    self._save_video(
                        video_tensor, seed, phase_name, output_dir, save_fn)

                decoded_videos[cand_idx] = video_tensor.cpu()

                del latent_to_decode, video_tensor
                torch.cuda.empty_cache()

            # Phase 2: Offload DiT + VAE, load 4RC for scoring
            if self.offload_models:
                self._offload_dit()
                self._offload_vae()
                self._load_4rc()

            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]
                video_tensor = decoded_videos[cand_idx]

                frames_pil = wan_output_to_pil(video_tensor)
                sampled = [frames_pil[i] for i in frame_indices_for_phase
                           if i < len(frames_pil)]

                r = self.recon_reward.compute_reward(sampled)

                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(R_static={r['R_static']:.4f}, "
                      f"R_dynamic={r['R_dynamic']:.4f}, "
                      f"R_motion={r['R_motion']:.4f}, "
                      f"G_anchor={r['G_anchor']:.2f})")

                if is_final:
                    total = float(r.get('total', float('nan')))
                    selection_score = total if np.isfinite(total) else -float("inf")
                    if best_cand_idx is None or selection_score > best_final_score:
                        if best_video is not None:
                            del best_video
                        best_video = video_tensor
                        best_cand_idx = cand_idx
                        best_final_score = selection_score
                    else:
                        del video_tensor
                else:
                    del video_tensor

                torch.cuda.empty_cache()

            del decoded_videos

            # Phase 3: Offload 4RC, reload DiT + VAE for next denoising phase
            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self._load_dit()
                self._load_vae()

            if is_final:
                break

            alive, _ = self._eliminate(
                scored, seeds, eliminated_at, phase_name, is_early)

        if best_cand_idx is None:
            raise RuntimeError("No final candidate was decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = self._build_result_log(
            seeds, rewards_log, eliminated_at, best_seed,
            sigma_checkpoints, elapsed, checkpoint_steps)
        result_log["reward_type"] = "v2_4rc"
        result_log["guidance"] = {
            "enabled": True,
            "scale": self.guidance.cfg.guidance_scale,
            "frequency": self.guidance.cfg.guidance_frequency,
            "sigma_min": self.guidance.cfg.sigma_min,
            "sigma_max": self.guidance.cfg.sigma_max,
            "guidance_frames": self.guidance.guidance_frames,
        }

        print(f"\n[BoNV2Guided] Best: seed_{best_seed} "
              f"(total={rewards_log[f'seed_{best_seed}']['final_sigma0.00']['total']:.4f}) "
              f"in {elapsed:.1f}s")

        return best_video, result_log, best_seed

    def _guidance_offload_dit(self):
        """Offload DiT for guidance step (4RC + VAE need GPU)."""
        if hasattr(self.wan, 'low_noise_model') and self.wan.low_noise_model is not None:
            self.wan.low_noise_model.cpu()
        if hasattr(self.wan, 'high_noise_model') and self.wan.high_noise_model is not None:
            self.wan.high_noise_model.cpu()
        torch.cuda.empty_cache()
        # Load 4RC + VAE for gradient computation
        if self.guidance.model_4rc is not None:
            self.guidance.model_4rc.cuda()
        if hasattr(self.wan, 'vae') and self.wan.vae is not None:
            if hasattr(self.wan.vae, 'model'):
                self.wan.vae.model.cuda()
            else:
                self.wan.vae.cuda()

    def _guidance_reload_dit(self):
        """Reload DiT after guidance step, offload 4RC + VAE.

        Explicitly moves DiT models back to GPU to handle the case where
        offload_model=False and init_on_cpu=False — in that scenario
        _prepare_model_for_timestep won't auto-reload from CPU.
        """
        if self.guidance.model_4rc is not None:
            self.guidance.model_4rc.cpu()
        # Also offload VAE (was loaded to GPU for decode_differentiable)
        if hasattr(self.wan, 'vae') and self.wan.vae is not None:
            if hasattr(self.wan.vae, 'model'):
                self.wan.vae.model.cpu()
            else:
                self.wan.vae.cpu()
        torch.cuda.empty_cache()
        # Explicitly reload both DiT models to GPU.
        device = self.wan.device
        if hasattr(self.wan, 'low_noise_model') and self.wan.low_noise_model is not None:
            if next(self.wan.low_noise_model.parameters()).device.type != device.type:
                self.wan.low_noise_model.to(device)
        if hasattr(self.wan, 'high_noise_model') and self.wan.high_noise_model is not None:
            if next(self.wan.high_noise_model.parameters()).device.type != device.type:
                self.wan.high_noise_model.to(device)


class GeoRewardBoNTreeBranching(GeoRewardBoNProgressiveV2):
    """
    Tree Branching + Progressive Elimination BoN.

    The first phase (Step 0 → branch_sigma) runs only K trunk trajectories,
    then branches into N candidates at the branch point. The remainder reuses
    progressive elimination logic from GeoRewardBoNProgressiveV2.

    Compute savings (shift=5.0, branch_sigma=0.90, N=8):
        134 DiT steps vs 190 (progressive) vs 320 (naive). ~29% faster than progressive.
    """

    DEFAULT_NUM_TRUNKS = 2
    DEFAULT_BRANCHES_PER_TRUNK = 4
    DEFAULT_BRANCH_SIGMA = 0.90
    DEFAULT_BRANCH_ETA = 0.10

    def __init__(self, wan_i2v, recon_reward,
                 num_trunks=None, branches_per_trunk=None,
                 branch_sigma=None, branch_eta=None,
                 frame_indices=None, max_frames=20,
                 sigma_checkpoints=None, elimination_ratio=None,
                 min_survivors=None, score_epsilon=None,
                 early_max_frames=None, offload_models=True):
        super().__init__(
            wan_i2v=wan_i2v,
            recon_reward=recon_reward,
            frame_indices=frame_indices,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.num_trunks = num_trunks or self.DEFAULT_NUM_TRUNKS
        self.branches_per_trunk = branches_per_trunk or self.DEFAULT_BRANCHES_PER_TRUNK
        self.branch_sigma = branch_sigma if branch_sigma is not None else self.DEFAULT_BRANCH_SIGMA
        self.branch_eta = branch_eta if branch_eta is not None else self.DEFAULT_BRANCH_ETA

        if not (np.isfinite(self.branch_sigma) and 0.0 < self.branch_sigma < 1.0):
            raise ValueError(
                f"branch_sigma must be finite and in (0, 1), got {self.branch_sigma}")
        if not (0.0 <= self.branch_eta <= 1.0):
            raise ValueError(
                f"branch_eta must be in [0, 1], got {self.branch_eta}")

    def generate(self, prompt, image, N, frame_num=81, seed_base=None,
                 output_dir=None, save_fn=None, **wan_kwargs):
        """
        Tree Branching generation flow.

        Returns: (best_video, result_log, best_seed)
        """
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        expected_N = self.num_trunks * self.branches_per_trunk
        if N != expected_N:
            print(f"[TreeBranching] Warning: N={N} != num_trunks({self.num_trunks}) "
                  f"* branches_per_trunk({self.branches_per_trunk}) = {expected_N}. "
                  f"Using {expected_N}.")
            N = expected_N

        trunk_seeds = [seed_base + i for i in range(self.num_trunks)]
        branch_seeds = [seed_base + 100 + i for i in range(N)]

        max_cp = max(self.sigma_checkpoints)
        if self.branch_sigma <= max_cp:
            raise ValueError(
                f"branch_sigma({self.branch_sigma}) must be > "
                f"max(sigma_checkpoints)({max_cp})")

        state = self.wan.prepare_progressive(
            input_prompt=prompt,
            img=image,
            seeds=trunk_seeds,
            frame_num=frame_num,
            **wan_kwargs,
        )

        try:
            return self._generate_tree(
                state, N, branch_seeds, frame_num, output_dir, save_fn
            )
        finally:
            self.wan.cleanup_progressive(
                state, offload_model=state.get('offload_model', True))

    def _generate_tree(self, state, N, branch_seeds, frame_num,
                       output_dir, save_fn):
        """Tree Branching core logic."""
        t_start = time.time()

        branch_step = self.wan.find_step_for_sigma(state, self.branch_sigma)
        if branch_step is None:
            raise ValueError(
                f"branch_sigma={self.branch_sigma} cannot map to a valid step. "
                f"Check that the sigma schedule contains this value.")

        print(f"[TreeBranching] branch_sigma={self.branch_sigma} -> "
              f"branch_step={branch_step}")
        print(f"[TreeBranching] num_trunks={self.num_trunks}, "
              f"branches_per_trunk={self.branches_per_trunk}, "
              f"eta={self.branch_eta}")

        # === Phase 1: Trunk denoising (Step 0 → branch_step) ===
        trunk_indices = list(range(self.num_trunks))
        print(f"[TreeBranching] Phase 1: Denoising {self.num_trunks} trunks "
              f"for {branch_step} steps...")

        self.wan.denoise_candidates(state, trunk_indices, 0, branch_step)

        # === Phase 2: Branching ===
        print(f"[TreeBranching] Phase 2: Branching {self.num_trunks} trunks "
              f"into {N} candidates (eta={self.branch_eta})...")

        state = self.wan.branch_candidates(
            state, trunk_indices, self.branches_per_trunk,
            self.branch_eta, branch_seeds
        )

        # === Phase 3: Progressive elimination from branch_step ===
        print(f"[TreeBranching] Phase 3: Progressive elimination from "
              f"step {branch_step}...")

        return self._progressive_elimination(
            state, N, branch_seeds, frame_num, output_dir, save_fn,
            start_step=branch_step, t_start=t_start
        )

    def _progressive_elimination(self, state, N, seeds, frame_num,
                                  output_dir, save_fn, start_step, t_start):
        """
        Execute progressive elimination starting from start_step.

        Returns: (best_video, result_log, best_seed)
        """
        total_steps = len(state['timesteps'])

        checkpoint_steps = []
        seen_end_steps = set()
        for sigma_target in sorted(self.sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is None or end_step <= start_step or end_step >= total_steps:
                continue
            if end_step in seen_end_steps:
                continue
            actual_step_idx = end_step - 1
            actual_sigma = float(
                state['candidates'][0]['scheduler'].sigmas[actual_step_idx])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
                "scheduler_step_index": actual_step_idx,
            })
            seen_end_steps.add(end_step)

        checkpoint_steps.sort(key=lambda item: item["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
            "scheduler_step_index": total_steps,
        })

        early_frame_indices = sample_frames(frame_num, self.early_max_frames)
        normal_frame_indices = sample_frames(frame_num, self.max_frames)

        alive = list(range(N))
        eliminated_at = {}
        rewards_log = {f"seed_{s}": {} for s in seeds}
        best_video = None
        best_cand_idx = None
        best_final_score = -float("inf")
        cur_step = start_step

        for cp_idx, cp in enumerate(checkpoint_steps):
            end_step = cp["end_step"]
            is_final = (cp_idx == len(checkpoint_steps) - 1)
            is_early = (cp_idx == 0 and not is_final)
            phase_name = (
                "final_sigma0.00" if is_final
                else f"checkpoint{cp_idx + 1}_sigma{cp['actual_sigma']:.4f}"
            )
            frame_indices = early_frame_indices if is_early else normal_frame_indices

            print(f"\n[TreeBranching] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive)")

            # 1. Denoise to checkpoint
            last_preds, pre_step_latents = self.wan.denoise_candidates(
                state, alive, cur_step, end_step
            )

            # 2. VAE decode
            decoded_videos = {}
            for cand_idx in alive:
                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx,
                        last_preds[cand_idx],
                        pre_step_latents[cand_idx]
                    )
                video_tensor = self.wan.decode_latent(latent_to_decode)
                decoded_videos[cand_idx] = video_tensor.cpu()
                del latent_to_decode, video_tensor
                torch.cuda.empty_cache()

            # 3. Offload DiT + VAE, load 4RC
            if self.offload_models:
                self._offload_dit()
                self._offload_vae()
                self._load_4rc()

            # 4. Score
            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]
                video_tensor = decoded_videos[cand_idx]
                frames_pil = wan_output_to_pil(video_tensor)
                sampled = [frames_pil[i] for i in frame_indices
                           if i < len(frames_pil)]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(R_static={r['R_static']:.4f}, "
                      f"R_dynamic={r['R_dynamic']:.4f}, "
                      f"R_motion={r['R_motion']:.4f}, "
                      f"G_anchor={r['G_anchor']:.2f})")

                if output_dir is not None:
                    self._save_video(
                        decoded_videos[cand_idx], seed, phase_name,
                        output_dir, save_fn)

                if is_final:
                    total = float(r.get('total', float('nan')))
                    selection_score = total if np.isfinite(total) else -float("inf")
                    if selection_score > best_final_score:
                        if best_video is not None:
                            del best_video
                        best_video = video_tensor
                        best_cand_idx = cand_idx
                        best_final_score = selection_score
                    else:
                        del video_tensor
                else:
                    del video_tensor

                torch.cuda.empty_cache()

            del decoded_videos

            # 5. Offload 4RC, reload DiT + VAE
            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self._load_dit()
                self._load_vae()

            # 6. Eliminate
            if not is_final:
                alive, _ = self._eliminate(
                    scored, seeds, eliminated_at, phase_name, is_early
                )

            cur_step = end_step

        if best_cand_idx is None:
            raise RuntimeError("No final candidate was decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = self._build_result_log(
            seeds, rewards_log, eliminated_at, best_seed,
            sigma_checkpoints=self.sigma_checkpoints,
            elapsed=elapsed,
            checkpoint_steps=checkpoint_steps,
        )
        result_log["reward_type"] = "v2_4rc"
        result_log["mode"] = "tree_branching_progressive"
        result_log["tree_branching"] = {
            "num_trunks": self.num_trunks,
            "branches_per_trunk": self.branches_per_trunk,
            "branch_sigma": self.branch_sigma,
            "branch_eta": self.branch_eta,
            "branch_step": start_step,
        }

        print(f"\n[TreeBranching] Best: seed_{best_seed} "
              f"(total={best_final_score:.4f}) in {elapsed:.1f}s")

        return best_video, result_log, best_seed


class GeoRewardBoNTreeBranchingGuided(GeoRewardBoNTreeBranching):
    """
    Tree Branching + Gradient Guidance after first elimination.

    Guidance is applied during denoising via denoise_candidates_with_guidance().
    The guidance sigma window (default sigma_max=0.83) ensures guidance only
    activates after the first elimination checkpoint, when candidates are fewer
    and x0 predictions are more reliable.

    Designed for 4-GPU resident mode: DiT cuda:0, VAE cuda:1/2, 4RC cuda:3.
    No model offload/reload needed during guidance steps.
    """

    def __init__(self, wan_i2v, recon_reward, guidance,
                 num_trunks=None, branches_per_trunk=None,
                 branch_sigma=None, branch_eta=None,
                 frame_indices=None, max_frames=20,
                 sigma_checkpoints=None, elimination_ratio=None,
                 min_survivors=None, score_epsilon=None,
                 early_max_frames=None, offload_models=True,
                 vae_setup_fn=None, vae_restore_fn=None):
        super().__init__(
            wan_i2v=wan_i2v,
            recon_reward=recon_reward,
            num_trunks=num_trunks,
            branches_per_trunk=branches_per_trunk,
            branch_sigma=branch_sigma,
            branch_eta=branch_eta,
            frame_indices=frame_indices,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.guidance = guidance
        self._vae_setup_fn = vae_setup_fn
        self._vae_restore_fn = vae_restore_fn

    def generate(self, prompt, image, N, frame_num=81, seed_base=None,
                 output_dir=None, save_fn=None, **wan_kwargs):
        """
        Tree Branching + Guidance generation flow.

        Overrides parent to insert VAE device split after prepare_progressive()
        (encoder needs to be on cuda:0 for encoding, then decoder splits to
        cuda:1/2 for differentiable guidance decode).
        """
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        expected_N = self.num_trunks * self.branches_per_trunk
        if N != expected_N:
            print(f"[TreeBranchingGuided] Warning: N={N} != "
                  f"num_trunks({self.num_trunks}) * "
                  f"branches_per_trunk({self.branches_per_trunk}) = {expected_N}. "
                  f"Using {expected_N}.")
            N = expected_N

        trunk_seeds = [seed_base + i for i in range(self.num_trunks)]
        branch_seeds = [seed_base + 100 + i for i in range(N)]

        max_cp = max(self.sigma_checkpoints)
        if self.branch_sigma <= max_cp:
            raise ValueError(
                f"branch_sigma({self.branch_sigma}) must be > "
                f"max(sigma_checkpoints)({max_cp})")

        state = self.wan.prepare_progressive(
            input_prompt=prompt,
            img=image,
            seeds=trunk_seeds,
            frame_num=frame_num,
            **wan_kwargs,
        )

        try:
            # Split VAE decoder to guidance devices after encode is done
            if self._vae_setup_fn is not None:
                self._vae_setup_fn()

            return self._generate_tree(
                state, N, branch_seeds, frame_num, output_dir, save_fn
            )
        finally:
            try:
                self.wan.cleanup_progressive(
                    state, offload_model=state.get('offload_model', True))
            finally:
                if self._vae_restore_fn is not None:
                    self._vae_restore_fn()

    def _progressive_elimination(self, state, N, seeds, frame_num,
                                  output_dir, save_fn, start_step, t_start):
        """
        Progressive elimination with gradient guidance.

        Same as parent but uses denoise_candidates_with_guidance().
        Guidance sigma window controls when guidance is active — steps with
        sigma > sigma_max are automatically skipped by should_guide().
        """
        total_steps = len(state['timesteps'])

        checkpoint_steps = []
        seen_end_steps = set()
        for sigma_target in sorted(self.sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is None or end_step <= start_step or end_step >= total_steps:
                continue
            if end_step in seen_end_steps:
                continue
            actual_step_idx = end_step - 1
            actual_sigma = float(
                state['candidates'][0]['scheduler'].sigmas[actual_step_idx])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
                "scheduler_step_index": actual_step_idx,
            })
            seen_end_steps.add(end_step)

        checkpoint_steps.sort(key=lambda item: item["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
            "scheduler_step_index": total_steps,
        })

        early_frame_indices = sample_frames(frame_num, self.early_max_frames)
        normal_frame_indices = sample_frames(frame_num, self.max_frames)

        alive = list(range(N))
        eliminated_at = {}
        rewards_log = {f"seed_{s}": {} for s in seeds}
        best_video = None
        best_cand_idx = None
        best_final_score = -float("inf")
        cur_step = start_step

        for cp_idx, cp in enumerate(checkpoint_steps):
            end_step = cp["end_step"]
            is_final = (cp_idx == len(checkpoint_steps) - 1)
            is_early = (cp_idx == 0 and not is_final)
            phase_name = (
                "final_sigma0.00" if is_final
                else f"checkpoint{cp_idx + 1}_sigma{cp['actual_sigma']:.4f}"
            )
            frame_indices = early_frame_indices if is_early else normal_frame_indices

            print(f"\n[TreeBranchingGuided] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive)")

            # Denoise with guidance (sigma window auto-skips high-sigma phases)
            last_preds, pre_step_latents = self.wan.denoise_candidates_with_guidance(
                state, alive, cur_step, end_step,
                guidance=self.guidance,
                guidance_offload_dit=None,
                guidance_reload_dit=None,
            )

            # VAE decode
            decoded_videos = {}
            vae_dev = getattr(self.guidance, 'vae_device', None)
            for cand_idx in alive:
                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx,
                        last_preds[cand_idx],
                        pre_step_latents[cand_idx]
                    )
                if vae_dev is not None:
                    latent_to_decode = latent_to_decode.to(vae_dev)
                video_tensor = self.wan.decode_latent(latent_to_decode)
                decoded_videos[cand_idx] = video_tensor.cpu()
                del latent_to_decode, video_tensor
                torch.cuda.empty_cache()

            # Offload DiT + VAE, load 4RC (only if single-GPU offload mode)
            if self.offload_models:
                self._offload_dit()
                self._offload_vae()
                self._load_4rc()

            # Score
            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]
                video_tensor = decoded_videos[cand_idx]
                frames_pil = wan_output_to_pil(video_tensor)
                sampled = [frames_pil[i] for i in frame_indices
                           if i < len(frames_pil)]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(R_static={r['R_static']:.4f}, "
                      f"R_dynamic={r['R_dynamic']:.4f}, "
                      f"R_motion={r['R_motion']:.4f}, "
                      f"G_anchor={r['G_anchor']:.2f})")

                if output_dir is not None:
                    self._save_video(
                        decoded_videos[cand_idx], seed, phase_name,
                        output_dir, save_fn)

                if is_final:
                    total = float(r.get('total', float('nan')))
                    selection_score = total if np.isfinite(total) else -float("inf")
                    if selection_score > best_final_score:
                        if best_video is not None:
                            del best_video
                        best_video = video_tensor
                        best_cand_idx = cand_idx
                        best_final_score = selection_score
                    else:
                        del video_tensor
                else:
                    del video_tensor

                torch.cuda.empty_cache()

            del decoded_videos

            # Offload 4RC, reload DiT + VAE (only if single-GPU offload mode)
            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self._load_dit()
                self._load_vae()

            # Eliminate
            if not is_final:
                alive, _ = self._eliminate(
                    scored, seeds, eliminated_at, phase_name, is_early
                )

            cur_step = end_step

        if best_cand_idx is None:
            raise RuntimeError("No final candidate was decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = self._build_result_log(
            seeds, rewards_log, eliminated_at, best_seed,
            sigma_checkpoints=self.sigma_checkpoints,
            elapsed=elapsed,
            checkpoint_steps=checkpoint_steps,
        )
        result_log["reward_type"] = "v2_4rc"
        result_log["mode"] = "tree_branching_progressive_guided"
        result_log["tree_branching"] = {
            "num_trunks": self.num_trunks,
            "branches_per_trunk": self.branches_per_trunk,
            "branch_sigma": self.branch_sigma,
            "branch_eta": self.branch_eta,
            "branch_step": start_step,
        }
        result_log["guidance"] = {
            "enabled": True,
            "scale": self.guidance.cfg.guidance_scale,
            "frequency": self.guidance.cfg.guidance_frequency,
            "sigma_min": self.guidance.cfg.sigma_min,
            "sigma_max": self.guidance.cfg.sigma_max,
            "guidance_frames": self.guidance.guidance_frames,
        }

        print(f"\n[TreeBranchingGuided] Best: seed_{best_seed} "
              f"(total={best_final_score:.4f}) in {elapsed:.1f}s")

        return best_video, result_log, best_seed
