"""
Best-of-N sampling pipeline with GeoReward (V1: DA3, V2: 4RC).

Generates N candidate videos with Wan2.2 I2V, scores each with GeoReward,
and selects the geometrically most consistent one.

Includes:
- GeoRewardBoN: original sequential BoN
- GeoRewardBoNProgressive: progressive elimination BoN (σ-based checkpoints)
- GeoRewardBoNProgressiveV2: progressive elimination with 4RC reward + model offloading
- GeoRewardBoNTreeBranching: tree branching + progressive elimination (shared trunk)
- GeoRewardBoNOffline: offline scoring of pre-generated videos

Supports reward_type='v1' (DA3) and reward_type='v2' (4RC) with automatic
model offloading between DiT denoising and reward scoring phases.
"""

import json
import os
import random
import time

import numpy as np
import torch

from .da3_reward import DA3GeoReward
from .utils import wan_output_to_pil, wan_output_to_da3_input, sample_frames


class GeoRewardBoN:
    """
    Best-of-N generation pipeline using DA3 geometry reward for selection.

    Workflow:
      1. Generate N candidate videos from the same prompt/image with different seeds.
      2. Extract keyframes from each candidate.
      3. Run DA3 inference to get depth/pose/confidence.
      4. Compute GeoReward V1 (scene + motion).
      5. Return the candidate with the highest total reward.
    """

    def __init__(self, wan_i2v, da3_reward, frame_indices=None, max_frames=20):
        self.wan = wan_i2v
        self.reward = da3_reward
        self.frame_indices = frame_indices
        self.max_frames = max_frames

    def generate(self, prompt, image, N=8, frame_num=81, seed_base=None, **wan_kwargs):
        """
        Generate N candidates and select the best by GeoReward.

        Returns:
            Tuple of (all_candidates, all_rewards_list, best_index).
        """
        if self.frame_indices is not None:
            indices = self.frame_indices
        else:
            indices = sample_frames(frame_num, self.max_frames)

        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        candidates = []
        rewards = []
        timings = []

        print(f"[GeoRewardBoN] Generating {N} candidates...")

        for i in range(N):
            seed = seed_base + i

            t0 = time.time()
            video = self.wan.generate(
                input_prompt=prompt,
                img=image,
                frame_num=frame_num,
                seed=seed,
                **wan_kwargs
            )
            gen_time = time.time() - t0

            if video is None:
                print(f"  Candidate {i+1}/{N}: generation returned None (non-rank-0?), skipping.")
                continue

            candidates.append(video)

            frames_pil = wan_output_to_da3_input(video)
            sampled_frames = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]

            t1 = time.time()
            r = self.reward.compute_reward(sampled_frames)
            reward_time = time.time() - t1
            rewards.append(r)
            timings.append({"gen": gen_time, "reward": reward_time})

            print(f"  Candidate {i+1}/{N} (seed={seed}): "
                  f"total={r['total']:.4f} "
                  f"(scene={r['scene']:.4f}, motion={r['motion']:.4f}, "
                  f"gate={r['motion_gate']:.2f}, shape={r['shape']:.4f}, "
                  f"smooth={r['smoothness']:.4f}) "
                  f"[gen={gen_time:.1f}s, reward={reward_time:.1f}s]")

        if len(candidates) == 0:
            raise RuntimeError("No valid candidates generated.")

        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
        print(f"\n[GeoRewardBoN] Selected candidate {best_idx+1}/{len(candidates)} "
              f"with reward {rewards[best_idx]['total']:.4f}")

        return candidates, rewards, best_idx


class GeoRewardBoNOffline:
    """
    Offline (post-hoc) scoring variant: score pre-generated videos without
    re-generating them. Useful for ablation studies.
    """

    def __init__(self, da3_reward, frame_indices=None, max_frames=20):
        self.reward = da3_reward
        self.frame_indices = frame_indices
        self.max_frames = max_frames

    def score_videos(self, video_tensors, frame_num=81):
        """
        Score a list of pre-generated video tensors.

        Returns:
            List of reward dicts, one per video.
        """
        if self.frame_indices is not None:
            indices = self.frame_indices
        else:
            indices = sample_frames(frame_num, self.max_frames)

        rewards = []
        for i, video in enumerate(video_tensors):
            frames_pil = wan_output_to_da3_input(video)
            sampled_frames = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]
            r = self.reward.compute_reward(sampled_frames)
            rewards.append(r)
            print(f"  Video {i+1}/{len(video_tensors)}: "
                  f"total={r['total']:.4f} "
                  f"(scene={r['scene']:.4f}, motion={r['motion']:.4f}, "
                  f"gate={r['motion_gate']:.2f}, shape={r['shape']:.4f}, "
                  f"smooth={r['smoothness']:.4f})")

        return rewards

    def select_best(self, video_tensors, **kwargs):
        """Score all videos and return the best one."""
        rewards = self.score_videos(video_tensors, **kwargs)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
        return video_tensors[best_idx], rewards, best_idx


class GeoRewardBoNProgressive:
    """
    Two-stage fixed-ratio progressive elimination Best-of-N pipeline.

    Flow:
      N=8 candidates start → Step 15 early checkpoint → eliminate bottom 50% →
      4 survivors → Step 25 mid checkpoint → eliminate bottom 50% →
      2 survivors → Step 40 final → pick best.

    Compute savings: 8×15 + 4×10 + 2×15 = 190 DiT steps vs 320 (original).

    Checkpoints are defined by σ thresholds (default [0.83, 0.63]) so the
    mechanism adapts if the total step count changes.

    Early checkpoint uses a simplified reward (motion_gate + R_scene only)
    since R_shape/R_smoothness are unstable on noisy pred_x0.
    Mid and final checkpoints use the full reward.
    """

    DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]
    DEFAULT_ELIMINATION_RATIO = 0.5
    DEFAULT_MIN_SURVIVORS = 2
    DEFAULT_SCORE_EPSILON = 0.02
    DEFAULT_EARLY_MAX_FRAMES = 12

    def __init__(
        self,
        wan_i2v,
        da3_reward,
        frame_indices=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
    ):
        self.wan = wan_i2v
        self.reward = da3_reward
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
        """Run progressive denoising with fixed-ratio elimination."""
        total_steps = len(state['timesteps'])
        checkpoint_steps = []
        seen_end_steps = set()

        for sigma_target in sorted(sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            if end_step is None or not 0 < end_step < total_steps:
                continue
            if end_step in seen_end_steps:
                print(
                    f"[BoNProgressive] Skipping duplicate checkpoint "
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

            print(f"\n[BoNProgressive] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive candidates)")

            last_preds, pre_step_latents = self.wan.denoise_candidates(
                state, alive, cur_step, end_step)
            cur_step = end_step

            frame_indices_for_phase = early_indices if is_early else indices

            scored = []
            for cand_idx in alive:
                seed = seeds[cand_idx]

                if is_final:
                    latent_to_decode = state['candidates'][cand_idx]['latent']
                else:
                    latent_to_decode = self.wan.extract_pred_x0(
                        state, cand_idx, last_preds[cand_idx],
                        pre_step_latents[cand_idx])

                video_tensor = self.wan.decode_latent(latent_to_decode)

                if output_dir is not None:
                    self._save_video(
                        video_tensor, seed, phase_name, output_dir, save_fn)

                frames_pil = wan_output_to_da3_input(video_tensor)
                sampled = [frames_pil[i] for i in frame_indices_for_phase
                           if i < len(frames_pil)]

                if is_early:
                    r = self.reward.compute_reward_early(sampled)
                else:
                    r = self.reward.compute_reward(sampled)

                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(scene={r['scene']:.4f}, gate={r['motion_gate']:.2f}"
                      f"{', shape=' + format(r['shape'], '.4f') + ', smooth=' + format(r['smoothness'], '.4f') if not is_early else ''})")

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

                del latent_to_decode
                torch.cuda.empty_cache()

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

        print(f"\n[BoNProgressive] Best: seed_{best_seed} "
              f"(total={rewards_log[f'seed_{best_seed}']['final_sigma0.00']['total']:.4f}) "
              f"in {elapsed:.1f}s")

        return best_video, result_log, best_seed

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

    Key differences from GeoRewardBoNProgressive:
    - Uses ReconstructionReward (4RC) instead of DA3GeoReward
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
            da3_reward=None,
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

    @property
    def reward(self):
        """V2 does not use DA3 reward. Fail loud if anyone calls through."""
        raise AttributeError(
            "GeoRewardBoNProgressiveV2 uses self.recon_reward (4RC), "
            "not self.reward (DA3). Use self.recon_reward.compute_reward() instead."
        )

    @reward.setter
    def reward(self, value):
        # Silently absorb parent __init__ setting self.reward = da3_reward.
        # The value is discarded; V2 uses self.recon_reward exclusively.
        # If parent code evolves to read self.reward outside _generate_prepared,
        # the property getter above will raise immediately.
        pass

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
        """Move DiT model back to GPU for denoising."""
        if hasattr(self.wan, 'low_noise_model') and self.wan.low_noise_model is not None:
            self.wan.low_noise_model.cuda()
        if hasattr(self.wan, 'high_noise_model') and self.wan.high_noise_model is not None:
            self.wan.high_noise_model.cuda()

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
