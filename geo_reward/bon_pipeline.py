"""
Best-of-N sampling pipeline with DA3 GeoReward V1.

Generates N candidate videos with Wan2.2 I2V, scores each with GeoReward,
and selects the geometrically most consistent one.

Includes:
- GeoRewardBoN: original sequential BoN
- GeoRewardBoNProgressive: progressive elimination BoN (σ-based checkpoints)
- GeoRewardBoNOffline: offline scoring of pre-generated videos
"""

import json
import os
import random
import time

import numpy as np
import torch

from .da3_reward import DA3GeoReward
from .utils import wan_output_to_da3_input, sample_frames


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
    Progressive elimination Best-of-N pipeline.

    Instead of generating all N candidates to completion, this pipeline
    inserts σ-based checkpoints during denoising.  At each checkpoint the
    predicted clean latent (pred_x0) is decoded and scored; statistically
    inferior candidates are eliminated so that remaining DiT steps only
    run on survivors.

    Checkpoint behaviour is controlled by ``sigma_checkpoints`` -- a list
    of σ thresholds, e.g. ``[0.65, 0.45]``.  At each checkpoint, candidates
    whose reward falls below ``mean - elimination_std * std`` are dropped.

    Every decoded checkpoint video is saved to disk for visual inspection.
    """

    DEFAULT_SIGMA_CHECKPOINTS = [0.65, 0.45]
    DEFAULT_ELIMINATION_STD = 1.5

    def __init__(
        self,
        wan_i2v,
        da3_reward,
        frame_indices=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_std=None,
    ):
        self.wan = wan_i2v
        self.reward = da3_reward
        self.frame_indices = frame_indices
        self.max_frames = max_frames
        self.sigma_checkpoints = sigma_checkpoints or self.DEFAULT_SIGMA_CHECKPOINTS
        self.elimination_std = (
            elimination_std if elimination_std is not None
            else self.DEFAULT_ELIMINATION_STD
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
        Progressive elimination BoN.

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
        if not np.isfinite(self.elimination_std) or self.elimination_std < 0:
            raise ValueError(
                "elimination_std must be a finite non-negative number, "
                f"got {self.elimination_std}."
            )

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

        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        seeds = [seed_base + i for i in range(N)]

        print(f"[BoNProgressive] Preparing {N} candidates (seeds {seeds[0]}..{seeds[-1]})")
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
        sigma_checkpoints,
        output_dir,
        save_fn,
    ):
        """Run progressive denoising for an already prepared shared state."""
        total_steps = len(state['timesteps'])
        checkpoint_steps = []
        seen_end_steps = set()

        for sigma_target in sorted(sigma_checkpoints, reverse=True):
            end_step = self.wan.find_step_for_sigma(state, sigma_target)
            # A checkpoint at total_steps would duplicate the final decode.
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
            phase_name = (
                "final_sigma0.00" if is_final
                else f"checkpoint{ckpt_idx + 1}_sigma{actual_sigma:.4f}"
            )

            print(f"\n[BoNProgressive] Phase: {phase_name} "
                  f"(steps {cur_step}->{end_step}, {len(alive)} alive candidates)")

            last_preds, pre_step_latents = self.wan.denoise_candidates(
                state, alive, cur_step, end_step)
            cur_step = end_step

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
                sampled = [frames_pil[i] for i in indices if i < len(frames_pil)]
                r = self.reward.compute_reward(sampled)
                rewards_log[f"seed_{seed}"][phase_name] = r
                scored.append((cand_idx, r, None))

                print(f"  seed_{seed}: total={r['total']:.4f} "
                      f"(scene={r['scene']:.4f}, gate={r['motion_gate']:.2f}, "
                      f"shape={r['shape']:.4f}, smooth={r['smoothness']:.4f})")

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
                scored, seeds, eliminated_at, phase_name)

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

    def _eliminate(self, scored, seeds, eliminated_at, phase_name):
        """
        Soft elimination: only drop candidates below mean - k*std.
        Always keep at least 2 candidates alive.
        """
        if len(scored) <= 2:
            return [c for c, _, _ in scored], []

        totals = np.array([
            float(r.get('total', float('nan'))) for _, r, _ in scored
        ], dtype=np.float64)
        finite = np.isfinite(totals)

        # A systemic reward failure must not silently eliminate every
        # candidate.  Keep the phase unchanged so a later checkpoint can
        # still recover and produce diagnostics.
        if not finite.any():
            print(
                f"  WARNING: no finite rewards at {phase_name}; "
                "skipping elimination."
            )
            return [c for c, _, _ in scored], []

        mean_t = totals[finite].mean()
        std_t = totals[finite].std()
        threshold = mean_t - self.elimination_std * std_t

        survivors = []
        eliminated = []
        for (cand_idx, r, _), total in zip(scored, totals):
            if np.isfinite(total) and total >= threshold:
                survivors.append(cand_idx)
            else:
                eliminated.append(cand_idx)
                eliminated_at[f"seed_{seeds[cand_idx]}"] = phase_name

        if len(survivors) < 2:
            ranked = sorted(
                scored,
                key=lambda x: (
                    float(x[1].get('total', float('nan')))
                    if np.isfinite(float(x[1].get('total', float('nan'))))
                    else -float("inf")
                ),
                reverse=True,
            )
            survivors = [c for c, _, _ in ranked[:2]]
            eliminated = [c for c, _, _ in ranked[2:]]
            for cand_idx in survivors:
                eliminated_at.pop(f"seed_{seeds[cand_idx]}", None)
            eliminated_at.update({
                f"seed_{seeds[c]}": phase_name for c in eliminated
            })

        if eliminated:
            elim_seeds = [seeds[c] for c in eliminated]
            print(f"  Eliminated {len(eliminated)} candidates: seeds {elim_seeds} "
                  f"(threshold={threshold:.4f})")

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
            "mode": "progressive_elimination",
            "sigma_checkpoints": sigma_checkpoints,
            "checkpoint_plan": [
                {
                    "completed_steps": item["end_step"],
                    "scheduler_step_index": item["scheduler_step_index"],
                    "target_sigma": item["target_sigma"],
                    "actual_sigma": item["actual_sigma"],
                }
                for item in checkpoint_steps
            ],
            "elimination_std": self.elimination_std,
            "seeds": seeds,
            "best_seed": best_seed,
            "total_time_sec": elapsed,
            "rewards": rewards_log,
            "eliminated_at": eliminated_at,
        }
