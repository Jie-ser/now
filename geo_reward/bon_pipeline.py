"""
Best-of-N sampling pipeline with DA3 GeoReward.

Generates N candidate videos with Wan2.2 I2V, scores each with GeoReward,
and selects the geometrically most consistent one.
"""

import random
import sys
import time
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
      4. Compute GeoReward (projection consistency + anchor + confidence).
      5. Return the candidate with the highest total reward.
    """

    def __init__(self, wan_i2v, da3_reward, frame_indices=None, max_frames=20):
        """
        Args:
            wan_i2v: Initialized WanI2V pipeline instance.
            da3_reward: DA3GeoReward instance.
            frame_indices: Specific frame indices to sample, or None to auto-compute.
            max_frames: Number of keyframes to sample when frame_indices is None.
        """
        self.wan = wan_i2v
        self.reward = da3_reward
        self.frame_indices = frame_indices
        self.max_frames = max_frames

    def generate(self, prompt, image, N=8, frame_num=81, seed_base=None,
                 reward_stride=2, **wan_kwargs):
        """
        Generate N candidates and select the best by GeoReward.

        Args:
            prompt: Text prompt (action instruction).
            image: PIL Image (first frame).
            N: Number of candidates to generate.
            frame_num: Number of video frames per candidate.
            seed_base: Base seed (if None, random). Each candidate uses seed_base + i.
            reward_stride: Frame stride for projection consistency computation.
            **wan_kwargs: Additional kwargs passed to wan_i2v.generate().

        Returns:
            Tuple of (all_candidates, all_rewards_list, best_index).
            - all_candidates: list of tensors, each shape (3, T, H, W), range [-1, 1]
            - all_rewards_list: list of reward dicts
            - best_index: index of the selected candidate
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

            # Generate candidate video
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

            # Convert to PIL and sample keyframes
            frames_pil = wan_output_to_da3_input(video)
            sampled_frames = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]

            # Compute reward
            t1 = time.time()
            r = self.reward.compute_reward(sampled_frames, stride=reward_stride)
            reward_time = time.time() - t1
            rewards.append(r)
            timings.append({"gen": gen_time, "reward": reward_time})

            print(f"  Candidate {i+1}/{N} (seed={seed}): "
                  f"total={r['total']:.4f} "
                  f"(proj={r['proj']:.4f}, anchor={r['anchor']:.4f}, conf={r['conf']:.4f}) "
                  f"[gen={gen_time:.1f}s, reward={reward_time:.1f}s]")

        if len(candidates) == 0:
            raise RuntimeError("No valid candidates generated.")

        # Select best
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

    def score_videos(self, video_tensors, frame_num=81, reward_stride=2):
        """
        Score a list of pre-generated video tensors.

        Args:
            video_tensors: List of tensors, each shape (3, T, H, W), range [-1, 1].
            frame_num: Number of frames per video.
            reward_stride: Stride for projection consistency.

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
            r = self.reward.compute_reward(sampled_frames, stride=reward_stride)
            rewards.append(r)
            print(f"  Video {i+1}/{len(video_tensors)}: "
                  f"total={r['total']:.4f} "
                  f"(proj={r['proj']:.4f}, anchor={r['anchor']:.4f}, conf={r['conf']:.4f})")

        return rewards

    def select_best(self, video_tensors, **kwargs):
        """Score all videos and return the best one."""
        rewards = self.score_videos(video_tensors, **kwargs)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
        return video_tensors[best_idx], rewards, best_idx
