"""
Warmup + TreeBranching+Guidance N=16 (2x8) timing benchmark.

Loads models ONCE, runs warmup N=1, then runs tree_guidance_N16_2x8.
Requires 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_timing_tree16_guidance.py
"""

import os
import sys
import json
import time
import torch
import logging
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Wan2.2"))

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
from geo_reward import ReconstructionReward, ReconRewardConfig
from geo_reward.bon_pipeline import GeoRewardBoNTreeBranchingGuided
from geo_reward.utils import wan_output_to_pil, sample_frames
from geo_reward.guidance import GeometricGuidance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ======================== CONFIG ========================
CKPT_DIR = "/pfs/mayuema/spj/wan/models/Wan2.2-I2V-A14B"
FOURRC_MODEL = "/pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC"
IMAGE_PATH = "/pfs/mayuema/spj/now/inputs/jishi/test_zhenji0014.jpg"
PROMPT = (
    "Use the input image as the exact first frame. A brown drawer box is in "
    "the right foreground with the upper drawer partly open. The left black "
    "robot arm starts from its current pose on the left platform, extends "
    "toward the open drawer front, pushes the drawer inward smoothly until "
    "fully closed and flush, then lifts slightly and returns to its initial "
    "standby area. The right manipulator stays in the background without "
    "interaction. Camera must stay locked and static: no zoom, no pan, no "
    "tilt, no dolly, no camera shift."
)
OUTPUT_BASE = "/pfs/mayuema/spj/now/outputs/jishi/Treebranch_guidance"
SIZE = "480*832"
SAMPLE_SHIFT = 5.0
FRAME_NUM = 81
SAMPLING_STEPS = 40
GUIDE_SCALE = 5.0

GUIDANCE_SCALE = 0.001
GUIDANCE_FREQUENCY = 3
GUIDANCE_SIGMA_MIN = 0.08
GUIDANCE_SIGMA_MAX = 0.83
GUIDANCE_FRAMES = 8

SIGMA_CHECKPOINTS = [0.83, 0.63]
BRANCH_SIGMA = 0.90
BRANCH_ETA = 0.10
MAX_FRAMES = 20
EARLY_MAX_FRAMES = 12
IMAGE_SIZE = 518
# ========================================================


def make_save_fn(wan_cfg):
    def _save_fn(tensor, path):
        save_video(
            tensor=tensor[None],
            save_file=path,
            fps=wan_cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    return _save_fn


def build_cfg():
    return ReconRewardConfig(
        static_weight=0.50, dynamic_weight=0.30, motion_weight=0.20,
        tau_reproj=0.05, tau_accel=0.02, tau_speed=1.5,
        tau_cam=0.02, tau_rot=0.05,
        min_motion=0.005, conf_valid_quantile=0.20,
        max_frames=MAX_FRAMES, image_size=IMAGE_SIZE,
        guidance_scale=GUIDANCE_SCALE,
        guidance_frequency=GUIDANCE_FREQUENCY,
        sigma_min=GUIDANCE_SIGMA_MIN,
        sigma_max=GUIDANCE_SIGMA_MAX,
    )


def vae_setup(wan_i2v, vae_device, vae_device_2):
    if vae_device_2 is not None:
        wan_i2v.vae.split_decoder_to_devices(vae_device, vae_device_2)
    elif hasattr(wan_i2v.vae, 'model'):
        wan_i2v.vae.model.to(vae_device)
    else:
        wan_i2v.vae.to(vae_device)
    wan_i2v.vae.mean = wan_i2v.vae.mean.to(vae_device)
    wan_i2v.vae.std = wan_i2v.vae.std.to(vae_device)
    wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]


def vae_restore(wan_i2v):
    dit_device = torch.device("cuda:0")
    if hasattr(wan_i2v.vae, 'model'):
        wan_i2v.vae.model.to(dit_device)
        wan_i2v.vae.model.decoder.split_layer_idx = None
        wan_i2v.vae.model.decoder.device_2 = None
        wan_i2v.vae.mean = wan_i2v.vae.mean.to(dit_device)
        wan_i2v.vae.std = wan_i2v.vae.std.to(dit_device)
        wan_i2v.vae.scale = [wan_i2v.vae.mean, 1.0 / wan_i2v.vae.std]
    else:
        wan_i2v.vae.to(dit_device)


def run_warmup(wan_i2v, recon_reward, guidance_obj, img, wan_cfg, indices,
               vae_device, vae_device_2):
    """Warmup N=1 sequential guidance to compile all CUDA kernels."""
    warmup_dir = os.path.join(OUTPUT_BASE, "_warmup")
    os.makedirs(warmup_dir, exist_ok=True)

    t0 = time.time()
    state = wan_i2v.prepare_progressive(
        input_prompt=PROMPT, img=img, seeds=[42], frame_num=FRAME_NUM,
        max_area=MAX_AREA_CONFIGS[SIZE], shift=SAMPLE_SHIFT,
        sample_solver="unipc", sampling_steps=SAMPLING_STEPS,
        guide_scale=GUIDE_SCALE, offload_model=False,
    )
    total_steps = len(state['timesteps'])

    vae_setup(wan_i2v, vae_device, vae_device_2)
    wan_i2v.denoise_candidates_with_guidance(
        state, [0], 0, total_steps,
        guidance=guidance_obj,
        guidance_offload_dit=None, guidance_reload_dit=None,
    )

    latent = state['candidates'][0]['latent'].to(vae_device)
    video = wan_i2v.decode_latent(latent)
    vae_restore(wan_i2v)
    wan_i2v.cleanup_progressive(state, offload_model=False)

    frames_pil = wan_output_to_pil(video)
    sampled = [frames_pil[i] for i in indices if i < len(frames_pil)]
    r = recon_reward.compute_reward(sampled)

    logger.info(f"Warmup done in {time.time()-t0:.1f}s, reward={r['total']:.4f}")


def run_tree_guidance(wan_i2v, recon_reward, guidance_obj, img, wan_cfg,
                      vae_device, vae_device_2,
                      num_trunks, branches_per_trunk, exp_name):
    """Run TreeBranching + guidance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_BASE, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    save_fn = make_save_fn(wan_cfg)

    bon = GeoRewardBoNTreeBranchingGuided(
        wan_i2v=wan_i2v,
        recon_reward=recon_reward,
        guidance=guidance_obj,
        num_trunks=num_trunks,
        branches_per_trunk=branches_per_trunk,
        branch_sigma=BRANCH_SIGMA, branch_eta=BRANCH_ETA,
        max_frames=MAX_FRAMES,
        sigma_checkpoints=SIGMA_CHECKPOINTS,
        elimination_ratio=0.5, min_survivors=2, score_epsilon=0.02,
        early_max_frames=EARLY_MAX_FRAMES,
        offload_models=False,
        vae_setup_fn=lambda: vae_setup(wan_i2v, vae_device, vae_device_2),
        vae_restore_fn=lambda: vae_restore(wan_i2v),
    )

    N = num_trunks * branches_per_trunk
    t0 = time.time()
    best_video, result_log, best_seed = bon.generate(
        prompt=PROMPT, image=img, N=N, frame_num=FRAME_NUM,
        seed_base=None, output_dir=exp_dir, save_fn=save_fn,
        max_area=MAX_AREA_CONFIGS[SIZE], shift=SAMPLE_SHIFT,
        sample_solver="unipc", sampling_steps=SAMPLING_STEPS,
        guide_scale=GUIDE_SCALE, offload_model=False,
    )
    total_time = time.time() - t0

    if best_video is not None:
        save_fn(best_video, os.path.join(exp_dir, f"seed_{best_seed}_BEST.mp4"))

    result_log["prompt"] = PROMPT
    result_log["image"] = IMAGE_PATH
    result_log["total_time_sec"] = total_time
    result_log["config"] = {
        "tree_branching": True, "guidance": True,
        "num_trunks": num_trunks, "branches_per_trunk": branches_per_trunk,
        "branch_sigma": BRANCH_SIGMA, "branch_eta": BRANCH_ETA,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_frequency": GUIDANCE_FREQUENCY,
        "guidance_sigma_min": GUIDANCE_SIGMA_MIN,
        "guidance_sigma_max": GUIDANCE_SIGMA_MAX,
        "guidance_frames": GUIDANCE_FRAMES,
        "sigma_checkpoints": SIGMA_CHECKPOINTS,
    }

    with open(os.path.join(exp_dir, "rewards.json"), "w", encoding="utf-8") as f:
        json.dump(result_log, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {exp_dir} ({total_time:.1f}s)")


def main():
    assert torch.cuda.device_count() >= 4, \
        f"Need 4 GPUs, found {torch.cuda.device_count()}"

    vae_device = torch.device("cuda:1")
    vae_device_2 = torch.device("cuda:2")
    fourrc_device = torch.device("cuda:3")

    logger.info("=" * 60)
    logger.info("Loading models (one-time)...")
    logger.info("=" * 60)

    from arc.models.arc.arc import Arc
    fourrc_model = Arc.from_pretrained(FOURRC_MODEL)
    fourrc_model = fourrc_model.to(fourrc_device).eval()

    wan_cfg = WAN_CONFIGS["i2v-A14B"]
    wan_i2v = wan.WanI2V(
        config=wan_cfg, checkpoint_dir=CKPT_DIR,
        device_id=0, rank=0, t5_cpu=True,
    )

    cfg = build_cfg()
    recon_reward = ReconstructionReward(
        model=fourrc_model, device=str(fourrc_device), cfg=cfg)

    guidance_obj = GeometricGuidance(
        model_4rc=fourrc_model, vae=wan_i2v.vae, cfg=cfg,
        guidance_frames=GUIDANCE_FRAMES,
        vae_device=vae_device, fourrc_device=fourrc_device,
        vae_device_2=vae_device_2,
    )

    img = Image.open(IMAGE_PATH).convert("RGB")
    indices = sample_frames(FRAME_NUM, MAX_FRAMES)
    logger.info(f"4-GPU: DiT cuda:0, VAE-front cuda:1, VAE-back cuda:2, 4RC cuda:3")
    logger.info("Models loaded.\n")

    # ========== Warmup ==========
    logger.info("=" * 60)
    logger.info("WARMUP: sequential guidance N=1")
    logger.info("=" * 60)
    run_warmup(wan_i2v, recon_reward, guidance_obj, img, wan_cfg, indices,
               vae_device, vae_device_2)
    logger.info("")

    # ========== TreeBranching + Guidance N=16 (2x8) ==========
    logger.info("=" * 60)
    logger.info("EXPERIMENT: tree_guidance_N16_2x8")
    logger.info("=" * 60)
    run_tree_guidance(
        wan_i2v, recon_reward, guidance_obj, img, wan_cfg,
        vae_device, vae_device_2,
        num_trunks=2, branches_per_trunk=8,
        exp_name="tree_guidance_N16_2x8",
    )

    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
