# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    # ------------------------------------------------------------------ #
    #  Progressive Elimination helpers                                    #
    # ------------------------------------------------------------------ #

    def prepare_progressive(
        self,
        input_prompt,
        img,
        seeds,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        offload_model=True,
    ):
        """
        Prepare shared conditioning and per-candidate initial states.

        Returns a dict with everything needed by ``denoise_step_batch``
        and ``decode_candidates``.
        """
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        y = torch.concat([msk, y])

        candidates = []
        for seed in seeds:
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)
            noise = torch.randn(
                16,
                (F - 1) // self.vae_stride[0] + 1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=self.device)

            if sample_solver == 'unipc':
                scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
            elif sample_solver == 'dpm++':
                scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                retrieve_timesteps(
                    scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            candidates.append({
                'latent': noise,
                'scheduler': scheduler,
                'generator': seed_g,
                'seed': seed,
                'step_index': 0,
            })

        timesteps = candidates[0]['scheduler'].timesteps

        return {
            'candidates': candidates,
            'timesteps': timesteps,
            'context': context,
            'context_null': context_null,
            'y': y,
            'max_seq_len': max_seq_len,
            'guide_scale': guide_scale,
            'offload_model': offload_model,
            'frame_num': F,
        }

    def denoise_candidates(self, state, alive_indices, start_step, end_step):
        """
        Run denoising steps [start_step, end_step) for alive candidates.

        Operates in-place on ``state['candidates']``.  Returns the last
        ``noise_pred`` for each alive candidate (needed to compute pred_x0).
        """
        offload_model = state['offload_model']
        guide_scale = state['guide_scale']
        timesteps = state['timesteps']
        boundary = self.boundary * self.num_train_timesteps

        arg_c = {
            'context': [state['context'][0]],
            'seq_len': state['max_seq_len'],
            'y': [state['y']],
        }
        arg_null = {
            'context': state['context_null'],
            'seq_len': state['max_seq_len'],
            'y': [state['y']],
        }

        last_noise_preds = {}
        last_pre_step_latents = {}

        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
        ):
            if offload_model:
                torch.cuda.empty_cache()

            for step_idx in range(start_step, end_step):
                t = timesteps[step_idx]
                is_phase_last_step = (step_idx == end_step - 1)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                for cand_idx in alive_indices:
                    cand = state['candidates'][cand_idx]
                    latent = cand['latent']
                    scheduler = cand['scheduler']

                    latent_model_input = [latent.to(self.device)]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    # pred_x0 is only extracted at the phase checkpoint.  Do
                    # not clone a full video latent at every preceding step.
                    if is_phase_last_step:
                        last_noise_preds[cand_idx] = noise_pred
                        last_pre_step_latents[cand_idx] = latent.clone()

                    temp = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=cand['generator'])[0]
                    cand['latent'] = temp.squeeze(0)

                    del latent_model_input, timestep
                    del noise_pred_cond, noise_pred_uncond

                if offload_model:
                    torch.cuda.empty_cache()

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

        return last_noise_preds, last_pre_step_latents

    def denoise_candidates_with_guidance(
        self, state, alive_indices, start_step, end_step, guidance,
        guidance_offload_dit=None, guidance_reload_dit=None,
    ):
        """
        Run denoising steps [start_step, end_step) with geometric guidance.

        Same as denoise_candidates() but applies guidance.guided_v_pred() at
        eligible steps. Guidance requires 4RC + VAE on GPU, so the caller
        provides offload/reload callbacks for the DiT model.

        Args:
            state: Progressive generation state dict.
            alive_indices: List of candidate indices still alive.
            start_step, end_step: Step range (exclusive end).
            guidance: GeometricGuidance instance.
            guidance_offload_dit: Callable to move DiT to CPU before guidance.
            guidance_reload_dit: Callable to reload DiT to GPU after guidance.

        Returns:
            (last_noise_preds, last_pre_step_latents) same as denoise_candidates.
        """
        offload_model = state['offload_model']
        guide_scale = state['guide_scale']
        timesteps = state['timesteps']
        boundary = self.boundary * self.num_train_timesteps

        arg_c = {
            'context': [state['context'][0]],
            'seq_len': state['max_seq_len'],
            'y': [state['y']],
        }
        arg_null = {
            'context': state['context_null'],
            'seq_len': state['max_seq_len'],
            'y': [state['y']],
        }

        last_noise_preds = {}
        last_pre_step_latents = {}

        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            if offload_model:
                torch.cuda.empty_cache()

            for step_idx in range(start_step, end_step):
                t = timesteps[step_idx]
                is_phase_last_step = (step_idx == end_step - 1)

                # Get current sigma for guidance check.
                # step_index may be None before the first scheduler.step() call;
                # fall back to using step_idx to index sigmas directly.
                scheduler_ref = state['candidates'][alive_indices[0]]['scheduler']
                si = scheduler_ref.step_index
                if si is None:
                    si = step_idx
                sigma_t = float(scheduler_ref.sigmas[si])
                needs_guidance = guidance.should_guide(sigma_t, step_idx)

                # Standard DiT forward (with no_grad as usual)
                with torch.no_grad():
                    model = self._prepare_model_for_timestep(
                        t, boundary, offload_model)
                    sample_guide_scale = (
                        guide_scale[1] if t.item() >= boundary
                        else guide_scale[0]
                    )

                    for cand_idx in alive_indices:
                        cand = state['candidates'][cand_idx]
                        latent = cand['latent']
                        scheduler = cand['scheduler']

                        latent_model_input = [latent.to(self.device)]
                        timestep = torch.stack([t]).to(self.device)

                        noise_pred_cond = model(
                            latent_model_input, t=timestep, **arg_c)[0]
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        noise_pred = noise_pred_uncond + sample_guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                        if is_phase_last_step:
                            last_noise_preds[cand_idx] = noise_pred
                            last_pre_step_latents[cand_idx] = latent.clone()

                        # Store noise_pred temporarily for guidance modification
                        cand['_pending_noise_pred'] = noise_pred

                        del latent_model_input, timestep
                        del noise_pred_cond, noise_pred_uncond

                # Apply guidance if conditions met
                if needs_guidance:
                    # Offload DiT to make room for 4RC + VAE
                    if guidance_offload_dit is not None:
                        guidance_offload_dit()

                    for cand_idx in alive_indices:
                        cand = state['candidates'][cand_idx]
                        latent = cand['latent']
                        noise_pred = cand['_pending_noise_pred']

                        guided_pred = guidance.guided_v_pred(
                            latent, noise_pred, sigma_t, step_idx
                        )
                        cand['_pending_noise_pred'] = guided_pred

                    # Reload DiT for next step
                    if guidance_reload_dit is not None:
                        guidance_reload_dit()

                # Scheduler step (always with no_grad)
                with torch.no_grad():
                    for cand_idx in alive_indices:
                        cand = state['candidates'][cand_idx]
                        latent = cand['latent']
                        scheduler = cand['scheduler']
                        noise_pred = cand.pop('_pending_noise_pred')

                        if is_phase_last_step and needs_guidance:
                            last_noise_preds[cand_idx] = noise_pred

                        temp = scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latent.unsqueeze(0),
                            return_dict=False,
                            generator=cand['generator'])[0]
                        cand['latent'] = temp.squeeze(0)

                if offload_model:
                    torch.cuda.empty_cache()

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

        return last_noise_preds, last_pre_step_latents

    def extract_pred_x0(self, state, cand_idx, noise_pred, pre_step_latent):
        """
        Compute the clean-latent prediction: x0 = x_t - sigma_t * v_pred.

        Uses the latent BEFORE scheduler.step() was called (pre_step_latent)
        and the sigma from that same step (step_index - 1, since step()
        already incremented it).
        """
        scheduler = state['candidates'][cand_idx]['scheduler']
        last_step_idx = scheduler.step_index - 1
        sigma_t = scheduler.sigmas[last_step_idx]
        pred_x0 = pre_step_latent - sigma_t * noise_pred
        return pred_x0

    def get_current_sigma(self, state, cand_idx):
        """Return the current sigma for a given candidate."""
        scheduler = state['candidates'][cand_idx]['scheduler']
        return float(scheduler.sigmas[scheduler.step_index])

    def decode_latent(self, latent):
        """VAE decode a single latent tensor. Returns (3, T, H, W) in [-1, 1]."""
        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
        ):
            videos = self.vae.decode([latent])
        return videos[0]

    def find_step_for_sigma(self, state, target_sigma):
        """
        Find how many denoising steps must be completed to evaluate at the
        first scheduler sigma that is <= ``target_sigma``.

        The returned value is an exclusive ``end_step`` suitable for
        ``range(start_step, end_step)``.  For example, when sigma[i] is the
        first qualifying value, this returns i + 1 so step i is actually run.
        Returns None if no scheduler timestep qualifies.
        """
        sigmas = state['candidates'][0]['scheduler'].sigmas
        timesteps = state['timesteps']
        best_step = None
        for step_idx in range(len(timesteps)):
            s = float(sigmas[step_idx])
            if s <= target_sigma:
                best_step = step_idx + 1
                break
        return best_step

    def branch_candidates(self, state, trunk_indices, branches_per_trunk, eta, branch_seeds):
        """
        From K trunk trajectories, create K * branches_per_trunk branched candidates.

        Args:
            state: dict returned by prepare_progressive()
            trunk_indices: List[int], indices of trunk candidates (e.g. [0, 1])
            branches_per_trunk: int, branches per trunk (e.g. 4)
            eta: float, diversity hyperparameter
            branch_seeds: List[int], one seed per branch (len = K * branches_per_trunk)

        Returns:
            Updated state with candidates list replaced by branched candidates.
        """
        import copy

        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta must be in [0, 1], got {eta}")

        new_candidates = []
        branch_idx = 0

        for trunk_idx in trunk_indices:
            trunk_cand = state['candidates'][trunk_idx]
            z_trunk = trunk_cand['latent'].clone()

            step_idx = trunk_cand['scheduler'].step_index
            sigma_t = float(trunk_cand['scheduler'].sigmas[step_idx])

            for b in range(branches_per_trunk):
                seed = branch_seeds[branch_idx]
                generator = torch.Generator(device=z_trunk.device).manual_seed(seed)
                epsilon = torch.randn(z_trunk.shape, generator=generator,
                                      device=z_trunk.device, dtype=z_trunk.dtype)

                z_branch = math.sqrt(1 - eta**2) * z_trunk + eta * sigma_t * epsilon

                new_scheduler = copy.deepcopy(trunk_cand['scheduler'])

                new_candidates.append({
                    'latent': z_branch,
                    'scheduler': new_scheduler,
                    'generator': generator,
                    'seed': seed,
                    'step_index': step_idx,
                    'parent_trunk': trunk_idx,
                })
                branch_idx += 1

        state['candidates'] = new_candidates
        return state

    def cleanup_progressive(self, state, offload_model=True):
        """Release references held by the progressive state."""
        if state is None:
            return

        for cand in state.get('candidates', []):
            cand.pop('latent', None)
            cand.pop('scheduler', None)
            cand.pop('generator', None)
        state.get('candidates', []).clear()

        # Shared conditioning can also hold CUDA tensors.  Remove the
        # references explicitly so cleanup remains effective in batch jobs
        # even when an exception interrupts a checkpoint phase.
        for key in ('context', 'context_null', 'y', 'timesteps'):
            state.pop(key, None)

        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
