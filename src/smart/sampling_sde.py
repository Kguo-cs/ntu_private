from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps,calculate_shift


def pipeline_with_logprob(device,x):
    num_inference_steps: int = 28
    sigmas = None

    scheduler=FluxPipeline()

    all_latents = [latents]
    all_log_probs = []
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    _num_timesteps = len(timesteps)

    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue
        self._current_timestep = t
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        latents_dtype = latents.dtype
        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            self.scheduler,
            noise_pred.float(),
            t.unsqueeze(0).repeat(latents.shape[0]),
            latents.float(),
            noise_level=noise_level,
        )
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        all_latents.append(latents)
        all_log_probs.append(log_prob)

