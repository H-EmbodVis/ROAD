from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm


@torch.no_grad()
def flow_sample(
    scheduler,
    diffusion_model: torch.nn.Module,
    shape: Union[List[int], Tuple[int, ...]],
    visual_cond: torch.Tensor,
    steps: int,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cuda",
    disable_progress: bool = False,
):
    if steps <= 0:
        raise ValueError("steps must be positive")
    do_cfg = guidance_scale != 1.0
    batch_size = visual_cond.shape[0] // 2 if do_cfg else visual_cond.shape[0]
    latents = torch.randn(
        (batch_size, *shape),
        generator=generator,
        device=device,
        dtype=visual_cond.dtype,
    )

    scheduler.set_timesteps(steps + 1, device=device)
    timesteps = scheduler.timesteps
    distances = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps

    for index, timestep in enumerate(
        tqdm(timesteps[:-1], disable=disable_progress, desc="Flow sampling", leave=False)
    ):
        model_input = torch.cat([latents, latents]) if do_cfg else latents
        timestep_batch = torch.full(
            (model_input.shape[0],), timestep, dtype=latents.dtype, device=device
        )
        prediction, _, _, _ = diffusion_model(
            model_input, timestep_batch, visual_cond, None, None, cls_input=None
        )
        prediction = prediction.sample
        if do_cfg:
            uncond, cond = prediction.chunk(2)
            prediction = uncond + guidance_scale * (cond - uncond)
        latents = latents - distances[index] * prediction
        yield latents, timestep
