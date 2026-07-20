# Modified from Step1X-3D: reduced to baseline and Uni3D-REPA training paths.
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.systems.utils import flow_sample


def get_sigmas(scheduler, timesteps, ndim=3, dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule = scheduler.timesteps.to(timesteps.device)
    indices = [(schedule == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[indices].flatten()
    while sigma.ndim < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class AlignMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.layers(features)


@step1x3d_geometry.register("rectified-flow-system")
class RectifiedFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        skip_validation: bool = False
        bounds: float = 1.05
        mc_level: float = 0.0
        octree_resolution: int = 384
        guidance_scale: float = 7.5
        num_inference_steps: int = 50

        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0.0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True

        shape_model_type: str = "michelangelo-autoencoder"
        shape_model: dict = field(default_factory=dict)
        visual_condition_type: str = "dinov2-encoder"
        visual_condition: dict = field(default_factory=dict)
        denoiser_model_type: str = "flux-denoiser"
        denoiser_model: dict = field(default_factory=dict)
        noise_scheduler_type: str = "diffusers.schedulers.FlowMatchEulerDiscreteScheduler"
        noise_scheduler: dict = field(default_factory=dict)
        denoise_scheduler_type: str = "diffusers.schedulers.FlowMatchEulerDiscreteScheduler"
        denoise_scheduler: dict = field(default_factory=dict)

        align_model_type: Optional[str] = None
        align_model: Optional[dict] = None
        alignment_start_epoch: int = 3
        alignment_token_count: int = 512
        matcher: str = "gpu"

    cfg: Config

    def configure(self):
        super().configure()
        self.shape_model = step1x3d_geometry.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.requires_grad_(False).eval()

        self.visual_condition = step1x3d_geometry.find(self.cfg.visual_condition_type)(
            self.cfg.visual_condition
        )
        self.visual_condition.requires_grad_(False).eval()
        self.denoiser_model = step1x3d_geometry.find(self.cfg.denoiser_model_type)(
            self.cfg.denoiser_model
        )

        self.noise_scheduler = step1x3d_geometry.find(self.cfg.noise_scheduler_type)(
            **self.cfg.noise_scheduler
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.denoise_scheduler = step1x3d_geometry.find(self.cfg.denoise_scheduler_type)(
            **self.cfg.denoise_scheduler
        )

        self.align_model = None
        self.align_mlp = None
        self.matcher = None
        if self.cfg.align_model_type is not None:
            self.align_model = step1x3d_geometry.find(self.cfg.align_model_type)(self.cfg.align_model)
            self.align_model.requires_grad_(False).eval()
            self.align_mlp = AlignMLP(
                input_dim=self.cfg.denoiser_model.width,
                hidden_dim=self.cfg.align_model.projector_dim,
                output_dim=self.cfg.align_model.z_size,
            )
            self.token_pool = nn.AdaptiveAvgPool1d(self.cfg.alignment_token_count)
            if self.cfg.matcher == "gpu":
                from step1x3d_geometry.utils.hungarianmatcher_gpus import HungarianMatcherWithLossGPU

                self.matcher = HungarianMatcherWithLossGPU()
            elif self.cfg.matcher == "cpu":
                from step1x3d_geometry.utils.hungarianmatcher import HungarianMatcherWithLoss

                self.matcher = HungarianMatcherWithLoss()
            else:
                raise ValueError("matcher must be 'cpu' or 'gpu'")

    def train(self, mode: bool = True):
        super().train(mode)
        self.shape_model.eval()
        self.visual_condition.eval()
        if self.align_model is not None:
            self.align_model.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        sharp_surface = batch.get("sharp_surface")
        if sharp_surface is not None:
            sharp_surface = sharp_surface[..., : 3 + self.cfg.shape_model.point_feats]
        _, latents, _, _ = self.shape_model.encode(
            batch["surface"][..., : 3 + self.cfg.shape_model.point_feats],
            sample_posterior=True,
            sharp_surface=sharp_surface,
        )

        visual_cond = self.visual_condition(batch).to(latents)
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        density = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        indices = (density * self.cfg.noise_scheduler.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(latents.device)
        sigmas = get_sigmas(self.noise_scheduler_copy, timesteps, ndim=3, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        prediction, intermediate, _, _ = self.denoiser_model(
            noisy_latents, timesteps.long(), visual_cond, None, None, cls_input=None
        )
        prediction = prediction.sample
        if self.cfg.precondition_outputs:
            prediction = prediction * (-sigmas) + noisy_latents
            target = latents
        else:
            target = noise - latents

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas
        )
        diffusion_loss = (
            weighting.float() * (prediction.float() - target.float()).square()
        ).reshape(batch_size, -1).mean(dim=1).mean()
        result = {"loss_diffusion": diffusion_loss}

        if self.align_model is not None:
            with torch.no_grad():
                teacher_tokens, _ = self.align_model(batch)
            student_tokens = self.align_mlp(intermediate)

            teacher_global = F.normalize(teacher_tokens.float(), dim=-1).mean(dim=1)
            student_global = F.normalize(student_tokens.float(), dim=-1).mean(dim=1)
            result["loss_proj"] = 1.0 - F.cosine_similarity(
                teacher_global, student_global, dim=-1
            ).mean()

            if self.current_epoch >= self.cfg.alignment_start_epoch:
                pooled_student = self.token_pool(student_tokens.transpose(1, 2)).transpose(1, 2)
                # Uni3D token 0 is the CLS token; local matching uses patch tokens only.
                teacher_local = teacher_tokens[:, 1:].detach()
                result["loss_opt"] = self.matcher(
                    pooled_student.float(), teacher_local.float()
                )
            else:
                result["loss_opt"] = latents.new_zeros(())

        result.update(
            {
                "latents": latents,
                "x_t": noisy_latents,
                "noise": noise,
                "noise_pred": prediction,
                "timesteps": timesteps,
            }
        )
        return result

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total = output["loss_diffusion"] * self.C(self.cfg.loss.lambda_diffusion)
        self.log("train/loss_diffusion", output["loss_diffusion"], prog_bar=True)
        if "loss_proj" in output:
            total = total + output["loss_proj"] * self.C(self.cfg.loss.lambda_proj)
            total = total + output["loss_opt"] * self.C(self.cfg.loss.lambda_opt)
            self.log("train/loss_proj", output["loss_proj"], prog_bar=True)
            self.log("train/loss_opt", output["loss_opt"], prog_bar=True)
        self.log("train/loss", total, prog_bar=True)
        return total

    @torch.no_grad()
    def sample(self, batch, seed: int = 2025):
        cond = self.visual_condition.encode_image(batch["image"]).to(self.device)
        if self.cfg.guidance_scale != 1.0:
            uncond = self.visual_condition.empty_image_embeds.repeat(cond.shape[0], 1, 1).to(cond)
            visual_cond = torch.cat([uncond, cond], dim=0)
        else:
            visual_cond = cond

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = None
        for latents, _ in flow_sample(
            self.denoise_scheduler,
            self.denoiser_model.eval(),
            shape=self.shape_model.latent_shape,
            visual_cond=visual_cond,
            steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
            device=self.device,
        ):
            pass
        return self.shape_model.decode(latents)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.cfg.skip_validation:
            return
        decoded = self.sample(batch)
        meshes = self.shape_model.extract_geometry(
            decoded,
            bounds=self.cfg.bounds,
            mc_level=self.cfg.mc_level,
            octree_resolution=self.cfg.octree_resolution,
            enable_pbar=False,
        )
        for uid, mesh in zip(batch["uid"], meshes):
            if mesh is not None and mesh.verts is not None and mesh.faces is not None:
                safe_uid = str(uid).replace("..", "_").lstrip("/")
                self.save_mesh(
                    f"validation/epoch_{self.current_epoch:04d}/{safe_uid}.glb",
                    mesh.verts,
                    mesh.faces,
                )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
