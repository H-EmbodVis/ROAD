# Modified from Step1X-3D: portable and minimal Uni3D teacher wrapper.
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

import step1x3d_geometry
from step1x3d_geometry.utils.base import BaseModule
from training.uni3d.models.uni3d import create_uni3d


def load_uni3d_config(path):
    defaults = {
        "pc_model": "eva_giant_patch14_560",
        "pretrained_pc": "",
        "drop_path_rate": 0.0,
        "patch_dropout": 0.0,
        "pc_feat_dim": 1408,
        "embed_dim": 1024,
        "num_group": 512,
        "group_size": 64,
        "pc_encoder_dim": 512,
        "ckpt_path": "",
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        defaults.update(json.load(handle))
    return SimpleNamespace(**defaults)


def point_encoder_state_dict(checkpoint):
    state = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for name, value in state.items():
        if name == "logit_scale":
            continue
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("point_encoder."):
            name = name[len("point_encoder.") :]
        cleaned[name] = value
    return cleaned


@step1x3d_geometry.register("uni3d-align")
class Uni3DAlign(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        config_path: str = "configs/uni3d_g.json"
        projector_dim: int = 2048
        z_size: int = 1024
        n_points: int = 10000

    cfg: Config

    def configure(self):
        self.args = load_uni3d_config(self.cfg.config_path)
        self.model = create_uni3d(self.args)
        checkpoint = torch.load(self.args.ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(
            point_encoder_state_dict(checkpoint), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Uni3D checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, batch):
        points = batch["align_points"][..., :3].contiguous().float()
        colors = torch.ones_like(points)
        output = self.model(points, colors)
        return output.detach(), None
