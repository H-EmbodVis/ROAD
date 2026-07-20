import random
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import torch
from PIL import Image

from step1x3d_geometry.utils.base import BaseModule


ImageType = Union[np.ndarray, torch.Tensor, Image.Image]


class BaseVisualEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        encode_camera: bool = False
        camera_embeds_type: str = "sincos"
        camera_embeds_dim: Optional[int] = None
        n_views: int = 1
        empty_embeds_ratio: float = 0.1
        normalize_embeds: bool = False
        zero_uncond_embeds: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        if self.cfg.encode_camera:
            raise NotImplementedError("Camera-conditioned encoding is not part of this release")

    def encode_image(
        self,
        images: Iterable[Optional[ImageType]],
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch):
        images = batch["image"]
        batch_size = images.shape[0] * images.shape[1] if images.ndim == 5 else images.shape[0]
        if random.random() < self.cfg.empty_embeds_ratio:
            embeds = self.empty_image_embeds.repeat(batch_size, 1, 1)
        else:
            embeds = self.encode_image(images)
        if self.cfg.normalize_embeds:
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds
