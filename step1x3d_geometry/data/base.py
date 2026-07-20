# Modified from Step1X-3D: portable dataset loader for the two released experiments.
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DataConfig:
    root_dir: str = "data/3d_data"
    batch_size: int = 4
    num_workers: int = 8
    n_samples: int = 32768
    align_points: int = 0
    with_sharp_data: bool = True
    scale: float = 1.0
    image_indices: List[int] = field(default_factory=lambda: list(range(12, 24)))
    validation_image_index: int = 13
    background_color: Tuple[int, int, int] = (255, 255, 255)
    foreground_ratio: float = 0.9
    random_flip: bool = True
    random_color_jitter: bool = True
    random_rotate: bool = True


class ShapeImageDataset(Dataset):
    """Dataset shared by the baseline and Uni3D-REPA experiments.

    Training samples contain surface points and one conditioning image. Validation
    samples only need an image because validation runs the generative sampler.
    """

    def __init__(self, cfg: DataConfig, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.root_dir)
        split_file = self.root / f"{split}.json"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        with split_file.open("r", encoding="utf-8") as handle:
            self.uids = json.load(handle)
        if not isinstance(self.uids, list):
            raise TypeError(f"{split_file} must contain a JSON list")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
        )
        self.rotate = transforms.RandomRotation(
            degrees=10, fill=(*self.cfg.background_color, 0)
        )

    def __len__(self) -> int:
        return len(self.uids)

    def _sample_rows(self, values: np.ndarray, count: int) -> np.ndarray:
        indices = np.random.default_rng().choice(len(values), count, replace=len(values) < count)
        return values[indices]

    def _load_geometry(self, uid: str) -> Dict[str, Any]:
        path = self.root / "surfaces" / f"{uid}.npz"
        with np.load(path) as payload:
            surface_source = payload["surface"]
            surface = self._sample_rows(surface_source, self.cfg.n_samples)
            result = {"surface": surface.astype(np.float32)}
            if self.cfg.with_sharp_data:
                sharp = self._sample_rows(payload["sharp_surface"], self.cfg.n_samples)
                result["sharp_surface"] = sharp.astype(np.float32)
            if self.cfg.align_points > 0:
                align = self._sample_rows(surface_source, self.cfg.align_points)
                result["align_points"] = align.astype(np.float32)

        for key in ("surface", "sharp_surface", "align_points"):
            if key in result:
                result[key][..., :3] *= self.cfg.scale
        return result

    def _image_path(self, uid: str, view_index: int) -> Path:
        return self.root / "3d_images" / uid / f"{view_index:03d}.png"

    def _load_image(self, uid: str) -> Dict[str, Any]:
        view_index = (
            random.choice(self.cfg.image_indices)
            if self.split == "train"
            else self.cfg.validation_image_index
        )
        with Image.open(self._image_path(uid, view_index)) as opened:
            image = opened.convert("RGBA")

        if self.split == "train" and self.cfg.random_color_jitter:
            rgb = self.color_jitter(image.convert("RGB"))
            image = Image.merge("RGBA", (*rgb.split(), image.getchannel("A")))
        if self.split == "train" and self.cfg.random_rotate:
            image = self.rotate(image)

        alpha = image.getchannel("A")
        bbox = alpha.getbbox() or (0, 0, image.width, image.height)
        image = image.crop(bbox)
        alpha = alpha.crop(bbox)

        background = Image.new("RGBA", image.size, (*self.cfg.background_color, 255))
        image = Image.alpha_composite(background, image)
        new_size = tuple(max(1, int(v * self.cfg.foreground_ratio)) for v in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        side = max(image.size)
        canvas = Image.new("RGBA", (side, side), (*self.cfg.background_color, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        canvas = canvas.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")

        return {
            "image": torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0),
            "view_index": view_index,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        uid = self.uids[index]
        try:
            result: Dict[str, Any] = {"uid": uid}
            if self.split == "train":
                result.update(self._load_geometry(uid))
            result.update(self._load_image(uid))

            if self.split == "train" and self.cfg.random_flip and random.random() < 0.5:
                for key in ("surface", "sharp_surface", "align_points"):
                    if key in result:
                        result[key][..., 0] *= -1
                        if result[key].shape[-1] >= 6:
                            result[key][..., 3] *= -1
                result["image"] = torch.flip(result["image"], dims=(1,))
            return result
        except Exception as exc:
            raise RuntimeError(f"Failed to load sample {uid!r}") from exc
