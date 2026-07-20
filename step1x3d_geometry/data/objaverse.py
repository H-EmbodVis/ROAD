from dataclasses import dataclass
from typing import Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from step1x3d_geometry import register
from step1x3d_geometry.utils.config import parse_structured

from .base import DataConfig, ShapeImageDataset


@register("objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DataConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ShapeImageDataset(self.cfg, "train")
        if stage in (None, "fit", "validate"):
            self.val_dataset = ShapeImageDataset(self.cfg, "val")
        if stage in (None, "test", "predict"):
            self.test_dataset = ShapeImageDataset(self.cfg, "test")

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.cfg.batch_size, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, 1, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, 1, False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
