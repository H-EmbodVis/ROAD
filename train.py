import argparse
import os
from pathlib import Path

def main(args, overrides):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Import CUDA-aware libraries only after GPU visibility is finalized. This
    # is required when selecting one GPU on a multi-GPU host.
    import pytorch_lightning as pl
    import torch
    import torch.distributed as dist
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    import step1x3d_geometry
    from step1x3d_geometry.utils.callbacks import ConfigSnapshotCallback
    from step1x3d_geometry.utils.config import load_config

    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config, cli_args=overrides, n_gpus=gpu_count)
    pl.seed_everything(cfg.seed, workers=True)
    data_module = step1x3d_geometry.find(cfg.data_type)(cfg.data)
    system = step1x3d_geometry.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(str(Path(cfg.trial_dir) / "artifacts"))

    callbacks = [TQDMProgressBar(refresh_rate=1)]
    loggers = []
    if args.train:
        callbacks.extend(
            [
                ModelCheckpoint(dirpath=Path(cfg.trial_dir) / "checkpoints", **cfg.checkpoint),
                LearningRateMonitor(logging_interval="step"),
                ConfigSnapshotCallback(args.config, cfg, Path(cfg.trial_dir) / "configs"),
            ]
        )
        loggers = [
            TensorBoardLogger(cfg.trial_dir, name="tensorboard"),
            CSVLogger(cfg.trial_dir, name="csv"),
        ]

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu",
        devices=-1,
        **cfg.trainer,
    )
    try:
        if args.train:
            trainer.fit(system, datamodule=data_module, ckpt_path=cfg.resume)
        elif args.validate:
            trainer.validate(system, datamodule=data_module, ckpt_path=cfg.resume)
        elif args.test:
            trainer.test(system, datamodule=data_module, ckpt_path=cfg.resume)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Step1X-3D baseline or Uni3D-REPA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="0", help="Visible GPU indices, e.g. 0,1,2,3")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--test", action="store_true")
    parsed, extras = parser.parse_known_args()
    main(parsed, extras)
