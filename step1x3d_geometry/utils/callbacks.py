from pathlib import Path

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from step1x3d_geometry.utils.config import dump_config


class ConfigSnapshotCallback(Callback):
    def __init__(self, config_path, config, save_dir):
        super().__init__()
        self.config_path = Path(config_path)
        self.config = config
        self.save_dir = Path(save_dir)

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.save_dir / "parsed.yaml", self.config)
        (self.save_dir / "source.yaml").write_text(
            self.config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
