from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


class SaverMixin:
    """Small artifact-saving mixin required by the training system."""

    _save_dir: Optional[str] = None

    def create_loggers(self, _cfg_loggers) -> None:
        return None

    def get_loggers(self):
        return []

    def set_save_dir(self, save_dir: str) -> None:
        self._save_dir = save_dir

    def get_save_path(self, filename: str) -> Path:
        if self._save_dir is None:
            raise RuntimeError("Save directory is not configured")
        path = Path(self._save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_mesh(self, filename, vertices, faces) -> None:
        path = self.get_save_path(filename)
        vertices = vertices.detach().cpu().numpy() if hasattr(vertices, "detach") else np.asarray(vertices)
        faces = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
