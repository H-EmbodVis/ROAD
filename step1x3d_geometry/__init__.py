import importlib
import logging

from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only


_MODULES = {}


def register(name):
    def decorator(cls):
        if name in _MODULES:
            raise ValueError(f"Module {name!r} is already registered")
        _MODULES[name] = cls
        return cls

    return decorator


def find(name):
    if name in _MODULES:
        return _MODULES[name]
    try:
        module_name, class_name = name.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:
        raise ValueError(f"Module {name!r} was not found") from exc


logger = logging.getLogger("pytorch_lightning")
debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warning(*args, **kwargs)


# Import only the components required by the baseline and Uni3D-REPA experiments.
from . import data, models, systems  # noqa: E402,F401
