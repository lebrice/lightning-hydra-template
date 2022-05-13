"""Utility functions / classes / stuffs."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pl_bolts.datamodules
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

FAST_TEMP_DIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
""" Temporary Directory where we can have really quick reads and writes and limited space. """

# SOURCE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))

BASE_LOG_DIR: Path = Path(os.environ.get("SCRATCH", ".")) / "logs"

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
DEFAULT_DATA_DIR: Path = Path(os.environ.get("SLURM_TMPDIR", "data"))
TORCHVISION_DATA_DIR: Path = Path(os.environ.get("SLURM_TMPDIR", "data"))
_SECRET_TRAIN_VAL_SPLIT_SEED = 42


VISION_DATAMODULES: dict[str, type[VisionDataModule]] = {
    name.replace("DataModule", "").lower(): cls
    for name, cls in vars(pl_bolts.datamodules).items()
    if callable(cls) and issubclass(cls, VisionDataModule)
}


def datamodule_with_name(name: str) -> type[VisionDataModule]:
    if name in VISION_DATAMODULES:
        return VISION_DATAMODULES[name]

    name = name.lower()
    for k, v in VISION_DATAMODULES.items():
        k = k.lower()
        if k == name:
            return v
        if k.replace("datamodule", "") == name:
            return v
    raise NotImplementedError(
        f"Can't find datamodule with name '{name}'.\n "
        f"Available datamodules: {list(VISION_DATAMODULES)}"
    )
