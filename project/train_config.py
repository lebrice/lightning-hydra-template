from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import dotenv
import hydra
import hydra_zen
from hydra.core.config_store import ConfigStore
from hydra_configs.pytorch_lightning.trainer import TrainerConf

from hydra_zen.typing import Partial
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from hydra_zen import MISSING

from project.models.base import VisionModel

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@dataclass
class Config:

    datamodule: Any = MISSING
    """ Datamodule to use."""

    original_work_dir: str = ""
    """
    path to original working directory
    hydra hijacks working directory by changing it to the new log directory
    https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
    """

    data_dir: str = "/data"
    """ path to folder with data """

    print_config: bool = True
    """ pretty print config at the start of the run using Rich library. """

    ignore_warnings: bool = False
    """ disable python warnings if they annoy you """

    train: bool = True
    # set False to skip model training

    validate: bool = True
    """ Evaluate on *validation* set, using best model weights achieved during training
    lightning chooses best weights based on the metric specified in checkpoint callback.
    """

    test: bool = False
    """
    Evaluate on test set, using best model weights achieved during training
    lightning chooses best weights based on the metric specified in checkpoint callback
    """

    seed: Optional[int] = None
    """ seed for random number generators in pytorch, numpy and python.random """

    name: str = "default"
    """
    Default name for the experiment, determines logging folder path
    (you can overwrite this name in experiment configs)
    """

    model: Any = MISSING
    """ Which model to train. """

    callbacks: ListConfig = hydra_zen.load_from_yaml("configs/callbacks/default.yaml")
    trainer: TrainerConf = hydra_zen.load_from_yaml("configs/trainer/default.yaml")

    logger: Partial[LightningLoggerBase] = hydra_zen.load_from_yaml(
        "configs/logger/none.yaml"
    )
    """ Set logger here or use command line (e.g. `python train.py logger=tensorboard`) """

    def __post_init__(self) -> None:
        self.original_work_dir = hydra.utils.get_original_cwd()
