from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import dotenv
import hydra
import hydra_zen
from hydra.core.config_store import ConfigStore
from hydra_configs.pytorch_lightning.trainer import TrainerConf

# from hydra_configs.torchvision.datasets.cifar import Cifar10Conf
from hydra_zen.typing import Partial, PartialBuilds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
# from hydra_configs.torchvision.models import AlexnetConf

from hydra_zen import MISSING
from pl_bolts.datamodules import MNISTDataModule


@dataclass
class Config:

    datamodule: Any = MISSING

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

    model: Partial[LightningModule] = hydra_zen.load_from_yaml(
        "configs/model/mnist.yaml"
    )

    callbacks: DictConfig = hydra_zen.load_from_yaml("configs/callbacks/default.yaml")
    trainer: TrainerConf = hydra_zen.load_from_yaml("configs/trainer/default.yaml")

    logger: Partial[LightningLoggerBase] = hydra_zen.load_from_yaml(
        "configs/logger/none.yaml"
    )
    """ Set logger here or use command line (e.g. `python train.py logger=tensorboard`) """

    def __post_init__(self) -> None:
        self.original_work_dir = hydra.utils.get_original_cwd()

    # datamodule:
    # model: mnist.yaml
    # callbacks: default.yaml
    # logger: null
    # trainer: default.yaml
    # log_dir: default.yaml


cs = ConfigStore.instance()
# from project.datamodules.mnist_datamodule import MNISTDataModule

cs.store(name="base_config", node=Config)

# from project.datamodules.vision import register_configs
from pl_bolts.datamodules import CIFAR10DataModule

# Register datamodule configs:
cs.store(
    group="datamodule",
    name="cifar10",
    node=hydra_zen.builds(CIFAR10DataModule, config="configs/datamodule/cifar10.yaml"),
)
cs.store(
    group="datamodule",
    name="mnist",
    node=hydra_zen.builds(MNISTDataModule, populate_full_signature=True),
)


from hydra_zen import instantiate


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    config_after = instantiate(config)
    # assert False, config.datamodule
    assert False, (type(config), type(config_after))
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils

    from project.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
