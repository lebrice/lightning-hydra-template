from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
from pl_bolts.datamodules import MNISTDataModule

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

from project.train_config import Config

cs = ConfigStore.instance()

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


# TODO: Seems a bit tough to regiser the models atm.
# from hydra_zen import MISSING, builds, make_custom_builds_fn

# pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
# # Register model configs:
# from project.models.base import VisionModel

# cs.store(
#     group="model",
#     name="base",
#     node=VisionModel.HParams,
# )

from hydra_zen import instantiate
from omegaconf import OmegaConf


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(raw_config: DictConfig):
    from project.utils import extras
    from project.training_pipeline import train

    # Applies optional utilities
    extras(raw_config)

    # Create the datamodules, etc.
    instantiated_config = instantiate(raw_config, data_dir="BOB")
    config = OmegaConf.to_object(instantiated_config)
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
