# Copyright (c) 2021 Massachusetts Institute of Technology
from re import T
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from torchmetrics import Accuracy, MetricCollection
from torchvision import datasets, transforms

from hydra_zen import MISSING, builds, make_custom_builds_fn


###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

#################
# CIFAR10 Dataset
#################
CIFARNormalize = builds(
    transforms.Normalize,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

# transforms.Compose takes a list of transforms
# - Each transform can be configured and appended to the list
TrainTransforms = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
        CIFARNormalize,
    ],
)

TestTransforms = builds(
    transforms.Compose,
    transforms=[builds(transforms.ToTensor), CIFARNormalize],
)

from torch.utils.data import random_split

import random

import numpy as np
import torch
from torch.utils.data import random_split as _random_split
from torch.utils.data.dataset import Dataset


def random_split(
    dataset: Dataset,
    val_split: float = 0.1,
    train: bool = True,
    random_seed: int = 32,
) -> Dataset:
    g = torch.Generator().manual_seed(random_seed)
    nval = int(len(dataset) * val_split)
    ntrain = len(dataset) - nval
    train_data, val_data = _random_split(dataset, [ntrain, nval], g)
    if train:
        return train_data
    return val_data


# Define a function to split the dataset into train and validation sets
SplitDataset = sbuilds(random_split, dataset=MISSING)

# The base configuration for torchvision.dataset.CIFAR10
# - `transform` is left as None and defined later
CIFAR10 = builds(
    datasets.CIFAR10,
    root=MISSING,
    train=True,
    transform=None,
    download=True,
)


# Uses the classmethod `LightningDataModule.from_datasets`
# - Each dataset is a dataclass with training or testing transforms
CIFAR10DataModule = builds(
    LightningDataModule.from_datasets,
    num_workers=4,
    batch_size=256,
    train_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TrainTransforms),
        train=True,
    ),
    val_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TestTransforms),
        train=False,
    ),
    test_dataset=CIFAR10(root="${..root}", transform=TestTransforms, train=False),
    zen_meta=dict(root="${data_dir}"),
)

####################################
# PyTorch Optimizer and LR Scheduler
####################################
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra_zen.typing import Partial, PartialBuilds

SGD = pbuilds(torch.optim.SGD, lr=0.1, momentum=0.9)
Adam = pbuilds(torch.optim.Adam, lr=0.1)
StepLR = pbuilds(torch.optim.lr_scheduler.StepLR, step_size=50, gamma=0.1)
CosineAnnealingLRConfig: PartialBuilds[CosineAnnealingLR] = pbuilds(
    torch.optim.lr_scheduler.CosineAnnealingLR,
)

# Copyright (c) 2021 Massachusetts Institute of Technology
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from typing_extensions import Literal

from torchvision.models.resnet import resnet18, resnet50

# Types for documentation
PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
Predictor = Callable[[Tensor], Tensor]


class BaseImageClassification(LightningModule):
    model: Module

    def __init__(
        self,
        predictor: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        """Initialization of Module

        Parameters
        ----------
        predict: Predictor
            The function to map the output of the model to predictions (e.g., `torch.softmax`)

        criterion: Criterion
            Criterion for calculating the loss. If `criterion` is a string the loss function
            is assumed to be an attribute of the model.

        optim: PartialOptimizer | None (default: None)
            Parameters for the optimizer. Current default optimizer is `SGD`.

        lr_scheduler: PartialLRScheduler | None (default: None)
            Parameters for StepLR. Current default scheduler is `StepLR`.

        metrics: MetricCollection | None (default: None)
            Define PyTorch Lightning `Metric`s.  This module utilizes `MetricCollection`.
        """
        super().__init__()
        self.predictor = predictor
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        if metrics is not None:
            assert isinstance(metrics, MetricCollection)

        self.metrics = metrics

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for Module."""
        return self.model(x)

    def predict(self, batch: Any, *args, **kwargs):
        return self.predictor(self(batch[0]))

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[Any]], None]:
        """Sets up optimizer and learning-rate scheduler."""
        if self.optim is not None:
            optim = self.optim(self.parameters())

            if self.lr_scheduler is not None:
                sched = [self.lr_scheduler(optim)]
                return [optim], sched

            return optim
        return None

    def step(
        self, batch: Tuple[Tensor, Tensor], stage: str = "Train"
    ) -> Dict[str, Tensor]:
        """Executes common step across training, validation and test."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/Loss", loss)
        #
        # Normally metrics would be updated here (See below when needing to support DP)
        #
        return dict(loss=loss, pred=self.predictor(logits), target=y)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Val")

    def test_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Test")

    # To support both DDP and DP we need to use `<stage>_step_end` to update metrics
    # See: https://github.com/PyTorchLightning/pytorch-lightning/pull/4494

    def training_step_end(self, outputs: Union[Tensor, Dict]) -> torch.Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Train")
            return loss
        return outputs

    def validation_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Val")
            return loss
        return outputs

    def test_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Test")
            return loss
        return outputs

    def update_metrics(
        self, outputs: Dict[Literal["pred", "target"], Tensor], stage: str = "Train"
    ):
        if "pred" not in outputs or "target" not in outputs:
            raise TypeError(
                "ouputs dictionary expected contain 'pred' and 'target' keys."
            )

        pred_y: Tensor = outputs["pred"]
        true_y: Tensor = outputs["target"]
        if self.metrics is not None:
            assert isinstance(self.metrics, MetricCollection)
            for key, metric in self.metrics.items():
                val = metric(pred_y, true_y)
                if isinstance(val, Tensor) and val.ndim == 0:
                    self.log(f"{stage}/{key}", val)


class ResNet18Classifier(BaseImageClassification):
    def __init__(
        self,
        *,
        predictor: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        super().__init__(
            predictor,
            criterion,
            optim=optim,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
        )

        self.model = resnet18()


class ResNet50Classifier(BaseImageClassification):
    def __init__(
        self,
        *,
        predictor: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        super().__init__(
            predictor,
            criterion,
            optim=optim,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
        )

        self.model = resnet50()


##########################
# PyTorch Lightning Module
##########################
ImageClassification = builds(
    BaseImageClassification,
    optim=None,
    predictor=builds(torch.nn.Softmax, dim=1),
    criterion=builds(torch.nn.CrossEntropyLoss),
    lr_scheduler=StepLR,
    metrics=builds(
        MetricCollection,
        builds(dict, Accuracy=builds(Accuracy)),
        hydra_convert="all",
    ),
)

ResNet18 = builds(ResNet18Classifier, builds_bases=(ImageClassification,))
ResNet50 = builds(ResNet50Classifier, builds_bases=(ImageClassification,))

###################
# Lightning Trainer
###################
TrainerConf = builds(
    Trainer,
    callbacks=[builds(ModelCheckpoint, mode="min")],  # easily build a list of callbacks
    accelerator="ddp",
    num_nodes=1,
    gpus=builds(torch.cuda.device_count),  # use all GPUs on the system
    max_epochs=100,
    populate_full_signature=True,
)


"""
Register Configs in Hydra's Config Store

This allows user to override configs with "GROUP=NAME" using Hydra's Command Line Interface
or by using hydra-zen's `launch`.

For example using Hydra's CLI:
$ python run_file.py optim=sgd

or the equivalent command using `hydra_run`:
>> launch(config, task_function, overrides=["optim=sgd"])
"""
cs = ConfigStore.instance()

cs.store(group="data", name="cifar10", node=CIFAR10DataModule)
cs.store(group="model", name="resnet18", node=ResNet18)
cs.store(group="model", name="resnet50", node=ResNet50)
cs.store(group="trainer", name="trainer", node=TrainerConf)
cs.store(group="model/optim", name="sgd", node=SGD)
cs.store(group="model/optim", name="adam", node=Adam)
cs.store(group="model/optim", name="none", node=None)
# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

from hydra_zen import MISSING, make_config

# If the repo isn't in the PYTHONPATH let's load it


# Experiment Configs
# - Replaces config.yaml
Config = make_config(
    #
    # Experiment Defaults: See https://hydra.cc/docs/next/advanced/defaults_list
    defaults=[
        "_self_",  # See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order
        {"data": "cifar10"},
        {"model": "resnet18"},
        {"model/optim": "sgd"},
    ],
    #
    # Experiment Modules
    data=MISSING,
    model=MISSING,
    trainer=TrainerConf,
    #
    # Experiment Constants
    data_dir=str(Path().home() / ".data"),
    random_seed=928,
    testing=False,
    ckpt_path=None,
)


cs = ConfigStore.instance()

cs.store(name="config", node=Config)
from pytorch_lightning import seed_everything


# Experiment Task Function
def task_fn(cfg: DictConfig) -> Module:
    # Set seed BEFORE instantiating anything
    seed_everything(cfg.random_seed)

    # Data and Lightning Modules
    data = instantiate(cfg.data)
    pl_module = instantiate(cfg.model)

    # Load a checkpoint if defined
    if cfg.ckpt_path is not None:
        ckpt_data = torch.load(cfg.ckpt_path)
        assert "state_dict" in ckpt_data
        pl_module.load_state_dict(ckpt_data["state_dict"])

    # The PL Trainer
    trainer = instantiate(cfg.trainer)

    # Set training or testing mode
    if cfg.testing:
        trainer.test(pl_module, datamodule=data)
    else:
        trainer.fit(pl_module, datamodule=data)

    return pl_module


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig):
    return task_fn(cfg)


if __name__ == "__main__":
    main()
