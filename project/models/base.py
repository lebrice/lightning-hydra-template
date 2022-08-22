from dataclasses import dataclass
from functools import partial
from logging import getLogger as get_logger
from typing import Any, Mapping, Optional, cast

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Trainer
from simple_parsing import field
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models

from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable

import torch
from hydra_zen import MISSING, builds
from hydra_zen.typing import Partial
from pytorch_lightning import LightningDataModule, LightningModule
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization.serializable import Serializable
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import Self

# from simple_parsing.helpers.partial import Partial, config_dataclass_for

# from project.config.optimizer import optimizer_choice
# from project.config.schedulers import lr_sheduler_choice
# from project.utils.utils import P
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

import inspect
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable, NewType, TypeVar

from simple_parsing import Serializable, choice
from torch import nn
from torchvision import models
from typing_extensions import ParamSpec

logger = get_logger(__name__)

P = ParamSpec("P")
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
ModuleType = TypeVar("ModuleType", bound=nn.Module)


BACKBONES: dict[str, type[nn.Module]] = {
    name: cls_or_fn
    for name, cls_or_fn in vars(models).items()
    if (callable(cls_or_fn) and "pretrained" in inspect.signature(cls_or_fn).parameters)
}


@dataclass
class BackboneConfig(Serializable):
    """Arguments for the backbone network."""

    type: Callable[..., nn.Module] = choice(
        BACKBONES,
        default=models.resnet18,
    )
    """ Which type of backbone network to use. """

    pretrained: bool = False

    def make_backbone(self, image_dims: tuple[C, H, W]) -> tuple[int, nn.Module]:
        return get_backbone_network(
            network_type=self.type, image_dims=image_dims, pretrained=self.pretrained
        )


def get_backbone_network(
    network_type: Callable[..., ModuleType],
    *,
    image_dims: tuple[C, H, W],
    pretrained: bool = False,
) -> tuple[int, ModuleType]:
    """Construct a backbone network using the given image dimensions and network type.

    Replaces the last fully-connected layer with a `nn.Identity`.

    TODO: Add support for more types of models.
    """

    backbone_signature = inspect.signature(network_type)
    if (
        "image_size" in backbone_signature.parameters
        or backbone_signature.return_annotation is models.VisionTransformer
    ):
        backbone = network_type(image_size=image_dims[-1], pretrained=pretrained)
    else:
        backbone = network_type(pretrained=pretrained)

    # Replace the output layer with a no-op, we'll create our own instead.
    if hasattr(backbone, "fc"):
        in_features: int = backbone.fc.in_features  # type: ignore
        backbone.fc = nn.Identity()
    elif isinstance(backbone, models.VisionTransformer):
        head_layers = backbone.heads
        fc = head_layers.get_submodule("head")
        fc_index = list(head_layers).index(fc)
        assert isinstance(fc, nn.Linear)
        in_features = fc.in_features
        head_layers[fc_index] = nn.Identity()
    else:
        raise NotImplementedError(
            f"TODO: Don't yet know how to remove last fc layer(s) of networks of type "
            f"{type(backbone)}!\n"
        )

    return in_features, backbone


import torchvision

from hydra_zen import MISSING, builds, make_custom_builds_fn


###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
from typing import Tuple, Iterable
from torch import Tensor, nn

# Types for documentation
PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[nn.Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
Predictor = Callable[[Tensor], Tensor]
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

class VisionModel(LightningModule):
    """Base class for all models."""

    @dataclass
    class HParams(Serializable):
        """Hyper-parameters of the model."""

        backbone: Any = MISSING
        """ Backbone model to use as a feature extractor. """

        optimizer: Partial[Optimizer] = builds(torch.optim.AdamW)
        """ Optimizer to use. """

        lr_scheduler: Partial[_LRScheduler] = builds(StepLR)
        """ Learning rate scheduler. """

    def __init__(self, datamodule: VisionDataModule, hp: HParams) -> None:
        """Create an instance of the model.

        Parameters
        ----------
        datamodule : VisionDataModule
            The datamodule that this model will use to learn and be evaluated on.
        hp : HParams | None, optional
            Hyper-parameters, by default None, in which case the default hyper-parameters are used.
            NOTE: This is called `hp` in order to avoid a conflict with the `self.hparams`
            property, which contains the values of all the constructor arguments.
        """
        super().__init__()
        self.datamodule = datamodule
        self.hp = hp or self.HParams()

        image_dims: tuple[int, int, int] = self.datamodule.dims
        assert hasattr(self.datamodule, "num_classes")
        num_classes: int = self.datamodule.num_classes  # type: ignore

        hidden_dims, backbone = self.hp.backbone.make_backbone(image_dims=image_dims)
        self.backbone = backbone
        self.output = nn.Linear(hidden_dims, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="none")

        metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "top5": Accuracy(num_classes=num_classes, top_k=5),
            }
        )
        self.metrics: Mapping[str, Metric] = cast(Mapping[str, Metric], metrics)
        self.save_hyperparameters(
            {
                "datamodule": self.datamodule.hparams,
                "hp": self.hp.to_dict(),
            }
        )

        self.trainer: Trainer
        logger.info("Model Hyper-Parameters:\n", self.hparams)
        self.example_input_array = torch.rand([self.datamodule.batch_size, *image_dims])
        # Whether the models have been wrapped with fairscale.
        self._model_are_wrapped = False

    def configure_sharded_model(self) -> None:
        # NOTE: From https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training
        # NOTE: This gets called during train / val / test, so we need to check that we don't wrap
        # the model twice.
        if not self._model_are_wrapped:
            # NOTE: Could probably use any of the cool things from fairscale here, like
            # mixture-of-experts sharding, etc!
            if ConfigAutoWrap.in_autowrap_context:
                logger.info(
                    "Wrapping models for model-parallel training using fairscale"
                )
            self.backbone = auto_wrap(self.backbone)
            self.output = auto_wrap(self.output)
            self._model_are_wrapped = True

    def configure_optimizers(self) -> Optimizer:
        optimizer = self.hp.optimizer(self.parameters())
        if not self.hp.lr_scheduler:
            return optimizer

        lr_scheduler = self.hp.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.backbone(x)
        logits = self.output(h_x)
        return logits

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="val")

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: str
    ) -> dict:
        x, y = batch
        y = y.to(self.device)
        logits = self.forward(x)
        loss: Tensor = self.loss(
            logits, y
        )  # partial loss (worker_batch_size, n_classes)
        return {
            "loss": loss,
            "logits": logits,
            "y": y,
        }

    def training_step_end(self, step_output: Tensor) -> Tensor:
        loss = self.shared_step_end(step_output, phase="train")
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step_end(self, step_output: Tensor) -> Tensor:
        loss = self.shared_step_end(step_output, phase="val")
        self.log("val/loss", loss, on_epoch=True, logger=True)
        return loss

    def shared_step_end(self, step_output: Tensor, phase: str) -> Tensor:
        assert isinstance(step_output, dict)
        loss = step_output["loss"]  # un-reduced loss (batch_size, n_classes)
        y = step_output["y"]
        logits = step_output["logits"]
        # Log the metrics in `shared_step_end` when they are fused from all workers.
        for name, metric in self.metrics.items():
            result = metric(logits, y)
            self.log(f"{phase}/{name}", metric, logger=True, prog_bar=True)
        return loss.mean()

    # NOTE: Adding these properties in case we are using the auto_find_lr or auto_find_batch_size
    # features of the Trainer, since it modifies these attributes.

    @property
    def batch_size(self) -> int:
        return self.datamodule.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        print(f"Changing batch size from {self.datamodule.batch_size} to {value}")
        self.datamodule.batch_size = value

    @property
    def lr(self) -> float:
        return self.hp.optimizer.lr

    @lr.setter
    def lr(self, lr: float) -> None:
        print(f"Changing lr from {self.hp.optimizer.lr} to {lr}")
        self.hp.optimizer.lr = lr
