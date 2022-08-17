from __future__ import annotations

import sys
import types
from functools import partial
from typing import (
    Any,
    Callable,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor, nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from typing_extensions import ParamSpec, Self

P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)
# Partial = Callable[P, T_co]
# Partial = partial[T_co]

OptimType_co = TypeVar("OptimType_co", covariant=True, bound=torch.optim.Optimizer)
Stage = Literal["train", "val", "test"]


class StepOutput(TypedDict):
    loss: Tensor
    logits: Tensor
    y: Tensor


class SimpleModel(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        datamodule: VisionDataModule,
        net: nn.Module,
        optimizer_partial: partial[OptimType_co] = partial(torch.optim.Adam),
    ):
        super().__init__()
        self.net = net
        self.optimizer_partial = optimizer_partial

        # loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        self.accuracy: dict[Stage, Accuracy] = {
            "train": Accuracy(),
            "val": Accuracy(),
            "test": Accuracy(),
        }
        self.accuracy = nn.ModuleDict(self.accuracy)  # type: ignore
        self.save_hyperparameters(logger=True, ignore=["net"])

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: Stage
    ) -> StepOutput:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return {"loss": loss, "logits": logits, "y": y}

    def shared_step_end(self, step_output: StepOutput, stage: Stage) -> StepOutput:
        self.accuracy[stage](step_output["logits"], step_output["y"])
        self.log(f"{stage}/accuracy", self.accuracy[stage])
        self.log(f"{stage}/loss", step_output["loss"])
        return step_output

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> StepOutput:
        return self.shared_step(batch, batch_idx, "train")

    def training_step_end(self, step_output: StepOutput) -> StepOutput:
        return self.shared_step_end(step_output, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, "val")

    def validation_step_end(self, step_output: StepOutput) -> StepOutput:
        return self.shared_step_end(step_output, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, "test")

    def test_step_end(self, step_output: StepOutput) -> StepOutput:
        return self.shared_step_end(step_output, "test")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.optimizer_partial(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
