""" GLUETransformer from the PL + HuggingFace example.

https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""
from __future__ import annotations
from dataclasses import dataclass, is_dataclass
import dataclasses
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict, cast
from typing_extensions import Unpack
from torch import Tensor, nn
import datasets
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import get_linear_schedule_with_warmup

from torchmetrics import MatthewsCorrCoef, Metric, Accuracy
from datasets.load import load_metric
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets.features import Features

from src.datamodules.text.glue_datamodule import GlueDataModule

Stage = Literal["train", "val", "test"]


class StepOutput(TypedDict):
    loss: Tensor
    outputs: SequenceClassifierOutput
    labels: Tensor
    preds: Tensor


class GLUETransformer(LightningModule):
    @dataclass
    class HParams:
        model_name_or_path: str | None = None
        learning_rate: float = 2e-5
        adam_epsilon: float = 1e-8
        warmup_steps: int = 0
        weight_decay: float = 0.0
        train_batch_size: int = 32
        eval_batch_size: int = 32

    def __init__(
        self,
        datamodule: GlueDataModule,
        hp: HParams | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.task_name = datamodule.task_name
        self.hp = hp or self.HParams()
        self.num_labels = datamodule.num_labels
        self.hp.model_name_or_path = (
            self.hp.model_name_or_path or datamodule.model_name_or_path
        )
        self.config = AutoConfig.from_pretrained(
            self.hp.model_name_or_path, num_labels=self.num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hp.model_name_or_path, config=self.config
        )
        self.metric = datasets.load.load_metric(
            "glue",
            self.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )
        # NOTE: Add more metrics here if you want.
        self.metrics: dict[str, dict[str, Metric]] = {
            "metrics_train": {
                "matthews_corr": MatthewsCorrCoef(num_classes=self.num_labels),
                "acc": Accuracy(num_classes=self.num_labels),
            },
            "metrics_val": {
                "matthews_corr": MatthewsCorrCoef(num_classes=self.num_labels),
                "acc": Accuracy(num_classes=self.num_labels),
            },
            "metrics_test": {
                "matthews_corr": MatthewsCorrCoef(num_classes=self.num_labels),
                "acc": Accuracy(num_classes=self.num_labels),
            },
        }
        self.metrics = nn.ModuleDict({k: nn.ModuleDict(v) for k, v in self.metrics.items()})  # type: ignore

        self.save_hyperparameters(
            {"hp": dataclasses.asdict(self.hp), "datamodule": self.datamodule.hparams}
        )

    def forward(self, inputs: dict[str, Tensor]) -> SequenceClassifierOutput:
        return self.model(**inputs)

    def training_step(self, batch: dict, batch_idx: int):
        return self.shared_step(batch, batch_idx=batch_idx)

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self.shared_step(batch, batch_idx=batch_idx)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx=batch_idx)

    def shared_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> StepOutput:
        outputs: SequenceClassifierOutput = self(batch)
        loss = outputs.loss
        labels = batch["labels"]
        logits = outputs.logits
        if self.num_labels > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            assert self.num_labels == 1
            preds = logits.squeeze()

        assert isinstance(loss, Tensor)
        return {"loss": loss, "outputs": outputs, "labels": labels, "preds": preds}

    def training_step_end(self, step_output: StepOutput) -> StepOutput:
        return self.shared_step_end(step_output, stage="train")

    def validation_step_end(
        self, step_output: StepOutput, dataloader_idx: int | None = None
    ) -> Optional[StepOutput]:
        return self.shared_step_end(
            step_output, stage="val", dataloader_idx=dataloader_idx
        )

    def test_step_end(
        self, step_output: StepOutput, dataloader_idx: int | None = None
    ) -> Optional[StepOutput]:
        return self.shared_step_end(
            step_output, stage="test", dataloader_idx=dataloader_idx
        )

    def shared_step_end(
        self, step_output: StepOutput, stage: Stage, dataloader_idx: int | None = None
    ) -> StepOutput:
        logits = step_output["outputs"].logits
        labels = step_output["labels"]
        preds = step_output["preds"]
        loss = step_output["loss"]
        # Log the torchmetrics:
        for key, metric in self.metrics[f"metrics_{stage}"].items():
            metric.update(preds=preds, target=labels)
            self.log(
                f"{stage}/{key}",
                metric,
                prog_bar=True,
                on_epoch=stage != "train",
                sync_dist=stage != True,
            )
        # Note: No need to log the loss, PL does it automatically for us.
        # self.log(f"{stage}/loss{suffix}", loss, prog_bar=True)
        return step_output

    def validation_epoch_end(self, outputs: list[StepOutput] | list[list[StepOutput]]):
        if isinstance(outputs[0], dict):
            step_outputs = cast(list[StepOutput], outputs)
            return self.shared_epoch_end(step_outputs, stage="val", dataloader_idx=None)
        else:
            step_outputs = cast(list[list[StepOutput]], outputs)
            for dataloader_idx, dataloader_outputs in enumerate(step_outputs):
                self.shared_epoch_end(
                    dataloader_outputs, stage="val", dataloader_idx=dataloader_idx
                )

    def shared_epoch_end(
        self,
        step_outputs: list[StepOutput],
        stage: Stage,
        dataloader_idx: int | None = None,
    ):
        loss = torch.stack([x["loss"] for x in step_outputs]).mean()
        preds = torch.cat([x["preds"] for x in step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in step_outputs]).detach().cpu().numpy()

        # Log the huggingface metrics:
        metric_value = self.metric.compute(predictions=preds, references=labels)
        suffix = f"_{dataloader_idx}" if dataloader_idx is not None else ""
        assert isinstance(metric_value, dict)
        for key, value in metric_value.items():
            self.log(f"{stage}/model_{key}{suffix}", value, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not any(nd in name for nd in no_decay)
                ],
                "weight_decay": self.hp.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if any(nd in name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hp.learning_rate,
            eps=self.hp.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hp.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
