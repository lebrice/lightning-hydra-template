"""Utility functions / classes / stuffs."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, TypeVar

import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F
from typing_extensions import ParamSpec

from .utils import TORCHVISION_DATA_DIR

StageStr = Literal["train", "sanity_check", "validate", "test", "predict", "tune"]

BatchType = TypeVar("BatchType")


P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)


class CustomVisionDataModule(VisionDataModule):
    """Creates a DataModule from a TorchVision dataset."""

    dataset_type: type[VisionDataset]

    def __init__(
        self,
        data_dir: Path = TORCHVISION_DATA_DIR,
        dataset_type: type[VisionDataset] | Callable[P, VisionDataset] | None = None,
        val_split: int | float = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transform: Callable | None = F.to_tensor,
        label_transforms: Callable | None = None,
        val_data_fraction: float = 0.1,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        super().__init__(
            data_dir=str(data_dir),
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )
        self.dataset_type = dataset_type or type(self).dataset_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_data_fraction = val_data_fraction
        self.transform = transform
        self.target_transform = label_transforms
        self.train_val_split_seed = seed
        self.save_hyperparameters()

        self._train_val_data: VisionDataset | None = None
        self._train_dataset: VisionDataset | Subset | None = None
        self._val_dataset: VisionDataset | Subset | None = None
        self._test_dataset: VisionDataset | Subset | None = None

        # FIXME: Setting these temporarily.
        self.num_classes = 1000
        self.dims = (3, 224, 244)

    def prepare_data(self) -> None:
        """download, split, etc...

        only called on 1 GPU/TPU in distributed
        """
        self._create_dataset(train=True, download=True)
        self._create_dataset(train=True, download=True)

    def _create_dataset(self, train: bool, download: bool = False) -> VisionDataset:
        """Create a dataset from the given split."""
        # NOTE: Here we could prevent calling ImageNet(download=True).
        return self.dataset_type(
            self.data_dir,
            train=train,
            transform=self.transform,
            target_transform=self.target_transform,
            download=download,
        )

    def setup(self, stage: StageStr | str | None = None):
        """Make assignments here (val/train/test split) Called on every process in DDP."""
        # TODO: Customize this.
        if stage in ["train", None]:
            self._train_val_data = self._create_dataset(train=True, download=False)
            total_length = len(self._train_val_data)

            val_length = int(total_length * self.val_data_fraction)
            train_length = total_length - val_length
            self._train_dataset, self._val_dataset = random_split(
                self._train_val_data,
                [train_length, val_length],
                generator=torch.Generator().manual_seed(self.train_val_split_seed),
            )

        if stage in ["test"]:
            self._test_dataset = self._create_dataset(train=False, download=False)

    def train_dataloader(self) -> DataLoader[BatchType]:
        """Create the training dataloader."""
        # self.prepare_data()
        # self.setup(stage="train")
        assert (
            self._train_dataset is not None
        ), "prepare_data and setup should have been called."
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[BatchType]:
        """Create the validation dataloader."""
        # val_split = Dataset(...)
        # return DataLoader(self.val_split)
        assert (
            self._val_dataset is not None
        ), "prepare_data and setup should have been called."
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader[BatchType]:
        """Create the test dataloader."""
        # test_split = Dataset(...)
        # return DataLoader(self.test_split)
        assert (
            self._test_dataset is not None
        ), "prepare_data and setup should have been called."
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

    def teardown(self, stage: StageStr) -> None:
        """clean up after fit or test called on every process in DDP."""
