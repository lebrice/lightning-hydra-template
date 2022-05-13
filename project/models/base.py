from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Callable

import torch
from hydra_zen import builds
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


logger = get_logger(__name__)


class Model(LightningModule):
    """Base class for all models."""

    @dataclass
    class HParams(Serializable):
        """Hyper-parameters of the model."""

        optimizer: Partial[Optimizer] = builds(torch.optim.AdamW)
        """ Optimizer to use. """

        lr_scheduler: Partial[_LRScheduler] = builds(StepLR)
        """ Learning rate scheduler. """

        @classmethod
        def add_argparse_args(
            cls,
            parser: ArgumentParser,
            dest: str = "hparams",
        ) -> None:
            """Add arguments to the parser."""
            parser.add_arguments(cls, dest=dest)

        @classmethod
        def from_argparse_args(cls, args: Namespace, dest="hparams") -> Self:
            """Create an instance of the model from the arguments."""
            hparams = getattr(args, dest)
            return hparams

    def __init__(
        self, datamodule: LightningDataModule, hp: HParams | None = None
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.hp = hp or self.HParams()

    @classmethod
    def add_argparse_args(
        cls, parser: ArgumentParser, args: Namespace | None = None
    ) -> None:
        """Add arguments to the parser."""
        # parser.add_arguments(cls, dest="hparams")
        cls.HParams.add_argparse_args(parser, dest="hparams", args=args)

    @classmethod
    def from_argparse_args(
        cls: Self | Callable[P, Self],
        parsed_args: Namespace,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Create an instance of the model from the arguments."""
        if not hasattr(parsed_args, "hparams"):
            assert False, vars(parsed_args).keys()
        hparams: Self.HParams = getattr(parsed_args, "hparams")
        return cls(*args, hp=hparams, **kwargs)
