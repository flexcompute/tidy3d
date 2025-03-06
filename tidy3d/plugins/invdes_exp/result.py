import typing
from collections.abc import Iterable

import pydantic.v1 as pd

from .base import InvdesBaseModel
from .optimizer import OptimizerState
from .region import AbstractDesignRegion


class Result(InvdesBaseModel):
    objective_history: typing.Tuple[typing.Any, ...] = pd.Field(
        ..., title="objective history", desription="history of objective values"
    )

    region_history: typing.Dict[str, typing.Tuple[AbstractDesignRegion, ...]] = pd.Field(
        ..., title="region history", desription="history of regions through the optimization"
    )

    optimizer_history: typing.Tuple[typing.Tuple[OptimizerState, ...], ...] = pd.Field(
        ...,
        title="optimizer history",
        desription="history of optimizer states through the optimization",
    )

    metadata_history: typing.Dict[str, typing.Tuple[typing.Any, ...]] = pd.Field(
        ..., title="metadata history", description="can include iteration, epoch, other useful info"
    )

    def __init__(
        self,
        objective_history,
        region_history,
        optimizer_history,
        metadata_history,
        **kwargs,
    ):
        super().__init__(
            objective_history=objective_history,
            region_history={key: tuple(val) for key, val in region_history.items()},
            optimizer_history=optimizer_history,
            metadata_history={key: tuple(val) for key, val in metadata_history.items()},
        )

    @property
    def history(self) -> typing.Dict[str, typing.Dict[str, list]]:
        objective_history = list(self.objective_history)
        region_history = {key: list(val) for key, val in self.region_history.items()}
        optimizer_history = list(self.optimizer_history)
        metadata_history = {key: list(val) for key, val in self.metadata_history.items()}

        return dict(
            objective_history=objective_history,
            region_history=region_history,
            optimizer_history=optimizer_history,
            metadata_history=metadata_history,
        )

    @staticmethod
    def figure_of_merit_history(objective_history):
        return [val if not isinstance(val, Iterable) else val[0] for val in objective_history]
