import typing

import pydantic.v1 as pd

from .base import InvdesBaseModel
from .optimizer import AbstractOptimizerState
from .region import AbstractDesignRegion


class Result(InvdesBaseModel):
    objective_history: typing.Dict[str, typing.Tuple[typing.Any, ...]] = pd.Field(
        ..., title="objective history", desription="history of objective values by objective name"
    )

    region_history: typing.Dict[str, typing.Tuple[AbstractDesignRegion, ...]] = pd.Field(
        ..., title="region history", desription="history of regions through the optimization"
    )

    optimizer_history: typing.Tuple[typing.Tuple[AbstractOptimizerState, ...], ...] = pd.Field(
        ...,
        title="optimizer history",
        desription="history of optimizer states through the optimization",
    )

    figure_of_merit_history: typing.Dict[str, typing.Tuple[float, ...]] = pd.Field(
        ..., title="figure of merit history", description="figure of merit for whole optimization"
    )

    metadata_history: typing.Dict[str, typing.Tuple[typing.Any, ...]] = pd.Field(
        {}, title="metadata history", description="can include iteration, epoch, other useful info"
    )

    def __init__(
        self,
        objective_history,
        region_history,
        optimizer_history,
        figure_of_merit_history,
        metadata_history,
        **kwargs,
    ):
        super().__init__(
            objective_history={key: tuple(val) for key, val in objective_history.items()},
            region_history={key: tuple(val) for key, val in region_history.items()},
            optimizer_history=optimizer_history,
            figure_of_merit_history={
                key: tuple(val) for key, val in figure_of_merit_history.items()
            },
            metadata_history={key: tuple(val) for key, val in metadata_history.items()},
        )

    @property
    def history(self) -> typing.Dict[str, typing.Dict[str, list]]:
        objective_history = {key: list(val) for key, val in self.objective_history.items()}
        region_history = {key: list(val) for key, val in self.region_history.items()}
        optimizer_history = list(self.optimizer_history)
        figure_of_merit_history = {
            key: list(val) for key, val in self.figure_of_merit_history.items()
        }
        metadata_history = {key: list(val) for key, val in self.metadata_history.items()}

        return dict(
            objective_history=objective_history,
            region_history=region_history,
            optimizer_history=optimizer_history,
            figure_of_merit_history=figure_of_merit_history,
            metadata_history=metadata_history,
        )
