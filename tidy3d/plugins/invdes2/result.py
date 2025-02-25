import typing

import pydantic.v1 as pd

from .base import InvdesBaseModel
from .objective import AbstractObjective


class Result(InvdesBaseModel):
    # optimizer_state: = pd.Field(
    #     ...,
    #     title="optimizer state",
    #     description="state that can be used to reinitialize the optimizer"
    # )

    # this will have gradients in it as well via the parameters as long as you save
    # before zeroing it
    # do we need this because these are also in the objectives
    # design_region_history: typing.Dict[str, typing.Tuple[DesignRegion, ...]] = pd.Field(
    #     ...,
    #     title="history of design regions",
    #     description="design region history indicated by region name"
    # )

    # this will have simulations in it as well
    objective_history: typing.Dict[str, typing.Tuple[AbstractObjective, ...]] = pd.Field(
        ..., title="objective history", desription="history of objective values by objective name"
    )

    figure_of_merit_history: typing.Dict[str, typing.Tuple[float, ...]] = pd.Field(
        ..., title="figure of merit history", description="figure of merit for whole optimization"
    )

    metadata_history: typing.Dict[str, typing.Tuple[typing.Any, ...]] = pd.Field(
        {}, title="metadata history", description="can include iteration, epoch, other useful info"
    )

    def __init__(self, objective_history, figure_of_merit_history, metadata_history, **kwargs):
        super().__init__(
            objective_history={
                # list((key, tuple(val)) for key, val in objective_history.items())
                # [(key, tuple(val)) for key, val in objective_history.items()]
                key: tuple(val)
                for key, val in objective_history.items()
            },
            figure_of_merit_history={
                # list((key, tuple(val)) for key, val in figure_of_merit_history.items())
                # [(key, tuple(val)) for key, val in figure_of_merit_history.items()]
                key: tuple(val)
                for key, val in figure_of_merit_history.items()
            },
            # metadata_history=dict(list((key, tuple(val)) for key, val in metadata_history.items())),
            metadata_history={
                # [(key, tuple(val)) for key, val in metadata_history.items()]
                key: tuple(val)
                for key, val in metadata_history.items()
            },
        )

    @property
    def history(self) -> typing.Dict[str, typing.Dict[str, list]]:
        objective_history = {
            # list((key, list(val)) for key, val in self.objective_history.items())
            # [(key, list(val)) for key, val in self.objective_history.items()]
            key: list(val)
            for key, val in self.objective_history.items()
        }
        figure_of_merit_history = {
            # list((key, list(val)) for key, val in self.figure_of_merit_history.items())
            # [(key, list(val)) for key, val in self.figure_of_merit_history.items()]
            key: list(val)
            for key, val in self.figure_of_merit_history.items()
        }
        metadata_history = {
            # list((key, list(val)) for key, val in self.metadata_history.items())
            # [(key, list(val)) for key, val in self.metadata_history.items()]
            key: list(val)
            for key, val in self.metadata_history.items()
        }

        return dict(
            objective_history=objective_history,
            figure_of_merit_history=figure_of_merit_history,
            metadata_history=metadata_history,
        )
