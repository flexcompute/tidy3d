# convenient container for the output of the inverse design (specifically the history)

import typing

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.types import ArrayLike

from .base import InvdesBaseModel
from .design import InverseDesignType

# TODO: implement more convenience methods for exporting to figures?


class InverseDesignResult(InvdesBaseModel):
    """Container for the result of an ``InverseDesign.run()`` call."""

    design: InverseDesignType = pd.Field(
        ...,
        title="Inverse Design Specification",
        description="Specification describing the inverse design problem we wish to optimize.",
    )

    params: typing.Tuple[ArrayLike, ...] = pd.Field(
        (),
        title="Parameter History",
        description="History of parameter arrays throughout the optimization.",
    )

    objective_fn_val: typing.Tuple[float, ...] = pd.Field(
        (),
        title="Objective Function History",
        description="History of objective function values throughout the optimization.",
    )

    grad: typing.Tuple[ArrayLike, ...] = pd.Field(
        (),
        title="Gradient History",
        description="History of objective function gradient arrays throughout the optimization.",
    )

    penalty: typing.Tuple[float, ...] = pd.Field(
        (),
        title="Penalty History",
        description="History of weighted sum of penalties throughout the optimization.",
    )

    post_process_val: typing.Tuple[float, ...] = pd.Field(
        (),
        title="Post-Process Function History",
        description="History of return values from ``post_process_fn`` throughout the optimization.",
    )

    simulation: typing.Tuple[td.Simulation, ...] = pd.Field(
        (),
        title="Simulation History",
        description="History of ``td.Simulation`` instances throughout the optimization.",
    )

    opt_state: typing.Tuple[dict, ...] = pd.Field(
        (),
        title="Optimizer State History",
        description="History of optimizer states throughout the optimization.",
    )

    @property
    def history(self) -> typing.Dict[str, list]:
        """The history-containing fields as a dictionary of lists."""
        return dict(
            params=list(self.params),
            objective_fn_val=list(self.objective_fn_val),
            grad=list(self.grad),
            penalty=list(self.penalty),
            post_process_val=list(self.post_process_val),
            opt_state=list(self.opt_state),
        )

    @property
    def keys(self) -> typing.List[str]:
        """Keys stored in the history."""
        return list(self.history.keys())

    @property
    def last(self) -> typing.Dict[str, typing.Any]:
        """Dictionary of last values in ``self.history``."""
        return {key: value[-1] for key, value in self.history.items()}

    def get(self, key: str, index: int = -1) -> typing.Any:
        """Get the value from the history at a certain index (-1 means last)."""
        if key not in self.keys:
            raise KeyError(f"'{key}' not present in 'Result.history' dict with: {self.keys}.")
        values = self.history.get(key)
        if not len(values):
            raise ValueError(f"Can't get the last value of '{key}' as there is no history present.")
        return values[index]

    def get_last(self, key: str) -> typing.Any:
        """Get the last value from the history."""
        return self.get(key=key, index=-1)

    def get_sim(self, index: int = -1) -> typing.Union[td.Simulation, typing.List[td.Simulation]]:
        """Get the simulation at a specific index in the history (list of sims if multi)."""
        params = np.array(self.get(key="params", index=index))
        return self.design.to_simulation(params=params)

    def get_sim_data(
        self, index: int = -1, **kwargs
    ) -> typing.Union[td.SimulationData, typing.List[td.SimulationData]]:
        """Get the simulation data at a specific index in the history (list of simdata if multi)."""
        params = np.array(self.get(key="params", index=index))
        return self.design.to_simulation_data(params=params, **kwargs)

    @property
    def sim_last(self) -> typing.Union[td.Simulation, typing.List[td.Simulation]]:
        """The last simulation."""
        return self.get_sim(index=-1)

    def sim_data_last(self, **kwargs) -> td.SimulationData:
        """Run the last simulation and return its data."""
        return self.get_sim_data(index=-1, **kwargs)

    def plot_optimization(self):
        """Plot the optimization progress from the history."""
        plt.plot(self.objective_fn_val, label="objective function")
        plt.plot(self.post_process_val, label="post process function")
        plt.plot(self.penalty, label="combined penalty")
        plt.xlabel("iteration number")
        plt.ylabel("value")
        plt.legend()
