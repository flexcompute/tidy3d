# convenient container for the output of the inverse design (specifically the history)

import typing

import matplotlib.pyplot as plt
import jax.numpy as jnp
import pydantic.v1 as pd

import tidy3d as td

from .base import InvdesBaseModel
from .design import InverseDesign


# TODO: implement more convenience methods for exporting to figures?


class InverseDesignResult(InvdesBaseModel):
    """Container for the result of an ``InverseDesign.run()`` call."""

    design: InverseDesign = pd.Field(
        ...,
        title="Inverse Design Specification",
        description="Specification describing the inverse design problem we wish to optimize.",
    )

    params: typing.Tuple[jnp.ndarray, ...] = pd.Field(
        (),
        title="Parameter History",
        description="History of parameter arrays throughout the optimization.",
    )

    objective_fn_val: typing.Tuple[float, ...] = pd.Field(
        (),
        title="Objective Function History",
        description="History of objective function values throughout the optimization.",
    )

    grad: typing.Tuple[jnp.ndarray, ...] = pd.Field(
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

    opt_state: typing.Tuple[tuple, ...] = pd.Field(
        (),
        title="Optimizer State History",
        description="History of ``optax`` optimizer states throughout the optimization.",
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
            simulation=list(self.simulation),
            opt_state=list(self.opt_state),
        )

    @property
    def keys(self) -> typing.List[str]:
        """Keys stored in the history."""
        return list(self.history.keys())

    @property
    def final(self) -> typing.Dict[str, typing.Any]:
        """Dictionary of final values in ``self.history``."""
        return {key: value[-1] for key, value in self.history.items()}

    def get_final(self, key: str) -> typing.Any:
        """Get the final value of a field in the ``self.history`` by key."""
        if key not in self.keys:
            raise KeyError(f"'{key}' not present in 'Result.history' dict with: {self.keys}.")
        values = self.history.get(key)
        if not len(values):
            raise ValueError(
                f"Can't get the final value of '{key}' as there is no history present."
            )
        return values[-1]

    @property
    def sim_final(self) -> td.Simulation:
        """The final simulation."""
        return self.get_final("simulation")

    def sim_data_final(self, task_name: str, **run_kwargs) -> td.SimulationData:
        """Run the final simulation and return its data."""
        return td.web.run(self.sim_final, task_name=task_name, **run_kwargs)

    def plot_optimization(self):
        """Plot the optimization progress from the history."""
        plt.plot(self.objective_fn_val, label="objective function")
        plt.plot(self.post_process_val, label="post process function")
        plt.plot(self.penalty, label="combined penalty")
        plt.xlabel("iteration number")
        plt.ylabel("value")
        plt.legend()
