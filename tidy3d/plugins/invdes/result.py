# convenient container for the output of the inverse design (specifically the history)

import typing

import matplotlib.pyplot as plt

import tidy3d as td


class OptimizeResult(td.components.base.Tidy3dBaseModel):
    """Container for the result of an ``InverseDesign.run()`` call."""

    history: typing.Dict[str, typing.Any]  # TODO: replace this with the actual fields.

    @property
    def keys(self) -> typing.List[str]:
        """Keys stored in the history."""
        return self.history.keys()

    @property
    def final(self) -> typing.Dict[str, typing.Any]:
        """Dictionary of final values in ``self.history``."""
        return {key: value[-1] for key, value in self.history.items()}

    def get_final(self, key: str) -> typing.Any:
        """Get the final value of a field in the ``self.history`` by key."""
        if key not in self.keys:
            raise KeyError(f"'{key}' not present in ``Result.history`` dict with: {self.keys}.")
        return self.history.get(key)[-1]

    @property
    def sim_final(self) -> td.Simulation:
        """The final simulation."""
        return self.get_final("simulation")

    def sim_data_final(self, **run_kwargs) -> td.SimulationData:
        """Run the final simulation and return its data."""
        return td.web.run(self.sim_final, **run_kwargs)

    # TODO: convenience methods for all of these GDS methods? or just refer to ``self.sim_final``?
    def to_gds_file(self, fname, **to_gds_file_kwargs) -> None:
        """Export the final simulation to GDS using ``Simulation.to_gds``."""
        sim_final = self.get_final("simulation")
        return sim_final.to_gds_file(fname, **to_gds_file_kwargs)

    def plot_optimization(self):
        """Plot the optimization progress from the history."""
        objective_fn_values = self.history.get("objective_fn_val")
        post_process_values = self.history.get("post_process_val")
        penalty_values = self.history.get("penalty")

        plt.plot(objective_fn_values, label="objective function")
        plt.plot(post_process_values, label="post process function")
        plt.plot(penalty_values, label="combined penalty")
        plt.xlabel("iteration number")
        plt.ylabel("value")
        plt.legend()

    # TODO: implement more convenience methods for exporting to figures, gds, etc.

    # TODO: implement way to continue optimization if you want more
