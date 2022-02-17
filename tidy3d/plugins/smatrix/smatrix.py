"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""

from typing import List, Tuple, Dict
import os

import pydantic as pd
import numpy as np

from ... import Simulation, Box, ModeSpec, ModeMonitor, ModeSource, GaussianPulse
from ...components.types import Direction, Ax
from ...constants import HERTZ
from ...web.container import Batch, DEFAULT_DATA_DIR

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10


class Port(Box):
    """Specifies a port in the scattering matrix."""

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )
    mode_spec: ModeSpec = pd.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Specifies how the mode solver will solve for the modes of the port.",
    )
    mode_indices: Tuple[pd.NonNegativeInt] = pd.Field(
        None,
        title="Mode Indices.",
        description="Indices into modes returned to mode solver to use in the port. "
        "If ``None``, all modes returned by the mode solver are used in the scattering matrix. "
        "Otherwise, the scattering matrix will include elements for each supplied mode index.",
    )
    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    @pd.validator("mode_indices", always=True)
    def evaluate_mode_indices(cls, val, values):
        """Evaluates mode indices based on number of modes in mode spec."""
        if val is None:
            num_modes = values.get("mode_spec").num_modes
            val = tuple(range(num_modes))
        return val


"""
s_matrix[port_name_in][port_name_out] gives a numpy array of shape (m, n)
relating the coupling amplitudes between the m and n mode orders in the two ports, respectively.
"""
SMatrixType = Dict[str, np.ndarray]


class ComponentModeler(pd.BaseModel):
    """Tool for modeling devices and computing scattering matrix elements."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources or monitors present.",
    )
    ports: List[Port] = pd.Field(
        [],
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each port, one siulation will be run with a modal source.",
    )
    freq: float = pd.Field(
        ...,
        title="Frequency",
        description="Frequency at which to evaluate the scattering matrix.",
        units=HERTZ,
    )
    batch_path: str = pd.Field(
        "smatrix_batch.json",
        title="Path to Batch",
        description="Path to the file where the Batch object will be saved.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError(f"Simulation must not have sources.")
        return val

    def _to_monitor(self, port: Port) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=[self.freq],
            mode_spec=port.mode_spec,
            name=port.name,
        )

    def _to_sources(self, port: Port) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        return [
            ModeSource(
                center=port.center,
                size=port.size,
                source_time=GaussianPulse(freq0=self.freq, fwidth=self.freq * FWIDTH_FRAC),
                mode_spec=port.mode_spec,
                mode_index=mode_index,
                direction=port.direction,
                name=port.name,
            )
            for mode_index in port.mode_indices
        ]

    def plot_sim(self, x: float = None, y: float = None, z: float = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        sim_plot = self.simulation.copy(deep=True)
        for port_source in self.ports:
            mode_source_0 = self._to_sources(port_source)[0]
            sim_plot.sources.append(mode_source_0)
        return sim_plot.plot(x=x, y=y, z=z)

    def _shift_port(self, port: Port) -> Port:
        """Generate a new port shifted by one grid cell in normal direction."""

        normal_index = port.size.index(0.0)
        dl = self.simulation.grid_size[normal_index]
        if not isinstance(dl, float):
            raise NotImplementedError("doesn't support nonuniform. How many grid cells to shift?")
        shift_value = dl if port.direction == "+" else -1 * dl

        center_shifted = list(port.center)
        center_shifted[normal_index] += shift_value
        port_shifted = port.copy(deep=True)
        port_shifted.center = center_shifted
        return port_shifted

    def _make_sims(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""
        sim_dict = {}
        for port_source in self.ports:
            for mode_source in self._to_sources(port_source):
                sim_copy = self.simulation.copy(deep=True)
                sim_copy.sources = [mode_source]
                for port_monitor in self.ports:
                    if port_source == port_monitor:
                        port_monitor = self._shift_port(port_source)
                    mode_monitor = self._to_monitor(port_monitor)
                    sim_copy.monitors.append(mode_monitor)
                    task_name = f"smatrix_port{port_source.name}_mode{mode_source.mode_index}"
                    sim_dict[task_name] = sim_copy
        return sim_dict

    def _run_sims(self, sim_dict: Dict[str, Simulation], folder_name: str, path_dir: str) -> Batch:
        """Run :class:`Simulations` for each port and return the batch after saving."""
        batch = Batch(simulations=sim_dict, folder_name=folder_name)
        batch.upload()

        # save after upload so that the jobs are saved too for later
        batch.to_file(self.batch_path)

        batch.start()
        batch.monitor()
        return batch

    def _construct_smatrix(self, batch: Batch, path_dir: str) -> SMatrixType:
        """Post process batch to generate scattering matrix."""

        s_matrix_dict = {}
        for port_source, (task_name, sim_data) in zip(self.ports, batch.items(path_dir=path_dir)):
            port_monitor_data = sim_data[port_source.name]
            normalize_amp = port_monitor_data.amps.sel(
                f=self.freq,
                direction=port_source.direction,
                mode_index=sim_data.simulation.sources[0].mode_index,
            ).values
            s_matrix_col = {}
            for monitor in sim_data.simulation.monitors:
                dir_out = "-" if port_source.direction == "+" else "+"
                mode_data = sim_data[monitor.name].amps.sel(f=self.freq, direction=dir_out)
                s_matrix_col[monitor.name] = np.array(mode_data.values) / normalize_amp
            s_matrix_dict[port_source.name] = s_matrix_col
        return s_matrix_dict

    def solve(self, folder_name: str = "default", path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Solves for the scattering matrix of the system."""

        sim_dict = self._make_sims()
        batch = self._run_sims(sim_dict=sim_dict, folder_name=folder_name, path_dir=path_dir)
        return self._construct_smatrix(batch=batch, path_dir=path_dir)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Load an Smatrix from a saved batch."""
        batch = Batch.from_file(self.batch_path)
        return self._construct_smatrix(batch=batch, path_dir=path_dir)
