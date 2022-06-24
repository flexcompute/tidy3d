"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""

from typing import List, Tuple, Dict

import pydantic as pd
import numpy as np

from ...constants import HERTZ, C_0
from ...components.simulation import Simulation
from ...components.geometry import Box
from ...components.mode import ModeSpec
from ...components.monitor import ModeMonitor
from ...components.source import ModeSource, GaussianPulse
from ...components.data import SimulationData
from ...components.types import Direction, Ax
from ...components.viz import add_ax_if_none, equal_aspect
from ...components.base import Tidy3dBaseModel
from ...log import SetupError, log
from ...web.container import Batch, BatchData

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "data"


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
    mode_indices: Tuple[pd.NonNegativeInt, ...] = pd.Field(
        None,
        title="Mode Indices.",
        description="Indices into modes returned by the mode solver to use in the port. "
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


class ComponentModeler(Tidy3dBaseModel):
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
        "For each port, one simulation will be run with a modal source.",
    )
    freq: float = pd.Field(
        ...,
        title="Frequency",
        description="Frequency at which to evaluate the scattering matrix.",
        units=HERTZ,
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of the folder for the tasks on web.",
    )

    batch: Tidy3dBaseModel = pd.Field(
        None,
        title="Batch",
        description="Batch of task used to compute S matrix. Set internally.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("Simulation must not have sources.")
        return val

    @pd.validator("batch", always=True)
    def _set_batch(cls, val, values):
        """Initialize the batch."""
        if val is not None:
            return val

        sim_dict = cls._make_sims(
            ports=values.get("ports"), simulation=values.get("simulation"), freq=values.get("freq")
        )

        return Batch(simulations=sim_dict, folder_name=values.get("folder_name"))

    @staticmethod
    def _to_monitor(port: Port, freq: float) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=[freq],
            mode_spec=port.mode_spec,
            name=port.name,
        )

    @staticmethod
    def _to_sources(port: Port, freq: float) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        return [
            ModeSource(
                center=port.center,
                size=port.size,
                source_time=GaussianPulse(freq0=freq, fwidth=freq * FWIDTH_FRAC),
                mode_spec=port.mode_spec,
                mode_index=mode_index,
                direction=port.direction,
                name=port.name,
            )
            for mode_index in port.mode_indices
        ]

    @staticmethod
    def _shift_value_signed(simulation: Simulation, port: Port) -> float:
        """How far (signed) to shift the source from the monitor."""

        # get the grid boundaries and sizes along port normal from the simulation
        normal_axis = port.size.index(0.0)
        grid = simulation.grid
        grid_boundaries = grid.boundaries.to_list[normal_axis]
        grid_centers = grid.centers.to_list[normal_axis]

        # get the index of the grid cell where the port lies
        port_position = port.center[normal_axis]
        port_index = np.argwhere(port_position > grid_boundaries)[-1]

        # shift the port to the left
        if port.direction == "+":
            shifted_index = port_index - 2
            if shifted_index < 0:
                raise SetupError(
                    f"Port {port.name} normal is too close to boundary "
                    f"on -{'xyz'[normal_axis]} side."
                )

        # shift the port to the right
        else:
            shifted_index = port_index + 2
            if shifted_index >= len(grid_centers):
                raise SetupError(
                    f"Port {port.name} normal is too close to boundary "
                    f"on +{'xyz'[normal_axis]} side."
                )

        new_pos = grid_centers[shifted_index]
        return new_pos - port_position

    @classmethod
    def _shift_port(cls, simulation: Simulation, port: Port) -> Port:
        """Generate a new port shifted by the shift amount in normal direction."""

        shift_value = cls._shift_value_signed(simulation=simulation, port=port)
        center_shifted = list(port.center)
        center_shifted[port.size.index(0.0)] += shift_value
        port_shifted = port.copy(update=dict(center=center_shifted))
        return port_shifted

    @staticmethod
    def _task_name(port_source: Port, mode_index: int) -> str:
        """The name of a task, determined by the port of the source and mode index."""
        return f"smatrix_port{port_source.name}_mode{mode_index}"

    @classmethod
    def _make_sims(
        cls, ports: List[Port], simulation: Simulation, freq: float
    ) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

        mode_monitors = [cls._to_monitor(port=port, freq=freq) for port in ports]
        sim_dict = {}
        for port_source in ports:
            port_source = cls._shift_port(simulation=simulation, port=port_source)
            for mode_source in cls._to_sources(port=port_source, freq=freq):
                sim_copy = simulation.copy(
                    update=dict(
                        sources=[mode_source], monitors=list(simulation.monitors) + mode_monitors
                    )
                )
                task_name = cls._task_name(port_source, mode_source.mode_index)
                sim_dict[task_name] = sim_copy
        return sim_dict

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        sim_plot = self.simulation.copy(deep=True)
        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self._to_sources(port=port_source, freq=self.freq)[0]
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    def _run_sims(self, path_dir: str) -> BatchData:
        """Run :class:`Simulations` for each port and return the batch after saving."""
        self.batch.start()
        self.batch.monitor()
        batch_data = self.batch.load(path_dir=path_dir)
        return batch_data

    def _normalization_factor(self, port_source: Port, sim_data: SimulationData) -> complex:
        """Compute the normalization amplitude based on the measured input mode amplitude."""

        port_monitor_data = sim_data[port_source.name]
        mode_index = sim_data.simulation.sources[0].mode_index

        normalize_amp = port_monitor_data.amps.sel(
            f=self.freq,
            direction=port_source.direction,
            mode_index=mode_index,
        ).values

        normalize_n_eff = port_monitor_data.n_eff.sel(f=self.freq, mode_index=mode_index).values

        k0 = 2 * np.pi * C_0 / self.freq
        k_eff = k0 * normalize_n_eff
        shift_value = self._shift_value_signed(simulation=self.simulation, port=port_source)
        return normalize_amp * np.exp(1j * k_eff * shift_value)

    def _construct_smatrix(self, batch_data: BatchData) -> SMatrixType:
        """Post process batch to generate scattering matrix."""

        # load all data
        s_matrix_dict = {
            port_source.name: {port_monitor.name: [] for port_monitor in self.ports}
            for port_source in self.ports
        }

        # loop through source ports
        for port_source in self.ports:

            source_name = port_source.name

            # loop through all source injection indices
            for mode_index in port_source.mode_indices:

                # get the data for this source and compute normalization by injection
                task_name = self._task_name(port_source, mode_index)
                sim_data = batch_data[task_name]
                norm_factor = self._normalization_factor(port_source, sim_data)

                # loop through all monitors
                for port_monitor in self.ports:

                    monitor_name = port_monitor.name

                    # compute the mode amplitude data
                    dir_out = "-" if port_monitor.direction == "+" else "+"
                    mode_data = sim_data[monitor_name].amps.sel(f=self.freq, direction=dir_out)
                    amps_normalized = np.array(mode_data.values) / norm_factor
                    s_matrix_dict[source_name][monitor_name].append(amps_normalized)

            # convert to an array
            for port_monitor in self.ports:
                monitor_name = port_monitor.name
                mode_matrix = np.array(s_matrix_dict[source_name][monitor_name])
                s_matrix_dict[source_name][monitor_name] = mode_matrix

        return s_matrix_dict

    def solve(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Solves for the scattering matrix of the system."""
        log.warning(
            "`ComponentModeler.solve()` is renamed to `ComponentModeler.run()` "
            "'and will be removed in a later version."
        )
        return self.run(path_dir=path_dir)

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Solves for the scattering matrix of the system."""

        batch_data = self._run_sims(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Load an Smatrix from saved BatchData object."""

        if self.batch is None:
            raise SetupError("Component modeler has no batch saved. Run .run() to generate.")
        batch_data = self.batch.load(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)
