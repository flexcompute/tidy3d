"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""

from typing import List, Tuple, Dict, Optional

import pydantic as pd
import numpy as np

from ...constants import HERTZ, C_0
from ...components.base import cached_property
from ...components.simulation import Simulation
from ...components.geometry import Box
from ...components.mode import ModeSpec
from ...components.monitor import ModeMonitor
from ...components.source import ModeSource, GaussianPulse
from ...components.data.sim_data import SimulationData
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
MatrixIndex = Tuple[Port, pd.NonNegativeInt]
Element = Tuple[MatrixIndex, MatrixIndex]
SMatrixType = Dict[Element, Element]


class ComponentModeler(Tidy3dBaseModel):
    """Tool for modeling devices and computing scattering matrix elements."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources or monitors present.",
    )
    ports: Tuple[Port, ...] = pd.Field(
        (),
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
    element_mappings: Dict[Element, Tuple[Element, bool]] = pd.Field(
        {},
        title="Element Mappings",
        description="Mapping between matrix indices of the scattering matrix, "
        "specified by (:class:`.Port`, `int`). "
        "``element_mappings(port, mode_index)`` returns ``(port, mode_index), pos_sign``, "
        "where the ``(port, mode_index)`` refers to the output matrix index "
        "and ``pos_sign`` determines whether the element should be copied with a positive sign.",
    )
    run_only: Optional[Tuple[MatrixIndex, ...]] = pd.Field(
        None,
        title="Run Only",
        description="If specified, a tuple of matrix indices, specified by (:class:`.Port`, `int`),"
        " to run only, excluding the other rows from the scattering matrix. "
        "If this option is used, the resulting scattering matrix will not be square or complete.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("Simulation must not have extraneous `sources`.")
        return val

    @cached_property
    def matrix_indices_monitor(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the possible matrix indices (port, mode_index) in the Component Modeler."""
        matrix_indices = []
        for port in self.ports:
            for mode_index in port.mode_indices:
                matrix_indices.append((port, mode_index))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_source(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""

        return self.run_only if self.run_only is not None else self.matrix_indices_monitor

    @cached_property
    def matrix_indices_run_sim(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""

        # for a given source_index, if all of the monitor indices are covered by mappings, can skip.

        elements_determined_by_map = [element for _, (element, _) in self.element_mappings.items()]

        matrix_indices_needed = []
        for row_index in self.matrix_indices_source:

            row_needed = False
            for col_index in self.matrix_indices_monitor:
                element = (row_index, col_index)
                if element not in elements_determined_by_map:
                    row_needed = True
                    break
            if row_needed:
                matrix_indices_needed.append(row_index)

        return matrix_indices_needed

    @cached_property
    def batch(self) -> Batch:
        """Batch containing all of the simulations needed for the scattering matrix."""

        return Batch(simulations=self.sim_dict, folder_name=self.folder_name)

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
    def _to_source(port: Port, mode_index: int, freq: float) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        return ModeSource(
            center=port.center,
            size=port.size,
            source_time=GaussianPulse(freq0=freq, fwidth=freq * FWIDTH_FRAC),
            mode_spec=port.mode_spec,
            mode_index=mode_index,
            direction=port.direction,
            name=port.name,
        )

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
    def _task_name(port: Port, mode_index: int) -> str:
        """The name of a task, determined by the port of the source and mode index."""
        return f"smatrix_port{port.name}_mode{mode_index}"

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

        _sim_dict = {}

        mode_monitors = [
            self._to_monitor(port=port, freq=self.freq) for port in self.ports
        ]  # pylint:disable=protected-access
        for (port, mode_index) in self.matrix_indices_run_sim:

            port_source = self._shift_port(
                simulation=self.simulation, port=port
            )  # pylint:disable=protected-access
            mode_source = self._to_source(
                port=port_source, mode_index=mode_index, freq=self.freq
            )  # pylint:disable=protected-access

            new_mnts = list(self.simulation.monitors) + mode_monitors
            sim_copy = self.simulation.copy(update=dict(sources=[mode_source], monitors=new_mnts))
            task_name = self._task_name(port=port, mode_index=mode_index)
            _sim_dict[task_name] = sim_copy
        return _sim_dict

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self._to_source(port=port_source, freq=self.freq, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    def _run_sims(self, path_dir: str) -> BatchData:
        """Run :class:`Simulations` for each port and return the batch after saving."""
        self.batch.start()
        self.batch.monitor()
        return self.batch.load(path_dir=path_dir)

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

    def _construct_smatrix(  # pylint:disable=too-many-locals
        self, batch_data: BatchData
    ) -> SMatrixType:
        """Post process batch to generate scattering matrix."""

        s_matrix = {}

        # loop through source ports
        for row_index in self.matrix_indices_source:

            port_in, mode_index_in = row_index

            for col_index in self.matrix_indices_monitor:

                port_out, mode_index_out = row_index

                element = (row_index, col_index)

                # element already filled in
                if element in s_matrix:
                    continue

                # element can be determined by user-defined mapping
                for element_in, (element_out, has_pos_sign) in self.element_mappings.items():
                    if element == element_in:
                        sign = 1 if has_pos_sign else -1
                        s_matrix[element_out] = sign * s_matrix[element_in]
                        continue

                # directly compute the element
                sim_data = batch_data[self._task_name(port=port_in, mode_index=mode_index_in)]
                mode_amps_data = sim_data[port_out.name].amps
                dir_out = "-" if port_out.direction == "+" else "+"
                amp = mode_amps_data.sel(f=self.freq, direction=dir_out, mode_index=mode_index_out)
                source_norm = self._normalization_factor(port_in, sim_data)

                s_matrix[element] = amp / source_norm

        return s_matrix

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
