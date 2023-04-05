"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""

from typing import List, Tuple, Optional, Callable, Dict

import pydantic as pd
import numpy as np

from ...constants import HERTZ, C_0
from ...components.simulation import Simulation
from ...components.geometry import Box
from ...components.mode import ModeSpec
from ...components.monitor import ModeMonitor
from ...components.source import ModeSource, GaussianPulse
from ...components.data.sim_data import SimulationData
from ...components.data.data_array import DataArray
from ...components.types import Direction, Ax, Complex
from ...components.viz import add_ax_if_none, equal_aspect
from ...components.base import Tidy3dBaseModel
from ...exceptions import SetupError, Tidy3dKeyError
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
    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )


MatrixIndex = Tuple[str, pd.NonNegativeInt]  # the 'i' in S_ij
Element = Tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij


class SMatrixDataArray(DataArray):
    """Scattering matrix elements.

    Example
    -------
    >>> port_in = ['port1', 'port2']
    >>> port_out = ['port1', 'port2']
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     mode_index_in=mode_index_in,
    ...     mode_index_out=mode_index_out,
    ...     f=f
    ... )
    >>> fd = SMatrixDataArray((1 + 1j) * np.random.random((2, 2, 2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "mode_index_out", "port_in", "mode_index_in", "f")
    _data_attrs = {"long_name": "scattering matrix element"}


class ComponentModeler(Tidy3dBaseModel):
    """Tool for modeling devices and computing scattering matrix elements."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )
    ports: Tuple[Port, ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )
    freqs: List[float] = pd.Field(
        ...,
        title="Frequencies",
        description="List of frequencies at which to evaluate the scattering matrix.",
        units=HERTZ,
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of the folder for the tasks on web.",
    )
    element_mappings: Tuple[Tuple[Element, Element, Complex], ...] = pd.Field(
        (),
        title="Element Mappings",
        description="Mapping between elements of the scattering matrix, "
        "as specified by pairs of ``(port name, mode index)`` matrix indices, where the "
        "first element of the pair is the output and the second element of the pair is the input."
        "Each item of ``element_mappings`` is a tuple of ``(element1, element2, c)``, where "
        "the scattering matrix ``Smatrix[element2]`` is set equal to ``c * Smatrix[element1]``."
        "If all elements of a given column of the scattering matrix are defined by "
        " ``element_mappings``, the simulation corresponding to this column "
        "is skipped automatically.",
    )
    run_only: Optional[Tuple[MatrixIndex, ...]] = pd.Field(
        None,
        title="Run Only",
        description="If given, a tuple of matrix indices, specified by (:class:`.Port`, ``int``),"
        " to run only, excluding the other rows from the scattering matrix. "
        "If this option is used, "
        "the data corresponding to other inputs will be missing in the resulting matrix.",
    )
    verbose: bool = pd.Field(
        False,
        title="Verbosity",
        description="Whether the :class:`.ComponentModeler` should print status and progressbars.",
    )
    batch: Optional[Batch] = pd.Field(
        None,
        title="Batch",
        description="Batch of generated simulations needed for each row of the scattering matrix."
        "Should be left ``None`` in almost all cases, which will generate the proper batch "
        "internally and store it as ``ComponentModeler.batch``.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("'ComponentModeler.simulation' must not have any sources.")
        return val

    @pd.validator("batch", always=True)
    def _set_batch(cls, val, values):
        """Initialize the batch if not supplied."""

        # if supplied, return it
        if val is not None:
            return val

        # otherwise, generate all sims and make a new batch
        sim_dict = cls.make_sim_dict(values=values)
        return Batch(
            simulations=sim_dict,
            folder_name=values.get("folder_name"),
            verbose=values.get("verbose"),
        )

    @classmethod
    def make_sim_dict(cls, values: dict) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

        sim_dict = {}
        ports = values.get("ports")
        simulation = values.get("simulation")
        freqs = values.get("freqs")

        mode_monitors = [
            cls._to_monitor(port=port, freqs=freqs) for port in ports
        ]  # pylint:disable=protected-access

        for (port_name, mode_index) in cls.matrix_indices_run_sim(
            ports=ports,
            run_only=values.get("run_only"),
            element_mappings=values.get("element_mappings"),
        ):

            port = cls.get_port_by_name(port_name=port_name, ports=ports)

            port_source = cls._shift_port(
                simulation=simulation, port=port
            )  # pylint:disable=protected-access
            mode_source = cls._to_source(
                port=port_source, mode_index=mode_index, freqs=freqs
            )  # pylint:disable=protected-access

            new_mnts = list(simulation.monitors) + mode_monitors
            sim_copy = simulation.copy(update=dict(sources=[mode_source], monitors=new_mnts))
            task_name = cls._task_name(port=port, mode_index=mode_index)
            sim_dict[task_name] = sim_copy
        return sim_dict

    @classmethod
    def matrix_indices_monitor(cls, ports: Tuple[Port, ...]) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the possible matrix indices (port, mode_index) in the Component Modeler."""
        matrix_indices = []
        for port in ports:
            for mode_index in range(port.mode_spec.num_modes):
                matrix_indices.append((port.name, mode_index))
        return tuple(matrix_indices)

    @classmethod
    def matrix_indices_source(
        cls, ports: Tuple[Port, ...], run_only: Tuple[MatrixIndex, ...] = None
    ) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""
        return run_only if run_only is not None else cls.matrix_indices_monitor(ports=ports)

    @classmethod
    def matrix_indices_run_sim(
        cls,
        ports: Tuple[Port, ...],
        element_mappings: Dict[Element, Dict[Element, Callable[[complex], complex]]] = None,
        run_only: Tuple[MatrixIndex, ...] = None,
    ) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""

        if element_mappings is None or element_mappings == {}:
            return cls.matrix_indices_source(ports=ports, run_only=run_only)

        # all the (i, j) pairs in `S_ij` that are tagged as covered by `element_mappings`
        elements_determined_by_map = [element_out for (_, element_out, _) in element_mappings]

        # loop through rows of the full s matrix and record rows that still need running.
        source_indices_needed = []
        for col_index in cls.matrix_indices_source(ports=ports, run_only=run_only):

            # loop through columns and keep track of whether each element is covered by mapping.
            matrix_elements_covered = []
            for row_index in cls.matrix_indices_monitor(ports=ports):
                element = (row_index, col_index)
                element_covered_by_map = element in elements_determined_by_map
                matrix_elements_covered.append(element_covered_by_map)

            # if any matrix elements in row still not covered by map, a source is needed for row.
            if not all(matrix_elements_covered):
                source_indices_needed.append(col_index)

        return source_indices_needed

    @staticmethod
    def get_port_by_name(ports: Tuple[Port, ...], port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @staticmethod
    def _to_monitor(port: Port, freqs: List[float]) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=freqs,
            mode_spec=port.mode_spec,
            name=port.name,
        )

    @staticmethod
    def _to_source(port: Port, mode_index: int, freqs: List[float]) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        freq0 = np.mean(freqs)
        fdiff = max(freqs) - min(freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        return ModeSource(
            center=port.center,
            size=port.size,
            source_time=GaussianPulse(freq0=freq0, fwidth=fwidth),
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
        port_pos_gt_grid_bounds = np.argwhere(port_position > grid_boundaries)

        # no port index can be determined
        if len(port_pos_gt_grid_bounds) == 0:
            raise SetupError(f"Port position '{port_position}' outside of simulation bounds.")
        port_index = port_pos_gt_grid_bounds[-1]

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
        return f"smatrix_{port.name}_{mode_index}"

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self._to_source(port=port_source, freqs=self.freqs, mode_index=0)
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

        normalize_amps = port_monitor_data.amps.sel(
            f=self.freqs,
            direction=port_source.direction,
            mode_index=mode_index,
        ).values

        normalize_n_eff = port_monitor_data.n_eff.sel(f=self.freqs, mode_index=mode_index).values

        k0s = 2 * np.pi * C_0 / np.array(self.freqs)
        k_effs = k0s * normalize_n_eff
        shift_value = self._shift_value_signed(simulation=self.simulation, port=port_source)
        return normalize_amps * np.exp(1j * k_effs * shift_value)

    @property
    def max_mode_index(self) -> Tuple[int, int]:
        """maximum mode indices for the smatrix dataset for the in and out ports, respectively."""

        def get_max_mode_indices(matrix_elements: Tuple[str, int]) -> int:
            """Get the maximum mode index for a list of (port name, mode index)."""
            return max(mode_index for _, mode_index in matrix_elements)

        matrix_elements_out = self.matrix_indices_monitor(ports=self.ports)
        matrix_elements_in = self.matrix_indices_source(ports=self.ports, run_only=self.run_only)

        max_mode_index_out = get_max_mode_indices(matrix_elements_out)
        max_mode_index_in = get_max_mode_indices(matrix_elements_in)

        return max_mode_index_out, max_mode_index_in

    @property
    def port_names(self) -> Tuple[List[str], List[str]]:
        """List of port names for inputs and outputs, respectively."""

        def get_port_names(matrix_elements: Tuple[str, int]) -> List[str]:
            """Get the port names from a list of (port name, mode index)."""
            port_names = []
            for port_name, _ in matrix_elements:
                if port_name not in port_names:
                    port_names.append(port_name)
            return port_names

        matrix_elements_in = self.matrix_indices_source(ports=self.ports, run_only=self.run_only)
        matrix_elements_out = self.matrix_indices_monitor(ports=self.ports)

        port_names_in = get_port_names(matrix_elements_in)
        port_names_out = get_port_names(matrix_elements_out)

        return port_names_out, port_names_in

    def _construct_smatrix(  # pylint:disable=too-many-locals
        self, batch_data: BatchData
    ) -> SMatrixDataArray:
        """Post process batch to generate scattering matrix."""

        max_mode_index_out, max_mode_index_in = self.max_mode_index
        num_modes_out = max_mode_index_out + 1
        num_modes_in = max_mode_index_in + 1
        port_names_out, port_names_in = self.port_names

        values = np.zeros(
            (len(port_names_out), len(port_names_in), num_modes_out, num_modes_in, len(self.freqs)),
            dtype=complex,
        )
        coords = dict(
            port_out=port_names_out,
            port_in=port_names_in,
            mode_index_out=range(num_modes_out),
            mode_index_in=range(num_modes_in),
            f=self.freqs,
        )

        s_matrix = SMatrixDataArray(values, coords=coords)

        # loop through source ports
        for col_index in self.matrix_indices_run_sim(
            ports=self.ports, run_only=self.run_only, element_mappings=self.element_mappings
        ):

            port_name_in, mode_index_in = col_index
            port_in = self.get_port_by_name(port_name=port_name_in, ports=self.ports)

            sim_data = batch_data[self._task_name(port=port_in, mode_index=mode_index_in)]

            for row_index in self.matrix_indices_monitor(ports=self.ports):

                port_name_out, mode_index_out = row_index
                port_out = self.get_port_by_name(port_name=port_name_out, ports=self.ports)

                # directly compute the element
                mode_amps_data = sim_data[port_out.name].copy().amps
                dir_out = "-" if port_out.direction == "+" else "+"
                amp = mode_amps_data.sel(f=self.freqs, direction=dir_out, mode_index=mode_index_out)
                source_norm = self._normalization_factor(port_in, sim_data)
                s_matrix_elements = np.array(amp.data) / np.array(source_norm)
                s_matrix.loc[
                    dict(
                        port_in=port_name_in,
                        mode_index_in=mode_index_in,
                        port_out=port_name_out,
                        mode_index_out=mode_index_out,
                    )
                ] = s_matrix_elements

        # element can be determined by user-defined mapping
        for ((row_in, col_in), (row_out, col_out), mult_by) in self.element_mappings:

            port_out_from, mode_index_out_from = row_in
            port_in_from, mode_index_in_from = col_in
            coords_from = dict(
                port_in=port_in_from,
                mode_index_in=mode_index_in_from,
                port_out=port_out_from,
                mode_index_out=mode_index_out_from,
            )

            port_out_to, mode_index_out_to = row_out
            port_in_to, mode_index_in_to = col_out
            coords_to = dict(
                port_in=port_in_to,
                mode_index_in=mode_index_in_to,
                port_out=port_out_to,
                mode_index_out=mode_index_out_to,
            )
            s_matrix.loc[coords_to] = mult_by * s_matrix.loc[coords_from].values

        return s_matrix

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixDataArray:
        """Solves for the scattering matrix of the system."""

        batch_data = self._run_sims(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> SMatrixDataArray:
        """Load an Smatrix from saved BatchData object."""

        if self.batch is None:
            raise SetupError("Component modeler has no batch saved. Run .run() to generate.")
        batch_data = self.batch.load(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)
