"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""
# TODO: The names "ComponentModeler" and "Port" should be changed to "ModalComponentModeler" and
# "ModalPort" to explicitly differentiate these from "TerminalComponentModeler" and "LumpedPort".
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Union
import os
from abc import ABC, abstractmethod

import pydantic.v1 as pd
import numpy as np

from ...constants import HERTZ, OHM, C_0
from ...components.simulation import Simulation
from ...components.geometry.base import Box
from ...components.structure import MeshOverrideStructure
from ...components.mode import ModeSpec
from ...components.monitor import ModeMonitor, FieldMonitor
from ...components.source import ModeSource, GaussianPulse, UniformCurrentSource
from ...components.data.sim_data import SimulationData
from ...components.data.data_array import DataArray
from ...components.types import Direction, Ax, Complex, FreqArray, Axis
from ...components.viz import add_ax_if_none, equal_aspect
from ...components.base import Tidy3dBaseModel, cached_property
from ...components.validators import assert_plane
from ...exceptions import SetupError, Tidy3dKeyError, ValidationError
from ...log import log
from ...web.api.container import BatchData, Batch

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "."
DEFAULT_PORT_NUM_CELLS = 3


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


class LumpedPort(Box):
    """Lumped port source."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    voltage_axis: Axis = pd.Field(
        ...,
        title="Voltage Integration Axis",
        description="Specifies the axis along which the E-field line integral is performed when "
        "computing the port voltage. The integration axis must lie in the plane of the port.",
    )

    impedance: Complex = pd.Field(
        50,
        title="Reference impedance",
        description="Reference port impedance for scattering parameter computation.",
        units=OHM,
    )

    refine_mesh: bool = pd.Field(
        True,
        title="Port mesh refinement",
        description="If ``True``, enables mesh refinement in the port region to ensure that "
        "the port voltage and current are computed accurately.",
    )

    num_grid_cells: pd.PositiveInt = pd.Field(
        DEFAULT_PORT_NUM_CELLS,
        title="Port grid cells",
        description="Number of mesh grid cells associated with the port along each direction. "
        "This is used only if ``refine_mesh`` is ``True``.",
    )

    _plane_validator = assert_plane()

    @cached_property
    def injection_axis(self):
        """Injection axis of the port."""
        return self.size.index(0.0)

    @pd.validator("voltage_axis", always=True)
    def _voltage_axis_in_plane(cls, val, values):
        """Ensure voltage integration axis is in the port's plane."""
        size = values.get("size")
        if val == size.index(0.0):
            raise ValidationError("'voltage_axis' must lie in the port's plane.")
        return val

    @cached_property
    def current_axis(self) -> Axis:
        """Integration axis for computing the port current via the magnetic field."""
        return 3 - self.injection_axis - self.voltage_axis


MatrixIndex = Tuple[str, pd.NonNegativeInt]  # the 'i' in S_ij
Element = Tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij


class ModalPortDataArray(DataArray):
    """Port parameter matrix elements for modal ports.

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
    >>> fd = ModalPortDataArray((1 + 1j) * np.random.random((2, 2, 2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "mode_index_out", "port_in", "mode_index_in", "f")
    _data_attrs = {"long_name": "modal port matrix element"}


class LumpedPortDataArray(DataArray):
    """Port parameter matrix elements for lumped ports.

    Example
    -------
    >>> port_in = ['port1', 'port2']
    >>> port_out = ['port1', 'port2']
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     f=f
    ... )
    >>> fd = LumpedPortDataArray((1 + 1j) * np.random.random((2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "port_in", "f")
    _data_attrs = {"long_name": "lumped port matrix element"}


class AbstractComponentModeler(ABC, Tidy3dBaseModel):
    """Tool for modeling devices and computing port parameters."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )

    ports: Tuple[Union[Port, LumpedPort], ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )

    freqs: FreqArray = pd.Field(
        ...,
        title="Frequencies",
        description="Array or list of frequencies at which to compute port parameters.",
        units=HERTZ,
    )

    remove_dc_component: bool = pd.Field(
        True,
        title="Remove DC Component",
        description="Whether to remove the DC component in the Gaussian pulse spectrum. "
        "If ``True``, the Gaussian pulse is modified at low frequencies to zero out the "
        "DC component, which is usually desirable so that the fields will decay. However, "
        "for broadband simulations, it may be better to have non-vanishing source power "
        "near zero frequency. Setting this to ``False`` results in an unmodified Gaussian "
        "pulse spectrum which can have a nonzero DC component.",
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of the folder for the tasks on web.",
    )

    verbose: bool = pd.Field(
        False,
        title="Verbosity",
        description="Whether the :class:`.ComponentModeler` should print status and progressbars.",
    )

    callback_url: str = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    path_dir: str = pd.Field(
        DEFAULT_DATA_DIR,
        title="Directory Path",
        description="Base directory where data and batch will be downloaded.",
    )

    solver_version: str = pd.Field(
        None,
        title="Solver Version",
        description_str="Custom solver version to use, "
        "otherwise uses default for the current front end version.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("'AbstractComponentModeler.simulation' must not have any sources.")
        return val

    @staticmethod
    def _task_name(port: Port, mode_index: int = None) -> str:
        """The name of a task, determined by the port of the source and mode index, if given."""
        if mode_index is not None:
            return f"smatrix_{port.name}_{mode_index}"
        return f"smatrix_{port.name}"

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

    @cached_property
    def batch(self) -> Batch:
        """Batch associated with this component modeler."""

        # first try loading the batch from file, if it exists
        batch_path = self._batch_path
        if os.path.exists(batch_path):
            return Batch.from_file(fname=batch_path)

        return Batch(
            simulations=self.sim_dict,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            verbose=self.verbose,
            solver_version=self.solver_version,
        )

    @cached_property
    def batch_path(self) -> str:
        """Path to the batch saved to file."""

        return self.batch._batch_path(path_dir=DEFAULT_DATA_DIR)

    def get_path_dir(self, path_dir: str) -> None:
        """Check whether the supplied 'path_dir' matches the internal field value."""

        if path_dir not in (DEFAULT_DATA_DIR, self.path_dir):
            log.warning(
                f"'ComponentModeler' method was supplied a 'path_dir' of '{path_dir}' "
                f"when its internal 'path_dir' field was set to '{self.path_dir}'. "
                "The passed value will be deprecated in later versions. "
                "Please set the internal 'path_dir' field to the desired value and "
                "remove the 'path_dir' from the method argument. "
                f"Using supplied '{path_dir}'."
            )
            return path_dir

        return self.path_dir

    @cached_property
    def _batch_path(self) -> str:
        """Where we store the batch for this ComponentModeler instance after the run."""
        return os.path.join(self.path_dir, "batch" + str(hash(self)) + ".json")

    def _run_sims(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Run :class:`Simulations` for each port and return the batch after saving."""
        batch = self.batch

        # TEMP
        # import sys
        # TIDY3D_CORE_PATH = '/home/shashwat/flexcompute/repositories/tidy3d-core'
        # sys.path.insert(0, TIDY3D_CORE_PATH)
        # from tidy3d_backend.run import run_sim
        # batch_data = {}
        # for name, sim in self.sim_dict.items():
        #     batch_data[name] = run_sim(sim, mpi=1)

        # TEMP
        batch_data = batch.run(path_dir=path_dir)

        batch.to_file(self._batch_path)
        return batch_data

    def get_port_by_name(self, port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in self.ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @abstractmethod
    def _construct_smatrix(self, batch_data: BatchData) -> DataArray:
        """Post process `BatchData` to generate scattering matrix."""

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Solves for the scattering matrix of the system."""
        path_dir = self.get_path_dir(path_dir)

        batch_data = self._run_sims(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Load a scattering matrix from saved `BatchData` object."""
        path_dir = self.get_path_dir(path_dir)

        batch_data = BatchData.load(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    @staticmethod
    def s_to_z(s_matrix: DataArray, reference: Complex) -> DataArray:
        """Get the impedance matrix given the scattering matrix and a reference impedance."""

        # move the input and output port dimensions to the end, for ease of matrix operations
        dims = list(s_matrix.dims)
        dims.append(dims.pop(dims.index("port_out")))
        dims.append(dims.pop(dims.index("port_in")))
        z_matrix = s_matrix.copy(deep=True).transpose(*dims)
        s_vals = z_matrix.values

        eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))
        z_vals = np.matmul(AbstractComponentModeler.inv(eye - s_vals), (eye + s_vals)) * reference

        z_matrix.data = z_vals
        return z_matrix

    @staticmethod
    def ab_to_s(a_matrix: DataArray, b_matrix: DataArray) -> DataArray:
        """Get the scattering matrix given the power wave matrices."""

        # move the input and output port dimensions to the end, for ease of matrix operations
        assert a_matrix.dims == b_matrix.dims
        dims = list(a_matrix.dims)
        dims.append(dims.pop(dims.index("port_out")))
        dims.append(dims.pop(dims.index("port_in")))

        s_matrix = a_matrix.copy(deep=True).transpose(*dims)
        a_vals = s_matrix.copy(deep=True).transpose(*dims).values
        b_vals = b_matrix.copy(deep=True).transpose(*dims).values

        s_vals = np.matmul(b_vals, AbstractComponentModeler.inv(a_vals))

        s_matrix.data = s_vals
        return s_matrix

    @staticmethod
    def inv(matrix: DataArray):
        """Helper to invert a port matrix."""
        return np.linalg.inv(matrix)


class ComponentModeler(AbstractComponentModeler):
    """Tool for modeling devices and computing scattering matrix elements with modal ports."""

    ports: Tuple[Port, ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
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

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

        sim_dict = {}
        mode_monitors = [self.to_monitor(port=port) for port in self.ports]

        for (port_name, mode_index) in self.matrix_indices_run_sim:

            port = self.get_port_by_name(port_name=port_name)

            port_source = self.shift_port(port=port)
            mode_source = self.to_source(port=port_source, mode_index=mode_index)

            new_mnts = list(self.simulation.monitors) + mode_monitors
            sim_copy = self.simulation.copy(update=dict(sources=[mode_source], monitors=new_mnts))
            task_name = self._task_name(port=port, mode_index=mode_index)
            sim_dict[task_name] = sim_copy
        return sim_dict

    @cached_property
    def matrix_indices_monitor(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the possible matrix indices (port, mode_index) in the Component Modeler."""
        matrix_indices = []
        for port in self.ports:
            for mode_index in range(port.mode_spec.num_modes):
                matrix_indices.append((port.name, mode_index))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_source(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""
        if self.run_only is not None:
            return self.run_only
        return self.matrix_indices_monitor

    @cached_property
    def matrix_indices_run_sim(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""

        if self.element_mappings is None or self.element_mappings == {}:
            return self.matrix_indices_source

        # all the (i, j) pairs in `S_ij` that are tagged as covered by `element_mappings`
        elements_determined_by_map = [element_out for (_, element_out, _) in self.element_mappings]

        # loop through rows of the full s matrix and record rows that still need running.
        source_indices_needed = []
        for col_index in self.matrix_indices_source:

            # loop through columns and keep track of whether each element is covered by mapping.
            matrix_elements_covered = []
            for row_index in self.matrix_indices_monitor:
                element = (row_index, col_index)
                element_covered_by_map = element in elements_determined_by_map
                matrix_elements_covered.append(element_covered_by_map)

            # if any matrix elements in row still not covered by map, a source is needed for row.
            if not all(matrix_elements_covered):
                source_indices_needed.append(col_index)

        return source_indices_needed

    @cached_property
    def port_names(self) -> Tuple[List[str], List[str]]:
        """List of port names for inputs and outputs, respectively."""

        def get_port_names(matrix_elements: Tuple[str, int]) -> List[str]:
            """Get the port names from a list of (port name, mode index)."""
            port_names = []
            for port_name, _ in matrix_elements:
                if port_name not in port_names:
                    port_names.append(port_name)
            return port_names

        port_names_in = get_port_names(self.matrix_indices_source)
        port_names_out = get_port_names(self.matrix_indices_monitor)

        return port_names_out, port_names_in

    def to_monitor(self, port: Port) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=self.freqs,
            mode_spec=port.mode_spec,
            name=port.name,
        )

    def to_source(self, port: Port, mode_index: int) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        freq0 = np.mean(self.freqs)
        fdiff = max(self.freqs) - min(self.freqs)
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

    def _shift_value_signed(self, port: Port) -> float:
        """How far (signed) to shift the source from the monitor."""

        # get the grid boundaries and sizes along port normal from the simulation
        normal_axis = port.size.index(0.0)
        grid = self.simulation.grid
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

    def shift_port(self, port: Port) -> Port:
        """Generate a new port shifted by the shift amount in normal direction."""

        shift_value = self._shift_value_signed(port=port)
        center_shifted = list(port.center)
        center_shifted[port.size.index(0.0)] += shift_value
        port_shifted = port.copy(update=dict(center=center_shifted))
        return port_shifted

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    def _normalization_factor(self, port_source: Port, sim_data: SimulationData) -> complex:
        """Compute the normalization amplitude based on the measured input mode amplitude."""

        port_monitor_data = sim_data[port_source.name]
        mode_index = sim_data.simulation.sources[0].mode_index

        normalize_amps = port_monitor_data.amps.sel(
            f=np.array(self.freqs),
            direction=port_source.direction,
            mode_index=mode_index,
        )

        return normalize_amps.values

    @cached_property
    def max_mode_index(self) -> Tuple[int, int]:
        """maximum mode indices for the smatrix dataset for the in and out ports, respectively."""

        def get_max_mode_indices(matrix_elements: Tuple[str, int]) -> int:
            """Get the maximum mode index for a list of (port name, mode index)."""
            return max(mode_index for _, mode_index in matrix_elements)

        max_mode_index_out = get_max_mode_indices(self.matrix_indices_monitor)
        max_mode_index_in = get_max_mode_indices(self.matrix_indices_source)

        return max_mode_index_out, max_mode_index_in

    def _construct_smatrix(self, batch_data: BatchData) -> ModalPortDataArray:
        """Post process `BatchData` to generate scattering matrix."""

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
            f=np.array(self.freqs),
        )
        s_matrix = ModalPortDataArray(values, coords=coords)

        # loop through source ports
        for col_index in self.matrix_indices_run_sim:

            port_name_in, mode_index_in = col_index
            port_in = self.get_port_by_name(port_name=port_name_in)

            sim_data = batch_data[self._task_name(port=port_in, mode_index=mode_index_in)]

            for row_index in self.matrix_indices_monitor:

                port_name_out, mode_index_out = row_index
                port_out = self.get_port_by_name(port_name=port_name_out)

                # directly compute the element
                mode_amps_data = sim_data[port_out.name].copy().amps
                dir_out = "-" if port_out.direction == "+" else "+"
                amp = mode_amps_data.sel(
                    f=coords["f"], direction=dir_out, mode_index=mode_index_out
                )
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


class TerminalComponentModeler(AbstractComponentModeler):
    """Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped ports."""

    ports: Tuple[LumpedPort, ...] = pd.Field(
        (),
        title="Lumped ports",
        description="Collection of lumped ports associated with the network. "
        "For each port, one simulation will be run with a lumped port source.",
    )

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = self.to_source(port=port_source)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    def to_source(self, port: LumpedPort) -> UniformCurrentSource:
        """Create a current source from a given lumped port."""
        # The source is expanded slightly to make sure that, when discretized, it covers the
        # entire source region. Discretized source amps are manually zeroed out later if they
        # fall on Yee grid locations outside the analytical source region.
        freq0 = np.mean(self.freqs)
        fdiff = max(self.freqs) - min(self.freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        component = "xyz"[port.voltage_axis]
        return UniformCurrentSource(
            center=port.center,
            size=port.size,
            source_time=GaussianPulse(
                freq0=freq0, fwidth=fwidth, remove_dc_component=self.remove_dc_component
            ),
            polarization=f"E{component}",
            name=port.name,
            interpolate=True,
            confine_to_bounds=True,
        )

    def to_monitor(self, port: LumpedPort) -> FieldMonitor:
        """Field monitor to compute port voltage and current."""

        e_component = "xyz"[port.voltage_axis]
        h_component = "xyz"[port.current_axis]

        mon_size = list(port.size)
        mon_size[port.injection_axis] = 1e-10  # * port.size[port.current_axis]

        return FieldMonitor(
            center=port.center,
            size=mon_size,
            freqs=self.freqs,
            fields=[f"E{e_component}", f"H{h_component}"],
            name=f"{port.name}_E{e_component}_H{h_component}",
            colocate=False,
        )

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the port parameter calculation."""

        sim_dict = {}

        port_monitors = [self.to_monitor(port=port) for port in self.ports]

        # Create a mesh override for each port in case refinement is needed.
        # The port is a flat surface, but when computing the port current,
        # we'll eventually integrate the magnetic field just above and below
        # this surface, so the mesh override needs to ensure that the mesh
        # is fine enough not only in plane, but also in the normal direction.
        # So in the normal direction, we'll make sure there are at least
        # 2 cell layers above and below whose size is the same as the in-plane
        # cell size in the override region. Also, to ensure that the port itself
        # is aligned with a grid boundary in the normal direction, two separate
        # override regions are defined, one above and one below the analytical
        # port region.
        mesh_overrides = []

        for port in self.ports:
            if port.refine_mesh:
                override_dl = [port.size[port.voltage_axis] / port.num_grid_cells] * 3
                override_dl[port.injection_axis] /= 2

                override_size = list(port.size)
                override_size[override_size.index(0)] = 2 * override_dl[port.injection_axis]

                override_center_above = list(port.center)
                override_center_above[port.injection_axis] += override_dl[port.injection_axis]

                override_center_below = list(port.center)
                override_center_below[port.injection_axis] -= override_dl[port.injection_axis]

                mesh_overrides.append(
                    MeshOverrideStructure(
                        geometry=Box(center=override_center_below, size=override_size),
                        dl=override_dl,
                    )
                )
                mesh_overrides.append(
                    MeshOverrideStructure(
                        geometry=Box(center=override_center_above, size=override_size),
                        dl=override_dl,
                    )
                )

        new_mnts = list(self.simulation.monitors) + port_monitors

        # also, use the highest frequency in the simulation to define the grid, rather than the
        # source's central frequency, to ensure an accurate solution over the entire range
        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
                "override_structures": list(self.simulation.grid_spec.override_structures)
                + mesh_overrides,
            }
        )

        for port in self.ports:

            port_source = self.to_source(port=port)
            update_dict = dict(
                sources=[port_source],
                monitors=new_mnts,
                grid_spec=grid_spec,
            )

            sim_copy = self.simulation.copy(update=update_dict)
            task_name = self._task_name(port=port)
            sim_dict[task_name] = sim_copy
        return sim_dict

    def _construct_smatrix(self, batch_data: BatchData) -> LumpedPortDataArray:
        """Post process `BatchData` to generate scattering matrix."""

        port_names = [port.name for port in self.ports]

        values = np.zeros(
            (len(port_names), len(port_names), len(self.freqs)),
            dtype=complex,
        )
        coords = dict(
            port_out=port_names,
            port_in=port_names,
            f=np.array(self.freqs),
        )
        a_matrix = LumpedPortDataArray(values, coords=coords)
        b_matrix = a_matrix.copy(deep=True)

        def port_iv(port: LumpedPort, sim_data: SimulationData):
            """Helper to compute the port current and voltage."""
            e_component = "xyz"[port.voltage_axis]
            h_component = "xyz"[port.current_axis]
            orth_component = "xyz"[port.injection_axis]

            field_data = sim_data[f"{port.name}_E{e_component}_H{h_component}"]

            e_field = field_data.field_components[f"E{e_component}"]
            e_coords = [e_field.x, e_field.y, e_field.z]

            h_field = field_data.field_components[f"H{h_component}"]
            h_coords = [h_field.x, h_field.y, h_field.z]

            # the tangential E field should be sampled at the source plane, and
            # the tangential H field should be sampled just above and below the source plane
            # and then subtracted, to compute the surface current density

            orth_index = np.argmin(
                np.abs(e_coords[port.injection_axis].values - port.center[port.injection_axis])
            )
            assert orth_index > 0

            # for the line integrals, the E and H fields should be sampled near the port's center
            e_index = np.argmin(
                np.abs(h_coords[port.voltage_axis].values - port.center[port.voltage_axis])
            )
            h_index = np.argmin(
                np.abs(e_coords[port.current_axis].values - port.center[port.current_axis])
            )

            e_field = e_field.isel(
                indexers={
                    orth_component: orth_index,
                    h_component: h_index,
                }
            )

            h1_field = h_field.isel(
                indexers={
                    orth_component: orth_index - 1,
                    e_component: e_index,
                }
            )
            h2_field = h_field.isel(
                indexers={
                    orth_component: orth_index,
                    e_component: e_index,
                }
            )
            h_field = h1_field - h2_field

            # interpolate E and H locations to coincide with port bounds along the integration path
            e_coords_interp = {
                e_component: np.linspace(
                    port.bounds[0][port.voltage_axis],
                    port.bounds[1][port.voltage_axis],
                    len(e_coords[port.voltage_axis]),
                )
            }
            h_coords_interp = {
                h_component: np.linspace(
                    port.bounds[0][port.current_axis],
                    port.bounds[1][port.current_axis],
                    len(h_coords[port.current_axis]),
                )
            }

            e_field = e_field.interp(**e_coords_interp)
            h_field = h_field.interp(**h_coords_interp)

            voltage = -e_field.integrate(coord=e_component).squeeze()
            current = h_field.integrate(coord=h_component).squeeze()

            if port.current_axis != (port.voltage_axis + 1) % 3:
                current *= -1

            # print("=== Port data ===")
            # print("--- orth_index ---")
            # print(orth_index)
            # print("--- e_field ---")
            # print(e_field)
            # print("--- h_field ---")
            # print(h_field)
            # print("======")

            return voltage, current

        def port_ab(port: LumpedPort, sim_data: SimulationData):
            """Helper to compute the port incident and reflected power waves."""
            voltage, current = port_iv(port, sim_data)
            power_a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
            power_b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
            return power_a, power_b

        # loop through source ports
        for port_in in self.ports:

            sim_data = batch_data[self._task_name(port=port_in)]

            for port_out in self.ports:

                a_out, b_out = port_ab(port_out, sim_data)

                a_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = a_out

                b_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = b_out

        s_matrix = self.ab_to_s(a_matrix, b_matrix)

        return s_matrix
