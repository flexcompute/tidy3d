"""Tool for generating an S matrix automatically from a Tidy3d simulation and terminal port definitions."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d import ClipOperation, GeometryGroup, GridSpec, PolySlab
from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.boundary import BroadbandModeABCSpec
from tidy3d.components.frequency_extrapolation import (
    AbstractLowFrequencySmoothingSpec,
    LowFrequencySmoothingSpec,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.bound_ops import bounds_union
from tidy3d.components.geometry.utils import _shift_object
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.index import SimulationMap
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.monitor import DirectivityMonitor, ModeMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Ax, Complex, Coordinate
from tidy3d.components.types.base import annotate_type
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import C_0, MICROMETER, OHM, fp_eps, inf
from tidy3d.exceptions import SetupError, Tidy3dKeyError, ValidationError
from tidy3d.log import log
from tidy3d.plugins.smatrix.component_modelers.base import (
    FWIDTH_FRAC,
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.data.data_array import PortDataArray
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.types import TerminalPortType
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.types import NetworkElement, NetworkIndex, SParamDef

AUTO_RADIATION_MONITOR_NAME = "radiation"
AUTO_RADIATION_MONITOR_BUFFER = 2
AUTO_RADIATION_MONITOR_NUM_POINTS_THETA = 100
AUTO_RADIATION_MONITOR_NUM_POINTS_PHI = 200


class DirectivityMonitorSpec(MicrowaveBaseModel):
    """Specification for automatically generating a :class:`.DirectivityMonitor`.

    Notes
    -----
        When included in the :attr:`.TerminalComponentModeler.radiation_monitors` tuple,
        a :class:`.DirectivityMonitor` will be automatically generated with the specified
        parameters. This allows users to mix manual :class:`.DirectivityMonitor` objects
        with automatically generated ones, each with customizable parameters.

        The default origin (`custom_origin`) for defining observation points in the automatically
        generated monitor is set to (0, 0, 0) in the global coordinate system.

    Example
    -------
    >>> auto_monitor = DirectivityMonitorSpec(
    ...     name="custom_auto",
    ...     buffer=3,
    ...     num_theta_points=50,
    ...     num_phi_points=100
    ... )
    """

    name: Optional[str] = pd.Field(
        None,
        title="Monitor Name",
        description=f"Optional name for the auto-generated monitor. "
        f"If not provided, defaults to '{AUTO_RADIATION_MONITOR_NAME}_' + index of the monitor in the list of radiation monitors.",
    )

    freqs: Optional[tuple[pd.NonNegativeInt, ...]] = pd.Field(
        None,
        title="Frequencies",
        description="Frequencies to obtain fields at. If not provided, uses all frequencies "
        "from the :class:`.TerminalComponentModeler`. Must be a subset of modeler frequencies if provided.",
    )

    buffer: pd.NonNegativeInt = pd.Field(
        AUTO_RADIATION_MONITOR_BUFFER,
        title="Buffer Distance",
        description="Number of grid cells to maintain between monitor and PML/domain boundaries. "
        f"Default: {AUTO_RADIATION_MONITOR_BUFFER} cells.",
    )

    num_theta_points: pd.NonNegativeInt = pd.Field(
        AUTO_RADIATION_MONITOR_NUM_POINTS_THETA,
        title="Elevation Angle Points",
        description="Number of elevation angle (theta) sample points from 0 to π. "
        f"Default: {AUTO_RADIATION_MONITOR_NUM_POINTS_THETA}.",
    )

    num_phi_points: pd.NonNegativeInt = pd.Field(
        AUTO_RADIATION_MONITOR_NUM_POINTS_PHI,
        title="Azimuthal Angle Points",
        description="Number of azimuthal angle (phi) sample points from -π to π. "
        f"Default: {AUTO_RADIATION_MONITOR_NUM_POINTS_PHI}.",
    )

    custom_origin: Optional[Coordinate] = pd.Field(
        (0, 0, 0),
        title="Local Origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "monitor's center.",
        units=MICROMETER,
    )


class ModelerLowFrequencySmoothingSpec(AbstractLowFrequencySmoothingSpec):
    """Specifies the low frequency smoothing parameters for the terminal component simulation.
    This specification affects only results at wave ports. Specifically, the mode decomposition data
    for frequencies for which the total simulation time in units of the corresponding period (T = 1/f) is less than
    the specified minimum sampling time will be overridden by extrapolation from the data in the trusted frequency range.
    The trusted frequency range is defined in terms of minimum and maximum sampling times (the total simulation time divided by the corresponding period).

    Example
    -------
    >>> low_freq_smoothing = ModelerLowFrequencySmoothingSpec(
    ...     min_sampling_time=3,
    ...     max_sampling_time=6,
    ...     order=1,
    ...     max_deviation=0.5,
    ... )
    """


class TerminalComponentModeler(AbstractComponentModeler, MicrowaveBaseModel):
    """
    Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped and wave ports.


    Notes
    -----

    **References**

    .. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
            J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

    .. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
            John Wiley & Sons, 2012.
    """

    ports: tuple[TerminalPortType, ...] = pd.Field(
        (),
        title="Terminal Ports",
        description="Collection of lumped and wave ports associated with the network. "
        "For each port, one simulation will be run with a source that is associated with the port.",
    )

    run_only: Optional[tuple[NetworkIndex, ...]] = pd.Field(
        None,
        title="Run Only",
        description="Set of matrix indices that define the simulations to run. "
        "If ``None``, simulations will be run for all indices in the scattering matrix. "
        "If a tuple is given, simulations will be run only for the given matrix indices.",
    )

    element_mappings: tuple[tuple[NetworkElement, NetworkElement, Complex], ...] = pd.Field(
        (),
        title="Element Mappings",
        description="Tuple of S matrix element mappings, each described by a tuple of "
        "(input_element, output_element, coefficient), where the coefficient is the "
        "element_mapping coefficient describing the relationship between the input and output "
        "matrix element. If all elements of a given column of the scattering matrix are defined "
        "by ``element_mappings``, the simulation corresponding to this column is skipped automatically.",
    )

    radiation_monitors: tuple[
        annotate_type(Union[DirectivityMonitor, DirectivityMonitorSpec]), ...
    ] = pd.Field(
        (),
        title="Radiation Monitors",
        description="Facilitates the calculation of figures-of-merit for antennas. "
        "These monitors will be included in every simulation and record the radiated fields. "
        "Users can specify a combination of :class:`.DirectivityMonitor` objects for manual placement and :class:`.DirectivityMonitorSpec` "
        "objects for automatic generation.",
    )

    assume_ideal_excitation: bool = pd.Field(
        False,
        title="Assume Ideal Excitation",
        description="If ``True``, only the excited port is assumed to have a nonzero incident wave "
        "amplitude power. This choice simplifies the calculation of the scattering matrix. "
        "If ``False``, every entry in the vector of incident wave amplitudes (a) is calculated "
        "explicitly. This choice requires a matrix inversion when calculating the scattering "
        "matrix, but may lead to more accurate scattering parameters when there are "
        "reflections from simulation boundaries. ",
    )

    s_param_def: SParamDef = pd.Field(
        "pseudo",
        title="Scattering Parameter Definition",
        description="Whether to compute scattering parameters using the 'pseudo' or 'power' wave definitions.",
    )

    low_freq_smoothing: Optional[ModelerLowFrequencySmoothingSpec] = pd.Field(
        None,
        title="Low Frequency Smoothing",
        description="The low frequency smoothing parameters for the terminal component simulation.",
    )

    @property
    def _sim_with_sources(self) -> Simulation:
        """Instance of :class:`.Simulation` with all sources and absorbers added for each port, for plotting."""

        sources = [port.to_source(self._source_time) for port in self.ports]
        absorbers = [
            port.to_absorber()
            for port in self.ports
            if isinstance(port, WavePort) and port.absorber
        ]
        return self.simulation.updated_copy(
            sources=sources, internal_absorbers=absorbers, validate=False
        )

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot a :class:`.Simulation` with all sources and absorbers.

        This is a convenience method to visualize the simulation setup for
        troubleshooting. It shows all sources and absorbers for each port.

        Parameters
        ----------
        x : float, optional
            x-coordinate for the cross-section.
        y : float, optional
            y-coordinate for the cross-section.
        z : float, optional
            z-coordinate for the cross-section.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        return self._sim_with_sources.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot permittivity of the :class:`.Simulation`.

        This method shows the permittivity distribution of the simulation with
        all sources and absorbers added for each port.

        Parameters
        ----------
        x : float, optional
            x-coordinate for the cross-section.
        y : float, optional
            y-coordinate for the cross-section.
        z : float, optional
            z-coordinate for the cross-section.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot_eps`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """

        return self._sim_with_sources.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @staticmethod
    def network_index(port: TerminalPortType, mode_index: Optional[int] = None) -> NetworkIndex:
        """Converts the port, and a ``mode_index`` when the port is a :class:`.WavePort`, to a unique string specifier.

        Parameters
        ----------
        port : ``TerminalPortType``
            The port to convert to an index.
        mode_index : Optional[int]
            Selects a single mode from those supported by the ``port``, which is only used when
            the ``port`` is a :class:`.WavePort`

        Returns
        -------
        NetworkIndex
            A unique string that is used to identify the row/column of the scattering matrix.
        """
        return TerminalComponentModeler.get_task_name(port=port, mode_index=mode_index)

    @cached_property
    def network_dict(self) -> dict[NetworkIndex, tuple[TerminalPortType, int]]:
        """Dictionary associating each unique ``NetworkIndex`` to a port and mode index."""
        network_dict = {}
        for port in self.ports:
            if isinstance(port, WavePort):
                for mode_index in port._mode_indices:
                    key = self.network_index(port, mode_index)
                    network_dict[key] = (port, mode_index)
            else:
                key = self.network_index(port, None)
                network_dict[key] = (port, None)
        return network_dict

    @staticmethod
    def _construct_matrix_indices_monitor(
        ports: tuple[TerminalPortType, ...],
    ) -> tuple[NetworkIndex, ...]:
        """Construct matrix indices for monitoring from terminal ports.

        Parameters
        ----------
        ports : tuple[TerminalPortType, ...]
            Tuple of terminal port objects (LumpedPort, CoaxialLumpedPort, or WavePort).

        Returns
        -------
        tuple[NetworkIndex, ...]
            Tuple of network index strings.
        """
        matrix_indices = []
        for port in ports:
            if isinstance(port, WavePort):
                for mode_index in port._mode_indices:
                    matrix_indices.append(TerminalComponentModeler.network_index(port, mode_index))
            else:
                matrix_indices.append(TerminalComponentModeler.network_index(port))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_monitor(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the possible matrix indices."""
        return self._construct_matrix_indices_monitor(self.ports)

    @cached_property
    def matrix_indices_source(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the source matrix indices, which may be less than the total number of
        ports."""
        return super().matrix_indices_source

    @cached_property
    def matrix_indices_run_sim(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the matrix indices that will be used to run simulations."""
        return super().matrix_indices_run_sim

    @cached_property
    def sim_dict(self) -> SimulationMap:
        """Generate all the :class:`.Simulation` objects for the port parameter calculation."""

        # Check base simulation for grid size at ports
        TerminalComponentModeler._check_grid_size_at_ports(self.base_sim, self._lumped_ports)
        TerminalComponentModeler._check_grid_size_at_wave_ports(self.base_sim, self._wave_ports)

        sim_dict = {}
        # Now, create simulations with wave port sources and mode solver monitors for computing port modes
        for network_index in self.matrix_indices_run_sim:
            task_name, sim_with_src = self._add_source_to_sim(network_index)
            # update simulation
            sim_dict[task_name] = sim_with_src

        return SimulationMap(keys=tuple(sim_dict.keys()), values=tuple(sim_dict.values()))

    @cached_property
    def _base_sim_no_radiation_monitors(self) -> Simulation:
        """The intermediate base simulation with all grid refinement options, port loads (if present), and monitors added,
        which is only missing the source excitations and radiation monitors.
        """
        # internal mesh override and snapping points are automatically generated from lumped elements.
        lumped_resistors = [port.to_load() for port in self._lumped_ports]

        # Apply the highest frequency in the simulation to define the grid, rather than the
        # source's central frequency, to ensure an accurate solution over the entire range
        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
            }
        )

        # Make an initial simulation with new grid_spec to determine where LumpedPorts are snapped
        sim_wo_source = self.simulation.updated_copy(
            grid_spec=grid_spec,
            lumped_elements=lumped_resistors,
            validate=False,
            deep=False,
        )
        snap_centers = {}
        for port in self._lumped_ports:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                sim_wo_source.grid, port_center_on_axis, port.injection_axis
            )
            snap_centers[port.name] = new_port_center

        # Create monitors and snap to the center positions
        field_monitors = [
            mon
            for port in self.ports
            for mon in port.to_monitors(
                self.freqs, snap_center=snap_centers.get(port.name), grid=sim_wo_source.grid
            )
        ]

        new_mnts = list(self.simulation.monitors) + field_monitors

        new_lumped_elements = list(self.simulation.lumped_elements) + [
            port.to_load(snap_center=snap_centers[port.name]) for port in self._lumped_ports
        ]

        # Add mesh overrides for any wave ports present
        mesh_overrides = list(sim_wo_source.grid_spec.override_structures)
        for wave_port in self._wave_ports:
            if wave_port.num_grid_cells is not None:
                mesh_overrides.extend(wave_port.to_mesh_overrides())
        new_grid_spec = sim_wo_source.grid_spec.updated_copy(override_structures=mesh_overrides)

        new_absorbers = list(sim_wo_source.internal_absorbers)
        for wave_port in self._wave_ports:
            if wave_port.absorber:
                # absorbers are shifted together with sources
                mode_src_pos = wave_port.center[
                    wave_port.injection_axis
                ] + self._shift_value_signed(wave_port)
                port_absorber = wave_port.to_absorber(
                    snap_center=mode_src_pos,
                    freq_spec=BroadbandModeABCSpec(
                        frequency_range=(np.min(self.freqs), np.max(self.freqs))
                    ),
                )
                new_absorbers.append(port_absorber)

        update_dict = {
            "monitors": new_mnts,
            "lumped_elements": new_lumped_elements,
            "grid_spec": new_grid_spec,
            "internal_absorbers": new_absorbers,
        }

        # propagate the low frequency smoothing specification to the simulation
        mode_monitors = [mnt.name for mnt in field_monitors if isinstance(mnt, ModeMonitor)]
        if mode_monitors and self.low_freq_smoothing is not None:
            update_dict["low_freq_smoothing"] = LowFrequencySmoothingSpec(
                monitors=mode_monitors,
                min_sampling_time=self.low_freq_smoothing.min_sampling_time,
                max_sampling_time=self.low_freq_smoothing.max_sampling_time,
                order=self.low_freq_smoothing.order,
                max_deviation=self.low_freq_smoothing.max_deviation,
            )

        # update base simulation with updated set of shared components
        sim_wo_source = sim_wo_source.updated_copy(
            **update_dict,
            validate=False,
            deep=False,
        )

        # extrude port structures
        sim_wo_source = self._extrude_port_structures(sim=sim_wo_source)

        return sim_wo_source

    @cached_property
    def _finalized_radiation_monitors(self) -> tuple[DirectivityMonitor, ...]:
        """
        The tuple of DirectivityMonitor objects for the radiation monitors.

        Expands any DirectivityMonitorSpec instances to actual DirectivityMonitor objects.
        DirectivityMonitor objects are kept as-is.
        """
        base_sim = self._base_sim_no_radiation_monitors
        finalized = []

        for index, rad_mon in enumerate(self.radiation_monitors):
            if isinstance(rad_mon, DirectivityMonitorSpec):
                # Generate DirectivityMonitor from DirectivityMonitorSpec spec
                if not rad_mon.name:
                    mon_name = f"{AUTO_RADIATION_MONITOR_NAME}_{index}"
                    rad_mon = rad_mon.updated_copy(name=mon_name)

                try:
                    generated = self._generate_radiation_monitor(
                        simulation=base_sim, auto_spec=rad_mon
                    )
                    finalized.append(generated)
                except ValueError as e:
                    raise ValueError(
                        "Automatic construction of radiation monitors failed. "
                        "Please address the reason or provide a tuple of DirectivityMonitor "
                        "objects to the 'radiation_monitors' parameter."
                    ) from e
            else:
                # DirectivityMonitor - use as-is
                finalized.append(rad_mon)

        return tuple(finalized)

    @cached_property
    def base_sim(self) -> Simulation:
        """The base simulation with all components added, including radiation monitors."""
        base_sim_tmp = self._base_sim_no_radiation_monitors
        mnts_with_radiation = list(base_sim_tmp.monitors) + list(self._finalized_radiation_monitors)
        grid_spec = GridSpec.from_grid(base_sim_tmp.grid)
        grid_spec.attrs["from_grid_spec"] = base_sim_tmp.grid_spec
        # We skipped validations up to now, here we finally validate the base sim
        return base_sim_tmp.updated_copy(monitors=mnts_with_radiation, grid_spec=grid_spec)

    def _generate_radiation_monitor(
        self, simulation: Simulation, auto_spec: DirectivityMonitorSpec
    ) -> DirectivityMonitor:
        """
        Generates a DirectivityMonitor object for the simulation.

        The monitor is placed at a specified buffer distance from PML boundaries
        (or domain boundaries if no PML). It samples the whole sphere with specified angular resolution.

        The monitor is validated to ensure it is far enough from simulation structures.

        Parameters
        ----------
        simulation : Simulation
            The simulation for which to generate the monitor.
        auto_spec : DirectivityMonitorSpec
            Specification for auto-generation.

        Returns
        -------
        DirectivityMonitor
            The generated monitor configured to measure radiation in all directions.

        Raises
        ------
        ValueError
            If the monitor is not far enough from structures.
        """

        # Extract parameters from auto_spec
        monitor_name = auto_spec.name
        monitor_buffer = auto_spec.buffer
        num_theta = auto_spec.num_theta_points
        num_phi = auto_spec.num_phi_points
        monitor_freqs = auto_spec.freqs or self.freqs

        # Get PML thicknesses in all directions
        pml_layers = simulation.num_pml_layers  # List of (minus, plus) layers for each axis
        grid = simulation.grid
        boundaries = grid.boundaries.to_list  # List of coordinate arrays for each axis
        num_cells = grid.num_cells  # List of number of cells for each axis

        # Calculate monitor span using the specified buffer distance
        mnt_span = [
            (minus_pml + monitor_buffer, num_cells_axis - plus_pml - monitor_buffer)
            for (minus_pml, plus_pml), num_cells_axis in zip(pml_layers, num_cells)
        ]

        # Calculate monitor bounds
        mnt_bounds = [
            (coords[start_idx], coords[end_idx]) if sim_size_axis > 0 else (-inf, inf)
            for (start_idx, end_idx), coords, sim_size_axis in zip(
                mnt_span, boundaries, simulation.size
            )
        ]

        mnt_bounds = np.transpose(mnt_bounds)

        mnt_box = Box.from_bounds(mnt_bounds[0], mnt_bounds[1])

        # Create angle arrays for full sphere sampling
        # theta: elevation angle [0, pi]
        # phi: azimuthal angle [-pi, pi]
        theta = np.linspace(0, np.pi, num_theta)
        phi = np.linspace(-np.pi, np.pi, num_phi)

        # Create the monitor
        monitor = DirectivityMonitor(
            center=mnt_box.center,
            size=mnt_box.size,
            freqs=monitor_freqs,
            name=monitor_name,
            theta=theta,
            phi=phi,
            custom_origin=auto_spec.custom_origin,
        )

        # Validate that monitor is far enough from structures
        self._validate_radiation_monitor_buffer(simulation, mnt_span, monitor_buffer)

        return monitor

    def _validate_radiation_monitor_buffer(
        self, simulation: Simulation, mnt_span: list[tuple[int, int]], buffer: int
    ) -> None:
        """Validate that the radiation monitor is far enough from simulation structures.

        Checks that each side of the monitor is at least AUTO_RADIATION_MONITOR_BUFFER cells
        away from the union of all structures and lumped elements, using grid cell indices.

        Parameters
        ----------
        simulation : Simulation
            The simulation containing structures and lumped elements.
        mnt_span : list[tuple[int, int]]
            The span (start, stop) indices of the monitor in each axis.
        buffer : int
            The buffer distance to use.

        Raises
        ------
        ValueError
            If the monitor is not far enough from structures.
        """
        # Get finalized simulation to include all structures
        finalized_sim = simulation._finalized

        # Get all structures (including finalized ones)
        structures = finalized_sim.structures

        # Get lumped elements
        lumped_elements = simulation.lumped_elements

        # If no structures or lumped elements, validation passes
        if not structures and not lumped_elements:
            return

        # Calculate union of bounding boxes for all structures and lumped elements
        all_geoms = []

        # Add structures
        for struct in structures:
            all_geoms.append(struct.geometry)

        # Add lumped elements (they have geometry)
        for elem in lumped_elements:
            all_geoms.append(elem.to_geometry())

        # Compute union of all bounds
        if all_geoms:
            union_bounds = all_geoms[0].bounds
            for geom in all_geoms[1:]:
                union_bounds = bounds_union(union_bounds, geom.bounds)

            # Convert union bounds to Box and get grid cell indices
            union_box = Box.from_bounds(union_bounds[0], union_bounds[1])
            grid = simulation.grid
            union_inds = grid.discretize_inds(union_box, extend=True)

            # Check each axis
            for axis in range(3):
                mnt_start, mnt_end = mnt_span[axis]
                union_start, union_end = union_inds[axis]

                axis_name = "xyz"[axis]

                # Check minus side: union should be at least BUFFER cells away from monitor start
                buffer_minus = union_start - mnt_start
                if buffer_minus < buffer:
                    raise ValueError(
                        f"Automatically generated radiation monitor is too close to structures on the negative {axis_name} side. "
                        f"Buffer: {buffer_minus} cells, required: {buffer} cells. "
                        f"Please increase simulation domain size."
                    )

                # Check plus side: union should be at least BUFFER cells away from monitor end
                buffer_plus = mnt_end - union_end
                if buffer_plus < buffer:
                    raise ValueError(
                        f"Automatically generated radiation monitor is too close to structures on the positive {axis_name} side. "
                        f"Buffer: {buffer_plus} cells, required: {buffer} cells. "
                        f"Please increase simulation domain size."
                    )

    def _add_source_to_sim(self, source_index: NetworkIndex) -> tuple[str, Simulation]:
        """Adds the source corresponding to the ``source_index`` to the base simulation."""
        port, mode_index = self.network_dict[source_index]
        if isinstance(port, WavePort):
            # Source is placed just before the field monitor of the port
            mode_src_pos = port.center[port.injection_axis] + self._shift_value_signed(port)
            port_source = port.to_source(
                self._source_time, snap_center=mode_src_pos, mode_index=mode_index
            )
        else:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                self.base_sim.grid, port_center_on_axis, port.injection_axis
            )
            port_source = port.to_source(
                self._source_time, snap_center=new_port_center, grid=self.base_sim.grid
            )
        task_name = self.get_task_name(port=port, mode_index=mode_index)

        return (
            task_name,
            self.base_sim.updated_copy(sources=[port_source], validate=False, deep=False),
        )

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequency range of interest."""
        if self.custom_source_time is not None:
            return self.custom_source_time
        if len(self.freqs) == 1:
            freq0 = self.freqs[0]
            return GaussianPulse(freq0=self.freqs[0], fwidth=freq0 * FWIDTH_FRAC)

        # Using the minimum_source_bandwidth, ensure we don't create a pulse that is too narrowband
        # when fmin and fmax are close together
        return GaussianPulse.from_frequency_range(
            fmin=np.min(self.freqs),
            fmax=np.max(self.freqs),
            remove_dc_component=self.remove_dc_component,
            minimum_source_bandwidth=FWIDTH_FRAC,
        )

    @pd.validator("simulation")
    def _validate_3d_simulation(cls, val):
        """Error if :class:`.Simulation` is not a 3D simulation"""

        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val

    @pd.validator("ports")
    @skip_if_fields_missing(["simulation"])
    def _validate_port_refinement_usage(cls, val, values):
        """Warn if port refinement options are enabled, but the supplied simulation
        does not contain a grid type that will make use of them."""

        sim: Simulation = values.get("simulation")
        # If grid spec is using AutoGrid
        # then set up is acceptable
        if sim.grid_spec.auto_grid_used:
            return val

        for port in val:
            if port._is_using_mesh_refinement:
                log.warning(
                    f"A port with name '{port.name}' has mesh refinement options enabled, but the "
                    "'Simulation' passed to the 'TerminalComponentModeler' was setup with a 'GridSpec' which "
                    "does not support mesh refinement. For accurate simulations, please setup the "
                    "'Simulation' to use an 'AutoGrid'. To suppress this warning, please explicitly disable "
                    "mesh refinement options in the port, which are by default enabled. For example, set "
                    "the 'enable_snapping_points=False' and 'num_grid_cells=None' for lumped ports."
                )

        return val

    @pd.validator("radiation_monitors")
    @skip_if_fields_missing(["freqs"])
    def _validate_radiation_monitors(cls, val, values):
        """Validate radiation monitors configuration.

        Validates that:
        - DirectivityMonitor frequencies are a subset of modeler frequencies
        - DirectivityMonitorSpec frequencies (if provided) are a subset of modeler frequencies
        """
        modeler_freqs = set(values.get("freqs", []))

        for index, rad_mon in enumerate(val):
            # Only validate freqs if explicitly provided
            # freqs are provided always in DirectivityMonitor
            # in DirectivityMonitorSpec, freqs may be not provided,
            # in this case, we use the modeler frequencies, so no validation is needed
            if rad_mon.freqs is not None:
                mon_freqs = set(rad_mon.freqs)
                is_subset = modeler_freqs.issuperset(mon_freqs)
                if not is_subset:
                    mon_name = rad_mon.name or f"{AUTO_RADIATION_MONITOR_NAME}_{index}"
                    raise ValidationError(
                        f"The frequencies in the radiation monitor '{mon_name}' "
                        f"must be equal to or a subset of the frequencies in the '{cls.__name__}'."
                    )

        return val

    @staticmethod
    def _check_grid_size_at_ports(
        simulation: Simulation, ports: list[Union[LumpedPort, CoaxialLumpedPort]]
    ) -> None:
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

    @staticmethod
    def _check_grid_size_at_wave_ports(simulation: Simulation, ports: list[WavePort]) -> None:
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        for port in ports:
            disc_grid = simulation.discretize(port)
            check_axes = port.transverse_axes
            msg_header = f"'WavePort' '{port.name}' "
            for axis in check_axes:
                sim_size = simulation.size[axis]
                dim_cells = disc_grid.num_cells[axis]
                if sim_size > 0 and dim_cells <= 2:
                    small_dim = "xyz"[axis]
                    raise SetupError(
                        msg_header + f"is too small along the "
                        f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                        "Please ensure that the port's 'num_grid_cells' is not 'None'. "
                        "You also may need to use an 'AutoGrid' or `QuasiUniformGrid` "
                        "for the simulation passed to the 'TerminalComponentModeler'."
                    )

    @cached_property
    def _lumped_ports(self) -> list[AbstractLumpedPort]:
        """A list of all lumped ports in the :class:`.TerminalComponentModeler`"""
        return [port for port in self.ports if isinstance(port, AbstractLumpedPort)]

    @cached_property
    def _wave_ports(self) -> list[WavePort]:
        """A list of all wave ports in the :class:`.TerminalComponentModeler`"""
        return [port for port in self.ports if isinstance(port, WavePort)]

    @staticmethod
    def _set_port_data_array_attributes(data_array: PortDataArray) -> PortDataArray:
        """Helper to set additional metadata for ``PortDataArray``."""
        data_array.name = "Z0"
        return data_array.assign_attrs(units=OHM, long_name="characteristic impedance")

    def get_radiation_monitor_by_name(self, monitor_name: str) -> DirectivityMonitor:
        """Find and return a :class:`.DirectivityMonitor` monitor by its name.

        Parameters
        ----------
        monitor_name : str
            Name of the monitor to find.

        Returns
        -------
        :class:`.DirectivityMonitor`
            The monitor matching the given name.

        Raises
        ------
        ``Tidy3dKeyError``
            If no monitor with the given name exists.
        """
        for monitor in self._finalized_radiation_monitors:
            if monitor.name == monitor_name:
                return monitor
        raise Tidy3dKeyError(f"No radiation monitor named '{monitor_name}'.")

    def task_name_from_index(self, source_index: NetworkIndex) -> str:
        """Compute task name for a given network index without constructing simulations."""
        port, mode_index = self.network_dict[source_index]
        return self.get_task_name(port=port, mode_index=mode_index)

    def _extrude_port_structures(self, sim: Simulation) -> Simulation:
        """
        Extrude structures intersecting a port plane when a wave port lies on a structure boundary.

        This method checks wave ports with ``extrude_structures==True`` and automatically extends the boundary structures
        to PEC plates associated with internal absorbers in the direction opposite to the mode source.
        This ensures that mode sources and internal absorbers are fully contained within the extrusion.

        Parameters
        ----------
        sim : Simulation
            Simulation object containing mode sources, internal absorbers, and monitors,
            after mesh overrides and snapping points are applied.

        Returns
        -------
        Simulation
            Updated simulation with extruded structures added to ``simulation.structures``.
        """

        # create list with extruded structures
        new_structures = []
        all_new_structures = []

        # get all mode sources from TerminalComponentModeler that correspond to ports with ``extrude_structures`` flag set to ``True``.
        for port in self.ports:
            if isinstance(port, WavePort) and port.extrude_structures:
                # compute snap_center and shift the internal absorber associated with the current port
                snap_center = port.center[port.injection_axis] + self._shift_value_signed(port)
                absorber = port.to_absorber(snap_center=snap_center)
                shifted_absorber = _shift_object(
                    obj=absorber,
                    grid=sim.grid,
                    bounds=sim.bounds,
                    direction=absorber.direction,
                    shift=absorber.grid_shift,
                )

                # get the PEC box with its face surfaces
                (box, inj_axis, direction) = sim._pec_frame_box(shifted_absorber, expand=True)
                surfaces = box.surfaces(box.size, box.center)

                # get extrusion coordinates and a cutting plane for inference of intersecting structures.
                sign = 1 if direction == "+" else -1
                back_pec_plane = surfaces[2 * inj_axis + (1 if direction == "+" else 0)]

                # get extrusion extent along injection axis
                extrude_to = back_pec_plane.center[inj_axis]

                # move cutting plane beyond the waveport plane along the `ModeSource` injection direction.
                center = list(back_pec_plane.center)
                center[inj_axis] = port.center[inj_axis] - sign * fp_eps * box.size[inj_axis]
                cutting_plane = back_pec_plane.updated_copy(center=center)

                # define extrusion bounds
                extrusion_bounds = [cutting_plane.center[inj_axis], extrude_to][::sign]

                # loop over structures and extrude those that intersect a waveport plane
                for structure in sim.structures:
                    # get geometries that intersect the plane on which the waveport is defined

                    shapely_geom = cutting_plane.intersections_with(structure.geometry)

                    polygon_list = []
                    for geom in shapely_geom:
                        polygon_list = polygon_list + ClipOperation.to_polygon_list(geom)

                    new_geoms = []
                    # loop over identified geometries and extrude them
                    for polygon in polygon_list:
                        # construct outer shell of an extruded geometry first
                        exterior_vertices = np.array(polygon.exterior.coords)
                        outer_shell = PolySlab(
                            axis=inj_axis, slab_bounds=extrusion_bounds, vertices=exterior_vertices
                        )

                        # construct innner shells that represent holes
                        hole_polyslabs = [
                            PolySlab(
                                axis=inj_axis,
                                slab_bounds=extrusion_bounds,
                                vertices=np.array(hole.coords),
                            )
                            for hole in polygon.interiors
                        ]

                        # construct final geometry by removing inner holes from outer shell
                        if hole_polyslabs:
                            holes = GeometryGroup(geometries=hole_polyslabs)
                            extruded_slab_new = ClipOperation(
                                operation="difference", geometry_a=outer_shell, geometry_b=holes
                            )
                        else:
                            extruded_slab_new = outer_shell

                        # append extruded geometry
                        new_geoms.append(extruded_slab_new)
                    if len(polygon_list) != 0:
                        # update structure and add it to the list
                        new_struct = structure.updated_copy(
                            geometry=GeometryGroup(geometries=new_geoms)
                        )
                        new_structures.append(new_struct)

                        # if current port does not intersect any structures raise error
                if not new_structures:
                    raise SetupError(
                        f"The 'WavePort' '{port.name}' does not intersect any structures."
                        f"Please ensure that it is located within or at the boundary of a structure."
                    )

                all_new_structures = all_new_structures + new_structures
                new_structures = []

        # if new structures are extruded (Lumped Port extrusion is ignored)
        if all_new_structures:
            # update structures in simulation while keeping the same grid
            sim = sim.updated_copy(
                grid_spec=GridSpec.from_grid(sim.grid),
                structures=[*sim.structures, *all_new_structures],
                validate=False,
                deep=False,
            )

        return sim


TerminalComponentModeler.update_forward_refs()
