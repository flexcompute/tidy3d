"""Tool for generating an S matrix automatically from a Tidy3d simulation and terminal port definitions."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BroadbandModeABCSpec
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.index import SimulationMap
from tidy3d.components.monitor import DirectivityMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Ax, Complex
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import C_0, OHM
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


class TerminalComponentModeler(AbstractComponentModeler):
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

    radiation_monitors: tuple[DirectivityMonitor, ...] = pd.Field(
        (),
        title="Radiation Monitors",
        description="Facilitates the calculation of figures-of-merit for antennas. "
        "These monitor will be included in every simulation and record the radiated fields. ",
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

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

    @pd.root_validator(pre=False)
    def _warn_refactor_2_10(cls, values):
        log.warning(
            f"ℹ️ ⚠️ The {cls.__name__} class was refactored in tidy3d version 2.10. Migration documentation will be provided, and existing functionality can be accessed in a different way.",
            log_once=True,
        )
        return values

    @property
    def _sim_with_sources(self) -> Simulation:
        """Instance of :class:`.Simulation` with all sources and absorbers added for each port, for troubleshooting."""

        sources = [port.to_source(self._source_time) for port in self.ports]
        absorbers = [
            port.to_absorber()
            for port in self.ports
            if isinstance(port, WavePort) and port.absorber
        ]
        return self.simulation.updated_copy(sources=sources, internal_absorbers=absorbers)

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **kwargs,
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
        **kwargs,
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
        # Currently the mode_index is ignored, but will be supported once multimodal WavePorts are enabled.
        return f"{port.name}"

    @cached_property
    def network_dict(self) -> dict[NetworkIndex, tuple[TerminalPortType, int]]:
        """Dictionary associating each unique ``NetworkIndex`` to a port and mode index."""
        network_dict = {}
        for port in self.ports:
            mode_index = None
            if isinstance(port, WavePort):
                mode_index = port.mode_index
            key = TerminalComponentModeler.network_index(port, mode_index)
            network_dict[key] = (port, mode_index)
        return network_dict

    @cached_property
    def matrix_indices_monitor(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the possible matrix indices."""
        matrix_indices = []
        for port in self.ports:
            if isinstance(port, WavePort):
                matrix_indices.append(self.network_index(port, port.mode_index))
            else:
                matrix_indices.append(self.network_index(port))
        return tuple(matrix_indices)

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

        sim_dict = {}
        # Now, create simulations with wave port sources and mode solver monitors for computing port modes
        for network_index in self.matrix_indices_run_sim:
            task_name, sim_with_src = self._add_source_to_sim(network_index)
            sim_dict[task_name] = sim_with_src

        # Check final simulations for grid size at ports
        for _, sim in sim_dict.items():
            TerminalComponentModeler._check_grid_size_at_ports(sim, self._lumped_ports)
            TerminalComponentModeler._check_grid_size_at_wave_ports(sim, self._wave_ports)

        return SimulationMap(keys=tuple(sim_dict.keys()), values=tuple(sim_dict.values()))

    @cached_property
    def base_sim(self) -> Simulation:
        """The base simulation with all grid refinement options, port loads (if present), and monitors added,
        which is only missing the source excitations.
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
            grid_spec=grid_spec, lumped_elements=lumped_resistors
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

        if self.radiation_monitors is not None:
            new_mnts = new_mnts + list(self.radiation_monitors)

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

        # This is the new default simulation will all shared components added
        return sim_wo_source.copy(update=update_dict)

    def _add_source_to_sim(self, source_index: NetworkIndex) -> tuple[str, Simulation]:
        """Adds the source corresponding to the ``source_index`` to the base simulation."""
        port, mode_index = self.network_dict[source_index]
        if isinstance(port, WavePort):
            # Source is placed just before the field monitor of the port
            mode_src_pos = port.center[port.injection_axis] + self._shift_value_signed(port)
            port_source = port.to_source(self._source_time, snap_center=mode_src_pos)
        else:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                self.base_sim.grid, port_center_on_axis, port.injection_axis
            )
            port_source = port.to_source(
                self._source_time, snap_center=new_port_center, grid=self.base_sim.grid
            )
        task_name = self.get_task_name(port=port, mode_index=mode_index)
        return (task_name, self.base_sim.updated_copy(sources=[port_source]))

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequency range of interest."""
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

    @pd.validator("radiation_monitors")
    def _validate_radiation_monitors(cls, val, values):
        freqs = set(values.get("freqs"))
        for rad_mon in val:
            mon_freqs = rad_mon.freqs
            is_subset = freqs.issuperset(mon_freqs)
            if not is_subset:
                raise ValidationError(
                    f"The frequencies in the radiation monitor '{rad_mon.name}' "
                    f"must be equal to or a subset of the frequencies in the '{cls.__name__}'."
                )
        return val

    @staticmethod
    def _check_grid_size_at_ports(
        simulation: Simulation, ports: list[Union[LumpedPort, CoaxialLumpedPort]]
    ):
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

    @staticmethod
    def _check_grid_size_at_wave_ports(simulation: Simulation, ports: list[WavePort]):
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
        for monitor in self.radiation_monitors:
            if monitor.name == monitor_name:
                return monitor
        raise Tidy3dKeyError(f"No radiation monitor named '{monitor_name}'.")


TerminalComponentModeler.update_forward_refs()
