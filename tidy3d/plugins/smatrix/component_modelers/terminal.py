"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Literal, Optional, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DataArray, FreqDataArray
from tidy3d.components.data.monitor_data import MonitorData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.components.monitor import DirectivityMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Ax
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import C_0, OHM
from tidy3d.exceptions import SetupError, Tidy3dError, Tidy3dKeyError, ValidationError
from tidy3d.log import log
from tidy3d.plugins.smatrix.data.terminal import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.web.api.container import BatchData

from .base import FWIDTH_FRAC, AbstractComponentModeler, TerminalPortType

NetworkIndex = str  # the 'i' in S_ij
NetworkElement = tuple[NetworkIndex, NetworkIndex]  # the 'ij' in S_ij

# The definition of wave amplitudes used to construct scattering matrix
SParamDef = Literal["pseudo", "power"]


class TerminalComponentModeler(AbstractComponentModeler[NetworkIndex, NetworkElement]):
    """Tool for modeling two-terminal multiport devices and computing port parameters
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
        """Plot a :class:`.Simulation` with all sources and absorbers added for each port, for troubleshooting."""

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
        """Plot permittivity of the :class:`.Simulation` with all sources and absorbers added for each port."""

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
    def sim_dict(self) -> dict[str, Simulation]:
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

        return sim_dict

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

        # Add port absorbers
        new_absorbers = list(sim_wo_source.internal_absorbers)
        for wave_port in self._wave_ports:
            if wave_port.absorber:
                # absorbers are shifted together with sources
                mode_src_pos = wave_port.center[
                    wave_port.injection_axis
                ] + self._shift_value_signed(wave_port)
                port_absorber = wave_port.to_absorber(
                    snap_center=mode_src_pos, frequency=0.5 * (min(self.freqs) + max(self.freqs))
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
        task_name = self._task_name(port=port, mode_index=mode_index)
        return (task_name, self.base_sim.updated_copy(sources=[port_source]))

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequency range of interest."""
        if len(self.freqs) == 1:
            freq0 = self.freqs[0]
            return GaussianPulse(freq0=self.freqs[0], fwidth=freq0 * FWIDTH_FRAC)
        return GaussianPulse.from_frequency_range(
            fmin=min(self.freqs), fmax=max(self.freqs), remove_dc_component=self.remove_dc_component
        )

    def _construct_smatrix(self) -> TerminalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix."""
        return self._internal_construct_smatrix(
            batch_data=self.batch_data,
            assume_ideal_excitation=self.assume_ideal_excitation,
            s_param_def=self.s_param_def,
        )

    def _internal_construct_smatrix(
        self,
        batch_data: BatchData,
        assume_ideal_excitation: bool = False,
        s_param_def: SParamDef = "pseudo",
    ) -> TerminalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix, for internal use only."""

        monitor_indices = list(self.matrix_indices_monitor)
        source_indices = list(self.matrix_indices_source)
        run_source_indices = list(self.matrix_indices_run_sim)

        values = np.zeros(
            (len(self.freqs), len(monitor_indices), len(source_indices)),
            dtype=complex,
        )
        coords = {
            "f": np.array(self.freqs),
            "port_out": monitor_indices,
            "port_in": source_indices,
        }
        a_matrix = TerminalPortDataArray(values, coords=coords)
        b_matrix = a_matrix.copy(deep=True)

        # Tabulate the reference impedances at each port and frequency
        port_impedances = self._port_reference_impedances(batch_data=batch_data)

        # loop through source ports
        for source_index in run_source_indices:
            port, mode_index = self.network_dict[source_index]
            sim_data = batch_data[self._task_name(port=port, mode_index=mode_index)]
            a, b = self.compute_wave_amplitudes_at_each_port(
                port_impedances, sim_data, s_param_def=s_param_def
            )

            indexer = {"port_in": source_index}
            a_matrix = a_matrix._with_updated_data(data=a.data, coords=indexer)
            b_matrix = b_matrix._with_updated_data(data=b.data, coords=indexer)

        # If excitation is assumed ideal, a_matrix is assumed to be diagonal
        # and the explicit inverse can be avoided. When only a subset of excitations
        # have been run, we cannot find the inverse anyways so must make this assumption.
        if len(monitor_indices) == len(run_source_indices) and not assume_ideal_excitation:
            s_matrix = self.ab_to_s(a_matrix, b_matrix)
        else:
            a_diag = np.diagonal(a_matrix, axis1=1, axis2=2)
            # Scale each column by the corresponding diagonal entry
            s_matrix = b_matrix / a_diag[:, np.newaxis, :]

        # element can be determined by user-defined mapping
        for (row_in, col_in), (row_out, col_out), mult_by in self.element_mappings:
            coords_from = {
                "port_in": col_in,
                "port_out": row_in,
            }
            coords_to = {
                "port_in": col_out,
                "port_out": row_out,
            }
            data = mult_by * s_matrix.loc[coords_from].data
            s_matrix = s_matrix._with_updated_data(data=data, coords=coords_to)

        return s_matrix

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

    def compute_wave_amplitudes_at_each_port(
        self,
        port_reference_impedances: PortDataArray,
        sim_data: SimulationData,
        s_param_def: SParamDef = "pseudo",
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        port_reference_impedances : :class:`.PortDataArray`
            Reference impedance at each port.
        sim_data : :class:`.SimulationData`
            Results from the simulation.
        s_param_def : SParamDef
            The type of waves computed, either pseudo waves defined by Equation 53 and Equation 54 in [1],
            or power waves defined by Equation 4.67 in [2].

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) wave amplitudes at each port.
        """
        network_indices = list(self.matrix_indices_monitor)
        values = np.zeros(
            (len(self.freqs), len(network_indices)),
            dtype=complex,
        )
        coords = {
            "f": np.array(self.freqs),
            "port": network_indices,
        }

        V_matrix = PortDataArray(values, coords=coords)
        I_matrix = V_matrix.copy(deep=True)
        a = V_matrix.copy(deep=True)
        b = V_matrix.copy(deep=True)

        for network_index in network_indices:
            port, mode_index = self.network_dict[network_index]
            V_out, I_out = self.compute_port_VI(port, sim_data)
            indexer = {"port": network_index}
            V_matrix = V_matrix._with_updated_data(data=V_out.data, coords=indexer)
            I_matrix = I_matrix._with_updated_data(data=I_out.data, coords=indexer)

        V_numpy = V_matrix.values
        I_numpy = I_matrix.values
        Z_numpy = port_reference_impedances.values

        # Check to make sure sign is consistent for all impedance values
        self._check_port_impedance_sign(Z_numpy)

        # # Check for negative real part of port impedance and flip the V and Z signs accordingly
        negative_real_Z = np.real(Z_numpy) < 0
        V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
        Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)

        F_numpy = TerminalComponentModeler._compute_F(Z_numpy, s_param_def)

        b_Zref = Z_numpy
        if s_param_def == "power":
            b_Zref = np.conj(Z_numpy)

        # Equations 53 and 54 from [1]
        # Equation 4.67 - Pozar - Microwave Engineering 4ed
        a.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
        b.values = F_numpy * (V_numpy - b_Zref * I_numpy)

        return a, b

    def compute_power_wave_amplitudes_at_each_port(
        self, port_reference_impedances: PortDataArray, sim_data: SimulationData
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected power wave amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        port_reference_impedances : :class:`.PortDataArray`
            Reference impedance at each port.
        sim_data : :class:`.SimulationData`
            Results from the simulation.

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) power wave amplitudes at each port.
        """
        return self.compute_wave_amplitudes_at_each_port(
            port_reference_impedances, sim_data, s_param_def="power"
        )

    @staticmethod
    def compute_port_VI(
        port_out: TerminalPortType, sim_data: SimulationData
    ) -> tuple[FreqDataArray, FreqDataArray]:
        """Compute the port voltages and currents.

        Parameters
        ----------
        port_out : ``TerminalPortType``
            Port for computing voltage and current.
        sim_data : :class:`.SimulationData`
            Results from simulation containing field data.

        Returns
        -------
        tuple[FreqDataArray, FreqDataArray]
            Voltage and current values at the port as frequency arrays.
        """
        voltage = port_out.compute_voltage(sim_data)
        current = port_out.compute_current(sim_data)
        return voltage, current

    @staticmethod
    def compute_power_wave_amplitudes(
        port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
    ) -> tuple[FreqDataArray, FreqDataArray]:
        """Compute the incident and reflected power wave amplitudes at a lumped port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        port : Union[:class:`.LumpedPort`, :class:`.CoaxialLumpedPort`]
            Port for computing voltage and current.
        sim_data : :class:`.SimulationData`
            Results from the simulation.

        Returns
        -------
        tuple[FreqDataArray, FreqDataArray]
            Incident (a) and reflected (b) power wave amplitude frequency arrays.
        """
        voltage, current = TerminalComponentModeler.compute_port_VI(port, sim_data)
        # Amplitudes for the incident and reflected power waves
        a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
        b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
        return a, b

    @staticmethod
    def compute_power_delivered_by_port(
        port: Union[LumpedPort, CoaxialLumpedPort],
        sim_data: SimulationData,
    ) -> FreqDataArray:
        """Compute the power delivered to the network by a lumped port.

        Parameters
        ----------
        port : Union[:class:`.LumpedPort`, :class:`.CoaxialLumpedPort`]
            Port for computing voltage and current.
        sim_data : :class:`.SimulationData`
            Results from the simulation.

        Returns
        -------
        FreqDataArray
            Power in units of Watts as a frequency array.
        """
        a, b = TerminalComponentModeler.compute_power_wave_amplitudes(sim_data=sim_data, port=port)
        # Power delivered is the incident power minus the reflected power
        return 0.5 * (np.abs(a) ** 2 - np.abs(b) ** 2)

    @staticmethod
    def ab_to_s(
        a_matrix: TerminalPortDataArray, b_matrix: TerminalPortDataArray
    ) -> TerminalPortDataArray:
        """Get the scattering matrix given the wave amplitude matrices."""
        TerminalComponentModeler._validate_square_matrix(a_matrix, "ab_to_s")
        # Ensure dimensions are ordered properly
        a_matrix = a_matrix.transpose(*TerminalPortDataArray._dims)
        b_matrix = b_matrix.transpose(*TerminalPortDataArray._dims)

        s_matrix = a_matrix.copy(deep=True)
        a_vals = s_matrix.copy(deep=True).values
        b_vals = b_matrix.copy(deep=True).values

        s_vals = np.matmul(b_vals, AbstractComponentModeler.inv(a_vals))

        s_matrix.data = s_vals
        return s_matrix

    @staticmethod
    def s_to_z(
        s_matrix: TerminalPortDataArray,
        reference: Union[complex, PortDataArray],
        s_param_def: SParamDef = "pseudo",
    ) -> DataArray:
        """Get the impedance matrix given the scattering matrix and a reference impedance.

        Parameters
        ----------
        s_matrix : :class:`.TerminalPortDataArray`
            Scattering matrix computed using either the pseudo or power wave formulation.
        reference : Union[complex, :class:`.PortDataArray`]
            The reference impedance used at each port.
        s_param_def : SParamDef
            The type of wave amplitudes used for computing the scattering matrix, either pseudo waves
            defined by Equation 53 and Equation 54 in [1] or power waves defined by Equation 4.67 in [2].
        """
        TerminalComponentModeler._validate_square_matrix(s_matrix, "s_to_z")
        # Ensure dimensions are ordered properly
        z_matrix = s_matrix.transpose(*TerminalPortDataArray._dims).copy(deep=True)
        s_vals = z_matrix.values
        eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))[np.newaxis, :, :]
        # Ensure that Zport, F, and Finv act as diagonal matrices when multiplying by left or right
        shape_left = (len(s_matrix.f), len(s_matrix.port_out), 1)
        shape_right = (len(s_matrix.f), 1, len(s_matrix.port_in))
        # Setup the port reference impedance array (scalar)
        if isinstance(reference, PortDataArray):
            Zport = reference.values.reshape(shape_right)
            F = TerminalComponentModeler._compute_F(Zport, s_param_def).reshape(shape_right)
            Finv = (1.0 / F).reshape(shape_left)
        else:
            Zport = reference
            F = TerminalComponentModeler._compute_F(Zport, s_param_def)
            Finv = 1.0 / F
        # Use conjugate when S matrix is power-wave based
        if s_param_def == "power":
            Zport_mod = np.conj(Zport)
        else:
            Zport_mod = Zport

        # From equation 74 from [1] for pseudo waves
        # From Equation 4.68 - Pozar - Microwave Engineering 4ed for power waves
        FinvSF = Finv * s_vals * F
        RHS = eye * Zport_mod + FinvSF * Zport
        LHS = eye - FinvSF
        z_vals = np.linalg.solve(LHS, RHS)

        z_matrix.data = z_vals
        return z_matrix

    @cached_property
    def port_reference_impedances(self) -> PortDataArray:
        """The reference impedance used at each port for definining wave amplitudes.

        Note
        ----
        By default, we choose reference impedances to be the load impedance.
        For wave ports, this corresponds with the the characteristic impedance.
        """
        return self._port_reference_impedances(self.batch_data)

    def _port_reference_impedances(self, batch_data: BatchData) -> PortDataArray:
        """Tabulates the reference impedance of each port at each frequency using the
        supplied :class:`.BatchData`.
        """
        values = np.zeros(
            (len(self.freqs), len(self.matrix_indices_monitor)),
            dtype=complex,
        )
        coords = {"f": np.array(self.freqs), "port": list(self.matrix_indices_monitor)}
        port_impedances = PortDataArray(values, coords=coords)
        # Mode solver data is available in each SimulationData,
        # so just choose the first available one
        if batch_data:
            sim_data = next(iter(batch_data.values()))
        else:
            raise Tidy3dError(
                "'batch_data' must contain at least one valid simulation result."
                "If you received this error, please create an issue in the Tidy3D "
                "github repository."
            )
        for network_index in self.matrix_indices_monitor:
            port, mode_index = self.network_dict[network_index]
            indexer = {"port": network_index}
            if isinstance(port, WavePort):
                # WavePorts have a port impedance calculated from its associated modal field distribution
                # and is frequency dependent.
                data = port.compute_port_impedance(sim_data).data
                port_impedances = port_impedances._with_updated_data(data=data, coords=indexer)
            else:
                # LumpedPorts have a constant reference impedance
                data = np.full(len(self.freqs), port.impedance)
                port_impedances = port_impedances._with_updated_data(data=data, coords=indexer)

        port_impedances = TerminalComponentModeler._set_port_data_array_attributes(port_impedances)
        return port_impedances

    @staticmethod
    def _compute_F(Z_numpy: np.array, s_param_def: SParamDef = "pseudo"):
        """Helper to convert port impedance matrix to F, which is used for
        computing scattering parameters
        """
        # Defined in [2] after equation 4.67
        if s_param_def == "power":
            return 1.0 / (2.0 * np.sqrt(np.real(Z_numpy)))
        # Equation 75 from [1]
        return np.sqrt(np.real(Z_numpy)) / (2.0 * np.abs(Z_numpy))

    @cached_property
    def _lumped_ports(self) -> list[AbstractLumpedPort]:
        """A list of all lumped ports in the ``TerminalComponentModeler``"""
        return [port for port in self.ports if isinstance(port, AbstractLumpedPort)]

    @cached_property
    def _wave_ports(self) -> list[WavePort]:
        """A list of all wave ports in the ``TerminalComponentModeler``"""
        return [port for port in self.ports if isinstance(port, WavePort)]

    @staticmethod
    def _set_port_data_array_attributes(data_array: PortDataArray) -> PortDataArray:
        """Helper to set additional metadata for ``PortDataArray``."""
        data_array.name = "Z0"
        return data_array.assign_attrs(units=OHM, long_name="characteristic impedance")

    @staticmethod
    def _check_port_impedance_sign(Z_numpy: np.ndarray):
        """Sanity check for consistent sign of real part of Z for each port across all frequencies."""
        for port_idx in range(Z_numpy.shape[1]):
            port_Z = Z_numpy[:, port_idx]
            signs = np.sign(np.real(port_Z))
            if not np.all(signs == signs[0]):
                raise Tidy3dError(
                    f"Inconsistent sign of real part of Z detected for port {port_idx}. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )

    @staticmethod
    def _validate_square_matrix(matrix: TerminalPortDataArray, method_name: str) -> None:
        """Check if the matrix has equal input and output port dimensions.

        Parameters
        ----------
        matrix : TerminalPortDataArray
            Matrix to validate
        method_name : str
            Name of the calling method for error message

        Raises
        ------
        DataError
            If the matrix is not square (unequal input/output dimensions).
        """
        n_out = len(matrix.port_out)
        n_in = len(matrix.port_in)
        if n_out != n_in:
            raise Tidy3dError(
                f"Cannot compute {method_name}: number of input ports ({n_in}) "
                f"!= the number of output ports ({n_out}). This usually means the 'TerminalComponentModeler' "
                "was run with only a subset of port excitations. Please ensure that the `run_only` field in "
                "the 'TerminalComponentModeler' is not being used."
            )

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

    def _monitor_data_at_port_amplitude(
        self,
        port: TerminalPortType,
        sim_data: SimulationData,
        monitor_data: MonitorData,
        a_port: Union[FreqDataArray, complex],
    ) -> MonitorData:
        """Normalize the monitor data to a desired complex amplitude of a port,
        represented by ``a_port``, where :math:`\\frac{1}{2}|a|^2` is the power
        incident from the port into the system.
        """
        a_raw, _ = self.compute_power_wave_amplitudes_at_each_port(
            self.port_reference_impedances, sim_data
        )
        a_raw_port = a_raw.sel(port=self.network_index(port))
        if not isinstance(a_port, FreqDataArray):
            freqs = list(monitor_data.monitor.freqs)
            array_vals = a_port * np.ones(len(freqs))
            a_port = FreqDataArray(array_vals, coords={"f": freqs})
        scale_array = a_port / a_raw_port
        return monitor_data.scale_fields_by_freq_array(scale_array, method="nearest")

    def get_antenna_metrics_data(
        self,
        port_amplitudes: Optional[dict[str, complex]] = None,
        monitor_name: Optional[str] = None,
    ) -> AntennaMetricsData:
        """Calculate antenna parameters using superposition of fields from multiple port excitations.

        The method computes the radiated far fields and port excitation power wave amplitudes
        for a superposition of port excitations, which can be used to analyze antenna radiation
        characteristics.

        Parameters
        ----------
        port_amplitudes : dict[str, complex] = None
            Dictionary mapping port names to their desired excitation amplitudes, ``a``. For each port,
            :math:`\\frac{1}{2}|a|^2` represents the incident power from that port into the system.
            If ``None``, uses only the first port without any scaling of the raw simulation data.
            When ``None`` is passed as a port amplitude, the raw simulation data is used for that port.
            Note that in this method ``a`` represents the incident wave amplitude
            using the power wave definition in [2].
        monitor_name : str = None
            Name of the :class:`.DirectivityMonitor` to use for calculating far fields.
            If None, uses the first monitor in `radiation_monitors`.

        Returns
        -------
        :class:`.AntennaMetricsData`
            Container with antenna parameters including directivity, gain, and radiation efficiency,
            computed from the superposition of fields from all excited ports.
        """
        # Use the first port as default if none specified
        if port_amplitudes is None:
            port_amplitudes = {self.ports[0].name: None}

        # Check port names, and create map from port to amplitude
        port_dict = {}
        for key in port_amplitudes.keys():
            port, _ = self.network_dict[key]
            port_dict[port] = port_amplitudes[key]
        # Get the radiation monitor, use first as default
        # if none specified
        if monitor_name is None:
            rad_mon = self.radiation_monitors[0]
        else:
            rad_mon = self.get_radiation_monitor_by_name(monitor_name)

        # Create data arrays for holding the superposition of all port power wave amplitudes
        f = list(rad_mon.freqs)
        coords = {"f": f, "port": list(self.matrix_indices_monitor)}
        a_sum = PortDataArray(
            np.zeros((len(f), len(self.matrix_indices_monitor)), dtype=complex), coords=coords
        )
        b_sum = a_sum.copy()
        # Retrieve associated simulation data
        combined_directivity_data = None
        for port, amplitude in port_dict.items():
            if amplitude == 0.0:
                continue
            sim_data_port = self.batch_data[self._task_name(port=port)]
            radiation_data = sim_data_port[rad_mon.name]

            a, b = self.compute_wave_amplitudes_at_each_port(
                self.port_reference_impedances, sim_data_port, s_param_def="power"
            )
            # Select a possible subset of frequencies
            a = a.sel(f=f)
            b = b.sel(f=f)
            a_raw = a.sel(port=self.network_index(port))

            if amplitude is None:
                # No scaling performed when amplitude is None
                scaled_directivity_data = sim_data_port[rad_mon.name]
                scale_factor = 1.0
            else:
                scaled_directivity_data = self._monitor_data_at_port_amplitude(
                    port, sim_data_port, radiation_data, amplitude
                )
                scale_factor = amplitude / a_raw
            a = scale_factor * a
            b = scale_factor * b

            # Combine the possibly scaled directivity data and the power wave amplitudes
            if combined_directivity_data is None:
                combined_directivity_data = scaled_directivity_data
            else:
                combined_directivity_data = combined_directivity_data + scaled_directivity_data
            a_sum += a
            b_sum += b

        # Compute and add power measures to results
        power_incident = np.real(0.5 * a_sum * np.conj(a_sum)).sum(dim="port")
        power_reflected = np.real(0.5 * b_sum * np.conj(b_sum)).sum(dim="port")
        return AntennaMetricsData.from_directivity_data(
            combined_directivity_data, power_incident, power_reflected
        )
