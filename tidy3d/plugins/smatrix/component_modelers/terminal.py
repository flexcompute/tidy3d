"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.data_array import DataArray, FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.geometry.utils_2d import snap_coordinate_to_grid
from ....components.simulation import Simulation
from ....components.source import GaussianPulse
from ....components.types import Ax
from ....components.viz import add_ax_if_none, equal_aspect
from ....constants import C_0, OHM
from ....exceptions import Tidy3dError, ValidationError
from ....web.api.container import BatchData
from ..ports.base_lumped import AbstractLumpedPort
from ..ports.base_terminal import AbstractTerminalPort, TerminalPortDataArray
from ..ports.coaxial_lumped import CoaxialLumpedPort
from ..ports.rectangular_lumped import LumpedPort
from ..ports.wave import WavePort
from .base import FWIDTH_FRAC, AbstractComponentModeler, TerminalPortType


class PortDataArray(DataArray):
    """Array of values over dimensions of frequency and port name.

    Example
    -------
    >>> f = [2e9, 3e9, 4e9]
    >>> ports = ["port1", "port2"]
    >>> coords = dict(f=f, port=ports)
    >>> fd = PortDataArray((1+1j) * np.random.random((3, 2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "port")


class TerminalComponentModeler(AbstractComponentModeler):
    """Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped and wave ports."""

    ports: Tuple[TerminalPortType, ...] = pd.Field(
        (),
        title="Terminal Ports",
        description="Collection of lumped and wave ports associated with the network. "
        "For each port, one simulation will be run with a source that is associated with the port.",
    )

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot a :class:`.Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot permittivity of the :class:`.Simulation` with all sources added for each port."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`.Simulation` objects for the port parameter calculation."""

        sim_dict = {}

        lumped_resistors = [port.to_load() for port in self._lumped_ports]
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
        for port, lumped_resistor in zip(self._lumped_ports, lumped_resistors):
            if port.num_grid_cells:
                mesh_overrides.extend(lumped_resistor.to_mesh_overrides())

        # also, use the highest frequency in the simulation to define the grid, rather than the
        # source's central frequency, to ensure an accurate solution over the entire range
        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
                "override_structures": list(self.simulation.grid_spec.override_structures)
                + mesh_overrides,
            }
        )

        # Make an initial simulation with new grid_spec to determine where LumpedPorts are snapped
        sim_wo_source = self.simulation.updated_copy(grid_spec=grid_spec)
        snap_centers = dict()
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
            for mon in port.to_field_monitors(
                self.freqs, snap_center=snap_centers.get(port.name), grid=sim_wo_source.grid
            )
        ]

        new_mnts = list(self.simulation.monitors) + field_monitors

        new_lumped_elements = list(self.simulation.lumped_elements) + [
            port.to_load(snap_center=snap_centers[port.name]) for port in self._lumped_ports
        ]

        update_dict = dict(
            monitors=new_mnts,
            lumped_elements=new_lumped_elements,
        )

        # This is the new default simulation will all shared components added
        sim_wo_source = sim_wo_source.copy(update=update_dict)

        # Next, simulations are generated that include the source corresponding with the excitation port
        for port in self._lumped_ports:
            port_source = port.to_source(
                self._source_time, snap_center=snap_centers[port.name], grid=sim_wo_source.grid
            )
            task_name = self._task_name(port=port)
            sim_dict[task_name] = sim_wo_source.updated_copy(sources=[port_source])

        # Check final simulations for grid size at lumped ports
        for _, sim in sim_dict.items():
            TerminalComponentModeler._check_grid_size_at_ports(sim, self._lumped_ports)

        # Now, create simulations with wave port sources and mode solver monitors for computing port modes
        for wave_port in self._wave_ports:
            mode_monitor = wave_port.to_mode_solver_monitor(freqs=self.freqs)
            # Source is placed just before the field monitor of the port
            mode_src_pos = wave_port.center[wave_port.injection_axis] + self._shift_value_signed(
                wave_port
            )
            port_source = wave_port.to_source(self._source_time, snap_center=mode_src_pos)

            new_mnts_for_wave = new_mnts + [mode_monitor]
            update_dict = dict(monitors=new_mnts_for_wave, sources=[port_source])

            task_name = self._task_name(port=wave_port)
            sim_dict[task_name] = sim_wo_source.copy(update=update_dict)
        return sim_dict

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequency range of interest."""
        freq0 = np.mean(self.freqs)
        fdiff = max(self.freqs) - min(self.freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        return GaussianPulse(
            freq0=freq0, fwidth=fwidth, remove_dc_component=self.remove_dc_component
        )

    def _construct_smatrix(self) -> TerminalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix."""
        return self._internal_construct_smatrix(batch_data=self.batch_data)

    def _internal_construct_smatrix(self, batch_data: BatchData) -> TerminalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix, for internal use only."""

        port_names = [port.name for port in self.ports]

        values = np.zeros(
            (len(self.freqs), len(port_names), len(port_names)),
            dtype=complex,
        )
        coords = dict(
            f=np.array(self.freqs),
            port_out=port_names,
            port_in=port_names,
        )
        a_matrix = TerminalPortDataArray(values, coords=coords)
        b_matrix = a_matrix.copy(deep=True)
        V_matrix = a_matrix.copy(deep=True)
        I_matrix = a_matrix.copy(deep=True)

        def port_VI(port_out: AbstractTerminalPort, sim_data: SimulationData):
            """Helper to compute the port voltages and currents."""
            voltage = port_out.compute_voltage(sim_data)
            current = port_out.compute_current(sim_data)
            return voltage, current

        # Tabulate the reference impedances at each port and frequency
        port_impedances = self.port_reference_impedances(batch_data=batch_data)

        # loop through source ports
        for port_in in self.ports:
            sim_data = batch_data[self._task_name(port=port_in)]
            for port_out in self.ports:
                V_out, I_out = port_VI(port_out, sim_data)
                V_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = V_out

                I_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = I_out

        # Reshape arrays so that broadcasting can be used to make F and Z act as diagonal matrices at each frequency
        # Ensure data arrays have the correct data layout
        V_numpy = V_matrix.transpose(*TerminalPortDataArray._dims).values
        I_numpy = I_matrix.transpose(*TerminalPortDataArray._dims).values
        Z_numpy = port_impedances.transpose(*PortDataArray._dims).values.reshape(
            (len(self.freqs), len(port_names), 1)
        )

        # Check to make sure sign is consistent for all impedance values
        self._check_port_impedance_sign(Z_numpy)

        # Check for negative real part of port impedance and flip the V and Z signs accordingly
        negative_real_Z = np.real(Z_numpy) < 0
        V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
        Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)

        F_numpy = TerminalComponentModeler._compute_F(Z_numpy)

        # Equation 4.67 - Pozar - Microwave Engineering 4ed
        a_matrix.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
        b_matrix.values = F_numpy * (V_numpy - np.conj(Z_numpy) * I_numpy)

        s_matrix = self.ab_to_s(a_matrix, b_matrix)

        return s_matrix

    @pd.validator("simulation")
    def _validate_3d_simulation(cls, val):
        """Error if :class:`.Simulation` is not a 3D simulation"""

        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val

    @staticmethod
    def _check_grid_size_at_ports(simulation: Simulation, ports: list[Union[AbstractLumpedPort]]):
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

    @staticmethod
    def compute_power_wave_amplitudes(
        port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
    ) -> tuple[FreqDataArray, FreqDataArray]:
        """Helper to compute the incident and reflected power wave amplitudes at a port for a given
        simulation result. The computed amplitudes have not been normalized.
        """
        voltage = port.compute_voltage(sim_data)
        current = port.compute_current(sim_data)
        # Amplitudes for the incident and reflected power waves
        a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
        b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
        return a, b

    @staticmethod
    def compute_power_delivered_by_port(
        port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
    ) -> FreqDataArray:
        """Helper to compute the total power delivered to the network by a port for a given
        simulation result. Units of power are Watts.
        """
        a, b = TerminalComponentModeler.compute_power_wave_amplitudes(sim_data=sim_data, port=port)
        # Power delivered is the incident power minus the reflected power
        return 0.5 * (np.abs(a) ** 2 - np.abs(b) ** 2)

    @staticmethod
    def ab_to_s(
        a_matrix: TerminalPortDataArray, b_matrix: TerminalPortDataArray
    ) -> TerminalPortDataArray:
        """Get the scattering matrix given the power wave matrices."""
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
        s_matrix: TerminalPortDataArray, reference: Union[complex, PortDataArray]
    ) -> DataArray:
        """Get the impedance matrix given the scattering matrix and a reference impedance."""

        # Ensure dimensions are ordered properly
        z_matrix = s_matrix.transpose(*TerminalPortDataArray._dims).copy(deep=True)
        s_vals = z_matrix.values
        eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))
        if isinstance(reference, PortDataArray):
            # From Equation 4.68 - Pozar - Microwave Engineering 4ed
            # Ensure that Zport, F, and Finv act as diagonal matrices when multiplying by left or right
            shape_left = (len(s_matrix.f), len(s_matrix.port_out), 1)
            shape_right = (len(s_matrix.f), 1, len(s_matrix.port_in))
            Zport = reference.values.reshape(shape_right)
            F = TerminalComponentModeler._compute_F(Zport).reshape(shape_right)
            Finv = (1.0 / F).reshape(shape_left)
            FinvSF = Finv * s_vals * F
            RHS = eye * np.conj(Zport) + FinvSF * Zport
            LHS = eye - FinvSF
            z_vals = np.matmul(AbstractComponentModeler.inv(LHS), RHS)
        else:
            # Simpler case when all port impedances are the same
            z_vals = (
                np.matmul(AbstractComponentModeler.inv(eye - s_vals), (eye + s_vals)) * reference
            )

        z_matrix.data = z_vals
        return z_matrix

    def port_reference_impedances(self, batch_data: BatchData) -> PortDataArray:
        """Tabulates the reference impedance of each port at each frequency."""
        port_names = [port.name for port in self.ports]

        values = np.zeros(
            (len(self.freqs), len(port_names)),
            dtype=complex,
        )
        coords = dict(f=np.array(self.freqs), port=port_names)
        port_impedances = PortDataArray(values, coords=coords)
        for port in self.ports:
            if isinstance(port, WavePort):
                # Mode solver data for each wave port is stored in its associated SimulationData
                sim_data_port = batch_data[self._task_name(port=port)]
                # WavePorts have a port impedance calculated from its associated modal field distribution
                # and is frequency dependent.
                impedances = port.compute_port_impedance(sim_data_port).values
                port_impedances.loc[dict(port=port.name)] = impedances.squeeze()
            else:
                # LumpedPorts have a constant reference impedance
                port_impedances.loc[dict(port=port.name)] = np.full(len(self.freqs), port.impedance)

        port_impedances = TerminalComponentModeler._set_port_data_array_attributes(port_impedances)
        return port_impedances

    @staticmethod
    def _compute_F(Z_numpy: np.array):
        """Helper to convert port impedance matrix to F, which is used for
        computing generalized scattering parameters."""
        return 1.0 / (2.0 * np.sqrt(np.real(Z_numpy)))

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

    def _check_port_impedance_sign(self, Z_numpy: np.ndarray):
        """Sanity check for consistent sign of real part of Z for each port across all frequencies."""
        for port_idx in range(Z_numpy.shape[1]):
            port_Z = Z_numpy[:, port_idx, 0]
            signs = np.sign(np.real(port_Z))
            if not np.all(signs == signs[0]):
                raise Tidy3dError(
                    f"Inconsistent sign of real part of Z detected for port {port_idx}. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )
