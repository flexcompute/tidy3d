"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.data_array import (
    DataArray,
    FreqDataArray,
)
from ....components.data.monitor_data import (
    MonitorData,
)
from ....components.data.sim_data import SimulationData
from ....components.geometry.utils_2d import snap_coordinate_to_grid
from ....components.microwave.data.monitor_data import AntennaMetricsData
from ....components.monitor import DirectivityMonitor
from ....components.simulation import Simulation
from ....components.source.time import GaussianPulse
from ....components.types import Ax
from ....components.viz import add_ax_if_none, equal_aspect
from ....constants import C_0, OHM
from ....exceptions import Tidy3dError, Tidy3dKeyError, ValidationError
from ....log import log
from ....web.api.container import BatchData
from ..data.terminal import PortDataArray, TerminalPortDataArray
from ..ports.base_lumped import AbstractLumpedPort
from ..ports.coaxial_lumped import CoaxialLumpedPort
from ..ports.rectangular_lumped import LumpedPort
from ..ports.wave import WavePort
from .base import AbstractComponentModeler, TerminalPortType


class TerminalComponentModeler(AbstractComponentModeler):
    """Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped and wave ports."""

    ports: Tuple[TerminalPortType, ...] = pd.Field(
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

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

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
            # Source is placed just before the field monitor of the port
            mode_src_pos = wave_port.center[wave_port.injection_axis] + self._shift_value_signed(
                wave_port
            )
            port_source = wave_port.to_source(self._source_time, snap_center=mode_src_pos)

            update_dict = dict(sources=[port_source])

            task_name = self._task_name(port=wave_port)
            sim_dict[task_name] = sim_wo_source.copy(update=update_dict)
        return sim_dict

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequency range of interest."""
        return GaussianPulse.from_frequency_range(
            fmin=min(self.freqs), fmax=max(self.freqs), remove_dc_component=self.remove_dc_component
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

        # Tabulate the reference impedances at each port and frequency
        port_impedances = self._port_reference_impedances(batch_data=batch_data)

        # loop through source ports
        for port_in in self.ports:
            sim_data = batch_data[self._task_name(port=port_in)]
            a, b = self.compute_power_wave_amplitudes_at_each_port(port_impedances, sim_data)
            indexer = dict(f=a.f, port_in=port_in.name, port_out=a.port)
            a_matrix.loc[indexer] = a
            b_matrix.loc[indexer] = b

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
    def _check_grid_size_at_ports(simulation: Simulation, ports: list[Union[AbstractLumpedPort]]):
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

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
        port_names = [port.name for port in self.ports]
        values = np.zeros(
            (len(self.freqs), len(port_names)),
            dtype=complex,
        )
        coords = dict(
            f=np.array(self.freqs),
            port=port_names,
        )

        V_matrix = PortDataArray(values, coords=coords)
        I_matrix = V_matrix.copy(deep=True)
        a = V_matrix.copy(deep=True)
        b = V_matrix.copy(deep=True)

        for port_out in self.ports:
            V_out, I_out = self.compute_port_VI(port_out, sim_data)
            indexer = dict(port=port_out.name)
            V_matrix.loc[indexer] = V_out
            I_matrix.loc[indexer] = I_out

        V_numpy = V_matrix.values
        I_numpy = I_matrix.values
        Z_numpy = port_reference_impedances.values

        # Check to make sure sign is consistent for all impedance values
        self._check_port_impedance_sign(Z_numpy)

        # # Check for negative real part of port impedance and flip the V and Z signs accordingly
        negative_real_Z = np.real(Z_numpy) < 0
        V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
        Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)

        F_numpy = TerminalComponentModeler._compute_F(Z_numpy)

        # Equation 4.67 - Pozar - Microwave Engineering 4ed
        a.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
        b.values = F_numpy * (V_numpy - np.conj(Z_numpy) * I_numpy)

        return a, b

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
        port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
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

    @cached_property
    def port_reference_impedances(self) -> PortDataArray:
        """The reference impedance used at each port for definining power wave amplitudes."""
        return self._port_reference_impedances(self.batch_data)

    def _port_reference_impedances(self, batch_data: BatchData) -> PortDataArray:
        """Tabulates the reference impedance of each port at each frequency using the
        supplied :class:`.BatchData`.
        """
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
        a_raw_port = a_raw.sel(port=port.name)
        if not isinstance(a_port, FreqDataArray):
            freqs = list(monitor_data.monitor.freqs)
            array_vals = a_port * np.ones(len(freqs))
            a_port = FreqDataArray(array_vals, coords=dict(f=freqs))
        scale_array = a_port / a_raw_port
        return monitor_data.scale_fields_by_freq_array(scale_array, method="nearest")

    def get_antenna_metrics_data(
        self, port_amplitudes: dict[str, complex] = None, monitor_name: str = None
    ) -> AntennaMetricsData:
        """Calculate antenna parameters using superposition of fields from multiple port excitations.

        The method computes the radiated far fields and port excitation power wave amplitudes
        for a superposition of port excitations, which can be used to analyze antenna radiation
        characteristics.

        Parameters
        ----------
        port_amplitudes : dict[str, complex] = None
            Dictionary mapping port names to their desired excitation amplitudes. For each port,
            :math:`\\frac{1}{2}|a|^2` represents the incident power from that port into the system.
            If None, uses only the first port without any scaling of the raw simulation data.
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
        port_names = [port.name for port in self.ports]
        # Check port names, and create map from port to amplitude
        port_dict = {}
        for key in port_amplitudes.keys():
            port = self.get_port_by_name(port_name=key)
            port_dict[port] = port_amplitudes[key]
        # Get the radiation monitor, use first as default
        # if none specified
        if monitor_name is None:
            rad_mon = self.radiation_monitors[0]
        else:
            rad_mon = self.get_radiation_monitor_by_name(monitor_name)

        # Create data arrays for holding the superposition of all port power wave amplitudes
        f = list(rad_mon.freqs)
        coords = dict(f=f, port=port_names)
        a_sum = PortDataArray(np.zeros((len(f), len(port_names)), dtype=complex), coords=coords)
        b_sum = a_sum.copy()
        # Retrieve associated simulation data
        combined_directivity_data = None
        for port, amplitude in port_dict.items():
            sim_data_port = self.batch_data[self._task_name(port=port)]
            radiation_data = sim_data_port[rad_mon.name]

            a, b = self.compute_power_wave_amplitudes_at_each_port(
                self.port_reference_impedances, sim_data_port
            )
            # Select a possible subset of frequencies
            a = a.sel(f=f)
            b = b.sel(f=f)
            a_raw = a.sel(port=port.name)

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
