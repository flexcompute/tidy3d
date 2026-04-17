"""Data structures for post-processing terminal component simulations to calculate S-matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from pydantic import Field, TypeAdapter

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.constants import C_0
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.base import AbstractComponentModelerData
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.types import LumpedPortType
from tidy3d.plugins.smatrix.types import SParamDef
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
    compute_power_delivered_by_port,
    compute_power_wave_amplitudes,
    s_to_z,
)

if TYPE_CHECKING:
    from tidy3d.components.data.monitor_data import MonitorData
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
    from tidy3d.plugins.smatrix.data.data_array import PortNameDataArray
    from tidy3d.plugins.smatrix.types import NetworkIndex

# Accepted types for the renormalized_reference_impedance field.
# - complex: uniform impedance applied to all ports (diagonal matrix)
# - PortDataArray: per-port impedance with dims (f, port) → diagonal matrix
# - TerminalPortDataArray: full impedance matrix with dims (f, port_out, port_in)
RenormalizedReferenceImpedance = Union[complex, TerminalPortDataArray, PortDataArray]
_ref_impedance_adapter = TypeAdapter(RenormalizedReferenceImpedance)


class MicrowaveSMatrixData(MicrowaveBaseModel):
    """Stores the computed S-matrix and reference impedances for the terminal ports."""

    port_reference_impedances: Optional[TerminalPortDataArray] = Field(
        None,
        title="Port Reference Impedances",
        description="Reference impedance matrix for each port used in the S-parameter calculation. "
        "Has dimensions (f, port_out, port_in) to support coupled impedances from TerminalWavePort. "
        "For WavePort and LumpedPort, the impedance matrix is diagonal.",
    )

    data: TerminalPortDataArray = Field(
        title="S-Matrix Data",
        description="An array containing the computed S-matrix of the device. The data is organized by terminal ports, representing the scattering parameters between them.",
    )

    s_param_def: SParamDef = Field(
        "pseudo",
        title="Scattering Parameter Definition",
        description="Wave definition: 'pseudo', 'power', or 'symmetric_pseudo'.",
    )


class TerminalComponentModelerData(AbstractComponentModelerData, MicrowaveBaseModel):
    """
    Data associated with a :class:`.TerminalComponentModeler` simulation run.


    Notes
    -----

    This class serves as a data container for the results of a component modeler simulation,
    with the original simulation definition, and port simulation data, and the solver log.


    **S-Parameter Definitions**

    The ``s_param_def`` parameter controls which wave definition is used to compute scattering
    parameters. Three definitions are supported:

    - ``"pseudo"`` (default): Pseudo-waves as defined by Marks and Williams [1]_. Uses scaling
      factor :math:`F = \\sqrt{\\text{Re}(Z)} / (2|Z|)`. Wave amplitudes are :math:`a = F(V + ZI)`
      and :math:`b = F(V - ZI)`.

    - ``"power"``: Power waves as defined by Kurokawa [3]_ and described in Pozar [2]_. Uses
      scaling factor :math:`F = 1 / (2\\sqrt{\\text{Re}(Z)})`. Wave amplitudes are
      :math:`a = F(V + ZI)` and :math:`b = F(V - Z^*I)` where :math:`Z^*` is the complex
      conjugate. Ensures :math:`|a|^2 - |b|^2` represents actual power flow.

    - ``"symmetric_pseudo"``: Equivalent to pseudo-waves except for the scaling factor. Uses
      :math:`F = 1 / (2\\sqrt{Z})` where the square root is complex. This choice of scaling
      factor ensures the S-matrix will be symmetric when the simulated device is reciprocal.


    **References**

    .. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
            J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

    .. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
            John Wiley & Sons, 2012.

    .. [3]  K. Kurokawa, "Power Waves and the Scattering Matrix," IEEE Trans.
            Microwave Theory Tech., vol. 13, no. 2, pp. 194-202, March 1965.
    """

    modeler: TerminalComponentModeler = Field(
        ...,
        title="TerminalComponentModeler",
        description="The original :class:`.TerminalComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
    )

    renormalized_reference_impedance: Optional[RenormalizedReferenceImpedance] = Field(
        None,
        title="Renormalized Reference Impedance",
        description="When set, overrides port_reference_impedances for all S-matrix computations. "
        "Accepts: complex (uniform), PortDataArray (per-port diagonal), "
        "or TerminalPortDataArray (full matrix).",
    )

    def renormalize(
        self, reference_impedance: RenormalizedReferenceImpedance
    ) -> TerminalComponentModelerData:
        """Return a new instance whose S-matrix is renormalized to a different reference impedance.

        The returned object recomputes wave amplitudes from the cached voltage/current
        data using the new reference impedance, so all downstream methods
        (:meth:`smatrix`, :meth:`s_to_z`, :meth:`compute_port_wave_amplitude_matrices`, etc.)
        automatically reflect the new impedance.

        Parameters
        ----------
        reference_impedance : RenormalizedReferenceImpedance
            The new reference impedance. Accepts:

            - ``complex`` — uniform impedance applied to all ports (e.g. ``50``)
            - :class:`.PortDataArray` — per-port impedance with dims ``(f, port)``
            - :class:`.TerminalPortDataArray` — full impedance matrix ``(f, port_out, port_in)``

        Returns
        -------
        :class:`.TerminalComponentModelerData`
            A copy of this object with the new reference impedance applied.

        Examples
        --------
        >>> s_50 = modeler_data.renormalize(50).smatrix()  # doctest: +SKIP
        """
        _ref_impedance_adapter.validate_python(reference_impedance)
        return self.updated_copy(
            renormalized_reference_impedance=reference_impedance,
            deep=False,
            validate=False,
        )

    def _build_reference_impedance_matrix(
        self, value: RenormalizedReferenceImpedance
    ) -> TerminalPortDataArray:
        """Convert any accepted impedance input to a full ``TerminalPortDataArray``.

        Parameters
        ----------
        value : RenormalizedReferenceImpedance
            The reference impedance value to convert.

        Returns
        -------
        :class:`.TerminalPortDataArray`
            Impedance matrix with dimensions ``(f, port_out, port_in)``.
        """
        network_indices = list(self.modeler.matrix_indices_monitor)
        num_ports = len(network_indices)
        freqs = np.array(self.modeler.freqs)
        num_freqs = len(freqs)

        coords = {
            "f": freqs,
            "port_out": network_indices,
            "port_in": network_indices,
        }

        if isinstance(value, TerminalPortDataArray):
            return value

        if isinstance(value, PortDataArray):
            # Per-port diagonal: PortDataArray has dims (f, port)
            values = np.zeros((num_freqs, num_ports, num_ports), dtype=complex)
            for i, port_name in enumerate(network_indices):
                values[:, i, i] = value.sel(port=port_name).values
            return TerminalPortDataArray(values, coords=coords)

        if isinstance(value, (int, float, complex)):
            # Uniform scalar → diagonal matrix
            values = np.zeros((num_freqs, num_ports, num_ports), dtype=complex)
            for i in range(num_ports):
                values[:, i, i] = complex(value)
            return TerminalPortDataArray(values, coords=coords)

        raise TypeError(
            f"Unsupported type for renormalized_reference_impedance: {type(value).__name__}. "
            "Expected complex, PortDataArray, TerminalPortDataArray, or 'Z0'."
        )

    def smatrix(
        self,
        assume_ideal_excitation: Optional[bool] = None,
        s_param_def: Optional[SParamDef] = None,
    ) -> MicrowaveSMatrixData:
        """Computes and returns the S-matrix and port reference impedances.

        Parameters
        ----------
        assume_ideal_excitation: If ``True``, assumes that exciting one port
            does not produce incident waves at other ports. This simplifies the
            S-matrix calculation and is required if not all ports are excited. If not
            provided, ``modeler.assume_ideal_excitation`` is used.
        s_param_def: Wave definition: "pseudo", "power", or "symmetric_pseudo".
            If not provided, ``modeler.s_param_def`` is used.
            See :class:`.TerminalComponentModeler` for details.

        Returns
        -------
        :class:`.MicrowaveSMatrixData`
            Container with the computed S-parameters and the port reference impedances.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import terminal_construct_smatrix

        terminal_port_data = terminal_construct_smatrix(
            modeler_data=self,
            assume_ideal_excitation=assume_ideal_excitation
            if (assume_ideal_excitation is not None)
            else self.modeler.assume_ideal_excitation,
            s_param_def=s_param_def if (s_param_def is not None) else self.modeler.s_param_def,
        )
        smatrix_data = MicrowaveSMatrixData(
            data=terminal_port_data,
            port_reference_impedances=self.port_reference_impedances,
            s_param_def=s_param_def if (s_param_def is not None) else self.modeler.s_param_def,
        )
        return smatrix_data

    def change_port_reference_planes(
        self, smatrix: MicrowaveSMatrixData, port_shifts: PortNameDataArray = None
    ) -> MicrowaveSMatrixData:
        """
        Performs S-parameter de-embedding by shifting reference planes ``port_shifts`` um.

        Parameters
        ----------
        smatrix : :class:`.MicrowaveSMatrixData`
            S-parameters before reference planes are shifted.
        port_shifts : :class:`.PortNameDataArray`
            Data array of shifts of wave ports' reference planes.
            The sign of a port shift reflects direction with respect to the axis normal to a ``WavePort`` plane:
            E.g.: ``PortNameDataArray(data=-a, coords={"port": "WP1"})`` defines a shift in the first ``WavePort`` by
            ``a`` um in the direction opposite to the positive axis direction (the axis normal to the port plane).

        Returns
        -------
        :class:`MicrowaveSMatrixData`
            De-embedded S-parameters with respect to updated reference frames.
        """

        # get s-parameters with respect to current `WavePort` locations
        S_matrix = smatrix.data.values
        S_new = np.zeros_like(S_matrix, dtype=complex)
        N_freq, N_ports, _ = S_matrix.shape

        # pre-allocate memory for effective propagation constants
        kvecs = np.zeros((N_freq, N_ports), dtype=complex)
        shifts_vec = np.zeros(N_ports)
        directions_vec = np.ones(N_ports)

        port_idxs = []
        n_complex_new = []

        # extract raw data
        key = self.data.keys_tuple[0]
        data = self.data[key].data
        ports = self.modeler.ports

        # get port names and names of ports to be shifted
        port_names = [port.name for port in ports]
        shift_names = port_shifts.coords["port"].values

        # Build a mapping for quick lookup from monitor name to monitor data
        mode_map = {mode_data.monitor.name: mode_data for mode_data in data}

        # form a numpy vector of port shifts
        for shift_name in shift_names:
            # ensure that port shifts were defined for valid ports
            if shift_name not in port_names:
                raise ValueError(
                    "The specified port could not be found in the simulation! "
                    f"Please, make sure the port name is from the following list {port_names}"
                )

            # get the port by the name
            port = self.modeler.get_port_by_name(shift_name)

            # if de-embedding is requested for lumped port
            if isinstance(port, LumpedPortType):
                raise ValueError(
                    "De-embedding currently supports only 'WavePort' instances. "
                    f"Received type: '{type(port).__name__}'."
                )
                # alternatively we can send a warning and set `shifts_vector[index]` to 0.
                # shifts_vector[index] = 0.0
            else:
                # Collect corresponding mode_data
                mode_data = mode_map[port._mode_monitor_name]
                for mode_index in port._mode_indices(
                    mode_spec=self.modeler._resolved_mode_specs.get(port.name)
                ):
                    network_index = self.modeler.network_index(port, mode_index)
                    idx = smatrix.data.indexes["port_in"].get_loc(network_index)
                    shifts_vec[idx] = port_shifts.sel(port=shift_name).values
                    directions_vec[idx] = -1 if port.direction == "-" else 1
                    port_idxs.append(idx)
                    n_complex = mode_data.n_complex.sel(mode_index=mode_index)
                    n_complex_new.append(np.squeeze(n_complex.data))

        # flatten port shift vector
        shifts_vec = np.ravel(shifts_vec)
        directions_vec = np.ravel(directions_vec)

        # Convert to stacked arrays
        freqs = np.array(self.modeler.freqs)
        n_complex_new = np.array(n_complex_new).T

        # construct transformation matrix P_inv
        kvecs[:, port_idxs] = 2 * np.pi * freqs[:, np.newaxis] * n_complex_new / C_0
        phase = -kvecs * shifts_vec * directions_vec
        P_inv = np.exp(1j * phase)

        # de-embed S-parameters: S_new = P_inv @ S_matrix @ P_inv
        S_new = S_matrix * P_inv[:, :, np.newaxis] * P_inv[:, np.newaxis, :]

        # create a new Port Data Array
        smat_data = TerminalPortDataArray(S_new, coords=smatrix.data.coords)

        return smatrix.updated_copy(data=smat_data)

    def smatrix_deembedded(self, port_shifts: np.ndarray = None) -> MicrowaveSMatrixData:
        """Interface function returns  de-embedded S-parameter matrix."""
        return self.change_port_reference_planes(self.smatrix(), port_shifts=port_shifts)

    def _monitor_data_at_port_amplitude(
        self,
        port_index: NetworkIndex,
        monitor_name: str,
        a_port: Union[FreqDataArray, complex],
        a_raw_port: FreqDataArray,
    ) -> MonitorData:
        """Normalize monitor data to a desired complex amplitude at a specific port.

        This method scales the monitor data so that the incident wave amplitude at the
        specified port matches the desired value, where :math:`\frac{1}{2}|a|^2` represents
        the power incident from the port into the system.

        Parameters
        ----------
        port_index : NetworkIndex
            The port at which to normalize the amplitude.
        monitor_name : str
            Name of the monitor to normalize.
        a_port : Union[:class:`.FreqDataArray`, complex]
            Desired complex amplitude at the port. If a complex number is provided,
            it is applied uniformly across all frequencies.
        a_raw_port : :class:`.FreqDataArray`
            Raw incident wave amplitude at the port from the simulation, used as
            the reference for scaling.

        Returns
        -------
        :class:`.MonitorData`
            Normalized monitor data scaled to the desired port amplitude.
        """
        task_name = self.modeler.task_name_from_index(port_index)
        sim_data_port = self.data[task_name]
        monitor_data = sim_data_port[monitor_name]
        if not isinstance(a_port, FreqDataArray):
            freqs = list(monitor_data.monitor.freqs)
            array_vals = a_port * np.ones(len(freqs))
            a_port = FreqDataArray(array_vals, coords={"f": freqs})
        scale_array = a_port / a_raw_port
        return monitor_data.scale_fields_by_freq_array(scale_array, method="nearest")

    def get_antenna_metrics_data(
        self,
        port_amplitudes: Optional[dict[NetworkIndex, complex]] = None,
        monitor_name: Optional[str] = None,
    ) -> AntennaMetricsData:
        """Calculate antenna parameters using superposition of fields from multiple port excitations.

        The method computes the radiated far fields and port excitation power wave amplitudes
        for a superposition of port excitations, which can be used to analyze antenna radiation
        characteristics.

        Note
        ----
        The ``NetworkIndex`` identifies a single excitation in the modeled device, so it represents
        a :class:`.LumpedPort` or a single mode from a :class:`.WavePort`. Use the static method
        :meth:`.TerminalComponentModeler.network_index` to convert port and optional mode index
        into the appropriate ``NetworkIndex`` for use in the ``port_amplitudes`` dictionary.

        Parameters
        ----------
        port_amplitudes : dict[NetworkIndex, complex] = None
            Dictionary mapping a network index to their desired excitation amplitudes. For each network port,
            :math:`\\frac{1}{2}|a|^2` represents the incident power from that port into the system.
            If ``None``, uses only the first port without any scaling of the raw simulation data. When
            ``None`` is passed as a port amplitude, the raw simulation data is used for that port. Note
            that in this method ``a`` represents the incident wave amplitude using the power wave definition
            in [2]_.
        monitor_name : str
            Name of the :class:`.DirectivityMonitor` to use for calculating far fields.
            If None, uses the first monitor in `radiation_monitors`.

        Returns
        -------
        :class:`.AntennaMetricsData`
            Container with antenna parameters including directivity, gain, and radiation efficiency,
            computed from the superposition of fields from all excited ports.
        """
        from tidy3d.plugins.smatrix.analysis.antenna import get_antenna_metrics_data

        antenna_metrics_data = get_antenna_metrics_data(
            terminal_component_modeler_data=self,
            port_amplitudes=port_amplitudes,
            monitor_name=monitor_name,
        )
        return antenna_metrics_data

    @cached_property
    def port_reference_impedances(self) -> TerminalPortDataArray:
        """Calculates the reference impedance matrix for each port across all frequencies.

        This function determines the characteristic impedance for every port defined
        in the modeler. It returns a matrix with dimensions (f, port_out, port_in)
        to support coupled impedances from :class:`.TerminalWavePort`. For
        :class:`.WavePort` and :class:`.LumpedPort`, the impedance matrix is diagonal.

        When :attr:`renormalized_reference_impedance` is set, returns the
        user-specified impedance instead of the simulation-derived one.

        Returns:
            A :class:`.TerminalPortDataArray` containing the complex impedance matrix
            with dimensions (f, port_out, port_in) for each frequency.
        """
        if self.renormalized_reference_impedance is not None:
            return self._build_reference_impedance_matrix(self.renormalized_reference_impedance)

        from tidy3d.plugins.smatrix.analysis.terminal import port_reference_impedances

        return port_reference_impedances(self)

    def compute_wave_amplitudes_at_each_port(
        self,
        sim_data: SimulationData,
        port_reference_impedances: Optional[TerminalPortDataArray] = None,
        s_param_def: SParamDef = "pseudo",
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Results from the simulation.
        port_reference_impedances : :class:`.TerminalPortDataArray`, optional
            Reference impedance matrix with dimensions (f, port_out, port_in).
            If not provided, it is computed from the cached
            property :meth:`.port_reference_impedances`. Defaults to ``None``.
        s_param_def : SParamDef
            Wave definition: "pseudo", "power", or "symmetric_pseudo".
            See :class:`.TerminalComponentModeler` for details.

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) wave amplitudes at each port.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import compute_wave_amplitudes_at_each_port

        port_reference_impedances_i = (
            port_reference_impedances
            if port_reference_impedances is not None
            else self.port_reference_impedances
        )

        return compute_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances_i,
            sim_data=sim_data,
            s_param_def=s_param_def,
        )

    def compute_power_wave_amplitudes_at_each_port(
        self,
        sim_data: SimulationData,
        port_reference_impedances: Optional[TerminalPortDataArray] = None,
    ) -> tuple[PortDataArray, PortDataArray]:
        """Compute the incident and reflected power wave amplitudes at each port.
        The computed amplitudes have not been normalized.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Results from the simulation.
        port_reference_impedances : :class:`.TerminalPortDataArray`, optional
            Reference impedance matrix with dimensions (f, port_out, port_in).
            If not provided, it is computed from the cached
            property :meth:`.port_reference_impedances`. Defaults to ``None``.

        Returns
        -------
        tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
            Incident (a) and reflected (b) power wave amplitudes at each port.
        """
        return self.compute_wave_amplitudes_at_each_port(
            sim_data, port_reference_impedances, s_param_def="power"
        )

    def s_to_z(
        self,
        assume_ideal_excitation: Optional[bool] = None,
        s_param_def: SParamDef = "pseudo",
    ) -> TerminalPortDataArray:
        """Converts the S-matrix to the Z-matrix using the port reference impedances.

        This method first computes the S-matrix of the device and then transforms it into the
        corresponding impedance matrix (Z-matrix), using the reference impedances from
        :meth:`port_reference_impedances`.

        Parameters
        ----------
        assume_ideal_excitation: If ``True``, assumes that exciting one port
            does not produce incident waves at other ports. This simplifies the
            S-matrix calculation and is required if not all ports are excited. If not
            provided, ``modeler.assume_ideal_excitation`` is used.
        s_param_def : SParamDef, optional
            Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
            See :class:`.TerminalComponentModeler` for details.

        Returns
        -------
        DataArray
            The computed impedance (Z) matrix, with dimensions corresponding to the ports of
            the device.

        Examples
        --------
        >>> z_matrix = component_modeler_data.s_to_z() # doctest: +SKIP
        >>> z_11 = z_matrix.sel(port_out="port_1@0", port_in="port_1@0") # doctest: +SKIP

        See Also
        --------
        smatrix : Computes the scattering matrix.
        """
        s_matrix = self.smatrix(
            assume_ideal_excitation=assume_ideal_excitation, s_param_def=s_param_def
        )
        return s_to_z(
            s_matrix=s_matrix.data,
            reference=self.port_reference_impedances,
            s_param_def=s_param_def,
        )

    @cached_property
    def port_voltage_current_matrices(self) -> tuple[TerminalPortDataArray, TerminalPortDataArray]:
        """Compute voltage and current matrices for all port combinations.

        This method returns two matrices containing the voltage and current values computed
        across all frequency points and port combinations. The matrices represent the response
        at each output port when each input port is excited individually.

        Returns
        -------
        tuple[:class:`.TerminalPortDataArray`, :class:`.TerminalPortDataArray`]
            A tuple containing the voltage matrix and current matrix. Each matrix has dimensions
            (f, port_out, port_in) representing the voltage/current response at
            each output port due to excitation at each input port.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import (
            _compute_port_voltages_currents,
        )

        ports_in = list(self.modeler.matrix_indices_run_sim)
        ports_out = list(self.modeler.matrix_indices_monitor)
        freqs = self.modeler.freqs
        values = np.zeros(
            (len(freqs), len(ports_out), len(ports_in)),
            dtype=complex,
        )
        coords = {
            "f": np.array(freqs),
            "port_out": ports_out,
            "port_in": ports_in,
        }

        port_voltage_matrix = TerminalPortDataArray(values, coords=coords)
        port_current_matrix = port_voltage_matrix.copy(deep=True)

        for source_index in self.modeler.matrix_indices_run_sim:
            task_name = self.modeler.task_name_from_index(source_index)
            sim_data = self.data[task_name]
            port_voltages, port_currents = _compute_port_voltages_currents(self.modeler, sim_data)
            indexer = {"port_in": source_index}
            port_voltage_matrix = port_voltage_matrix._with_updated_data(
                data=port_voltages.data, coords=indexer
            )
            port_current_matrix = port_current_matrix._with_updated_data(
                data=port_currents.data, coords=indexer
            )
        return port_voltage_matrix, port_current_matrix

    def compute_port_wave_amplitude_matrices(
        self,
        s_param_def: SParamDef = "pseudo",
    ) -> tuple[TerminalPortDataArray, TerminalPortDataArray]:
        """Compute wave amplitude matrices for all port combinations.

        This method computes the incident (a) and reflected (b) wave amplitude matrices
        for all frequency points and port combinations using the specified wave definition.
        The matrices represent the forward and backward traveling wave amplitudes at each
        output port when each input port is excited individually.

        Parameters
        ----------
        s_param_def : SParamDef, optional
            Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
            See :class:`.TerminalComponentModeler` for details.

        Returns
        -------
        tuple[:class:`.TerminalPortDataArray`, :class:`.TerminalPortDataArray`]
            A tuple containing the incident (a) and reflected (b) wave amplitude matrices.
            Each matrix has dimensions (f, port_out, port_in) representing
            the wave amplitudes at each output port due to excitation at each input port.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import (
            _compute_wave_amplitudes_from_VI,
        )

        port_voltage_matrix, port_current_matrix = self.port_voltage_current_matrices
        a_matrix = port_voltage_matrix.copy(deep=True)
        b_matrix = port_voltage_matrix.copy(deep=True)
        for source_index in self.modeler.matrix_indices_run_sim:
            port_a, port_b = _compute_wave_amplitudes_from_VI(
                self.port_reference_impedances,
                port_voltage_matrix.sel(port_in=source_index, drop=True),
                port_current_matrix.sel(port_in=source_index, drop=True),
                s_param_def=s_param_def,
            )
            indexer = {"port_in": source_index}
            a_matrix = a_matrix._with_updated_data(data=port_a.data, coords=indexer)
            b_matrix = b_matrix._with_updated_data(data=port_b.data, coords=indexer)
        return a_matrix, b_matrix

    @cached_property
    def port_pseudo_wave_matrices(self) -> tuple[TerminalPortDataArray, TerminalPortDataArray]:
        """Compute pseudo-wave amplitude matrices for all port combinations.

        This method returns the incident (a) and reflected (b) pseudo-wave amplitude matrices
        computed using the pseudo-wave definition from Marks and Williams [1]_. The matrices
        represent the forward and backward traveling wave amplitudes at each output port when
        each input port is excited individually.

        Returns
        -------
        tuple[:class:`.TerminalPortDataArray`, :class:`.TerminalPortDataArray`]
            A tuple containing the incident (a) and reflected (b) pseudo-wave amplitude matrices.
            Each matrix has dimensions (f, port_out, port_in) representing
            the pseudo-wave amplitudes at each output port due to excitation at each input port.
        """
        return self.compute_port_wave_amplitude_matrices(s_param_def="pseudo")

    @cached_property
    def port_power_wave_matrices(self) -> tuple[TerminalPortDataArray, TerminalPortDataArray]:
        """Compute power-wave amplitude matrices for all port combinations.

        This method returns the incident (a) and reflected (b) power-wave amplitude matrices
        computed using the power-wave definition from Pozar [2]_. The matrices represent the
        forward and backward traveling wave amplitudes at each output port when each input
        port is excited individually.

        Returns
        -------
        tuple[:class:`.TerminalPortDataArray`, :class:`.TerminalPortDataArray`]
            A tuple containing the incident (a) and reflected (b) power-wave amplitude matrices.
            Each matrix has dimensions (f, port_out, port_in) representing
            the power-wave amplitudes at each output port due to excitation at each input port.
        """
        return self.compute_port_wave_amplitude_matrices(s_param_def="power")

    @cached_property
    def port_symmetric_pseudo_wave_matrices(
        self,
    ) -> tuple[TerminalPortDataArray, TerminalPortDataArray]:
        """Compute symmetric-pseudo wave amplitude matrices for all port combinations.

        This method returns the incident (a) and reflected (b) symmetric-pseudo wave amplitude
        matrices. These are equivalent to pseudo-waves except for the scaling factor, which
        ensures the S-matrix will be symmetric when the simulated device is reciprocal.

        Returns
        -------
        tuple[:class:`.TerminalPortDataArray`, :class:`.TerminalPortDataArray`]
            A tuple containing the incident (a) and reflected (b) symmetric-pseudo wave
            amplitude matrices. Each matrix has dimensions (f, port_out, port_in) representing
            the wave amplitudes at each output port due to excitation at each input port.
        """
        return self.compute_port_wave_amplitude_matrices(s_param_def="symmetric_pseudo")

    # Mirror Utils
    # So they can be reused elsewhere without a class reimport
    ab_to_s = staticmethod(ab_to_s)
    compute_F = staticmethod(compute_F)
    check_port_impedance_sign = staticmethod(check_port_impedance_sign)
    compute_port_VI = staticmethod(compute_port_VI)
    compute_power_wave_amplitudes = staticmethod(compute_power_wave_amplitudes)
    compute_power_delivered_by_port = staticmethod(compute_power_delivered_by_port)
