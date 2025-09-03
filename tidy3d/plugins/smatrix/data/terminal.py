"""Data structures for post-processing terminal component simulations to calculate S-matrices."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.monitor_data import MonitorData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.log import log
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.base import AbstractComponentModelerData
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.network import SParamDef
from tidy3d.plugins.smatrix.ports.types import TerminalPortType
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
    compute_power_delivered_by_port,
    compute_power_wave_amplitudes,
    s_to_z,
)


class MicrowaveSMatrixData(Tidy3dBaseModel):
    """Stores the computed S-matrix and reference impedances for the terminal ports."""

    port_reference_impedances: Optional[PortDataArray] = pd.Field(
        None,
        title="Port Reference Impedances",
        description="Reference impedance for each port used in the S-parameter calculation. This is optional and may not be present if not specified or computed.",
    )

    data: TerminalPortDataArray = pd.Field(
        ...,
        title="S-Matrix Data",
        description="An array containing the computed S-matrix of the device. The data is organized by terminal ports, representing the scattering parameters between them.",
    )

    s_param_def: SParamDef = pd.Field(
        "pseudo",
        title="Scattering Parameter Definition",
        description="Whether scattering parameters are defined using the 'pseudo' or 'power' wave definitions.",
    )


class TerminalComponentModelerData(AbstractComponentModelerData):
    """
    Data associated with a :class:`TerminalComponentModeler` simulation run.

    This class serves as a data container for the results of a component modeler simulation,
    with the original simulation definition, and port simulation data, and the solver log.

    Notes
    -----

    **References**

    .. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
            J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

    .. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
            John Wiley & Sons, 2012.

    See Also
    --------
    :func:`tidy3d.plugins.smatrix.utils.ab_to_s`
    :func:`tidy3d.plugins.smatrix.utils.check_port_impedance_sign`
    :func:`tidy3d.plugins.smatrix.utils.compute_F`
    :func:`tidy3d.plugins.smatrix.utils.compute_port_VI`
    :func:`tidy3d.plugins.smatrix.utils.compute_power_delivered_by_port`
    :func:`tidy3d.plugins.smatrix.utils.compute_power_wave_amplitudes`
    :func:`tidy3d.plugins.smatrix.utils.s_to_z`
    """

    modeler: TerminalComponentModeler = pd.Field(
        ...,
        title="TerminalComponentModeler",
        description="The original :class:`TerminalComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
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
        s_param_def: The definition of S-parameters to use, determining whether
            "pseudo waves" or "power waves" are calculated. If not provided,
            ``modeler.s_param_def`` is used.

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

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

    def _monitor_data_at_port_amplitude(
        self,
        port: TerminalPortType,
        monitor_name: str,
        a_port: Union[FreqDataArray, complex],
    ) -> MonitorData:
        """Normalize the monitor data to a desired complex amplitude of a port,
        represented by ``a_port``, where :math:`\frac{1}{2}|a|^2` is the power
        incident from the port into the system.
        """
        sim_data_port = self.data[self.modeler.get_task_name(port)]
        monitor_data = sim_data_port[monitor_name]
        a_raw, _ = self.compute_power_wave_amplitudes_at_each_port(
            self.port_reference_impedances, sim_data=sim_data_port
        )
        a_raw_port = a_raw.sel(port=self.modeler.network_index(port))
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
            :math:`\frac{1}{2}|a|^2` represents the incident power from that port into the system.
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
        from tidy3d.plugins.smatrix.analysis.antenna import get_antenna_metrics_data

        antenna_metrics_data = get_antenna_metrics_data(
            terminal_component_modeler_data=self,
            port_amplitudes=port_amplitudes,
            monitor_name=monitor_name,
        )
        return antenna_metrics_data

    @cached_property
    def port_reference_impedances(self) -> PortDataArray:
        """Calculates the reference impedance for each port across all frequencies.

        This function determines the characteristic impedance for every port defined
        in the modeler. It handles two types of ports differently: for a
        :class:`.WavePort`, the impedance is frequency-dependent and computed from
        modal properties, while for other types like :class:`.LumpedPort`, the
        impedance is a user-defined constant value.

        Returns:
            A data array containing the complex impedance for each port at each
            frequency.
        """
        from tidy3d.plugins.smatrix.analysis.terminal import port_reference_impedances

        return port_reference_impedances(self)

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
        from tidy3d.plugins.smatrix.analysis.terminal import compute_wave_amplitudes_at_each_port

        return compute_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances,
            sim_data=sim_data,
            s_param_def=s_param_def,
        )

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
        from tidy3d.plugins.smatrix.analysis.terminal import (
            compute_power_wave_amplitudes_at_each_port,
        )

        return compute_power_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances,
            sim_data=sim_data,
        )

    # Mirror Utils
    # So they can be reused elsewhere without a class reimport
    ab_to_s = ab_to_s
    compute_F = compute_F
    check_port_impedance_sign = check_port_impedance_sign
    compute_port_VI = compute_port_VI
    compute_power_wave_amplitudes = compute_power_wave_amplitudes
    compute_power_delivered_by_port = compute_power_delivered_by_port
    s_to_z = s_to_z
