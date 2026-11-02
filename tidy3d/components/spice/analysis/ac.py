from __future__ import annotations

from abc import ABC
from typing import ClassVar

from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.spice.analysis.dc import (
    IsothermalSteadyChargeDCAnalysis,
    SteadyChargeDCAnalysis,
)
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import HERTZ, VOLT


class AbstractSSACAnalysis(Tidy3dBaseModel, ABC):
    """
    Abstract base class for Small-Signal AC (SSAC) analysis parameters.

    Notes
    -----
    This class provides the common interface for SSAC analysis by adding the
    ``freqs`` field to any DC analysis class.

    Small-signal analysis is performed by linearizing the device equations
    around a DC operating point. The analysis computes the small-signal response
    at the specified frequencies.

    Examples
    --------
    >>> import tidy3d as td
    >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)
    >>> sweep_freqs = freq_range.sweep_decade(num_points_per_decade=10)
    >>> ssac_spec = td.SSACAnalysis(freqs=sweep_freqs)
    """

    freqs: ArrayFloat1D = Field(
        title="Small Signal AC Frequencies",
        description="List of frequencies for small signal AC analysis. "
        "At least one :class:`.SSACVoltageSource` must be present in the boundary conditions.",
        json_schema_extra={"units": HERTZ},
    )

    at_voltages: ArrayFloat1D | None = Field(
        default=None,
        title="SSAC Bias Voltages",
        description="DC operating point voltages at which SSAC is run. "
        "If ``None``, SSAC runs only at the last configured DC voltage. "
        "Pass an explicit array (a subset of the DC sweep) to select specific bias points. "
        "This bias-point selection is only supported by the GPU accelerated charge solver; "
        "with the CPU charge solver, leave this as ``None``.",
        json_schema_extra={"units": VOLT},
    )


class SSACAnalysis(SteadyChargeDCAnalysis, AbstractSSACAnalysis):
    """
    Configures Small-Signal AC (SSAC) analysis parameters for charge simulation.

    Notes
    -----
    With a :class:`.DistributedGeneration`, it also returns
    ``DeviceCharacteristics.ac_photocurrent_voltage``.

    Examples
    --------
    >>> import tidy3d as td
    >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)
    >>> sweep_freqs = freq_range.sweep_decade(num_points_per_decade=10)
    >>> ssac_spec = td.SSACAnalysis(freqs=sweep_freqs)
    """

    # The DC default would send a scalar 'temperature' to a class with no AC fields.
    _isothermal_equivalent: ClassVar[str] = "IsothermalSSACAnalysis"


class IsothermalSSACAnalysis(IsothermalSteadyChargeDCAnalysis, AbstractSSACAnalysis):
    """
    Configures Isothermal Small-Signal AC (SSAC) analysis parameters for charge simulation.

    Notes
    -----
    This analysis class provides an interface for SSAC analysis.

    With a :class:`.DistributedGeneration`, it also returns
    ``DeviceCharacteristics.ac_photocurrent_voltage``.

    Examples
    --------
    >>> import tidy3d as td
    >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)
    >>> sweep_freqs = freq_range.sweep_decade(num_points_per_decade=10)
    >>> ssac_spec = td.IsothermalSSACAnalysis(freqs=sweep_freqs)
    """
