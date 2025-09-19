from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.spice.analysis.dc import (
    IsothermalSteadyChargeDCAnalysis,
    SteadyChargeDCAnalysis,
)
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import HERTZ
from tidy3d.constants import inf as td_inf


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

    freqs: ArrayFloat1D = pd.Field(
        ...,
        title="Small Signal AC Frequencies",
        description="List of frequencies for small signal AC analysis. "
        "At least one :class:`.SSACVoltageSource` must be present in the boundary conditions.",
        units=HERTZ,
    )

    @pd.validator("freqs")
    def validate_freqs(cls, val):
        if len(val) == 0:
            raise ValueError("'freqs' cannot be empty (size 0).")
        else:
            for freq in val:
                if freq == td_inf:
                    raise ValueError("'freqs' cannot contain infinite frequencies.")
                elif freq < 0:
                    raise ValueError("'freqs' cannot contain negative frequencies.")
        return val


class SSACAnalysis(SteadyChargeDCAnalysis, AbstractSSACAnalysis):
    """
    Configures Small-Signal AC (SSAC) analysis parameters for charge simulation.

    Examples
    --------
    >>> import tidy3d as td
    >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)
    >>> sweep_freqs = freq_range.sweep_decade(num_points_per_decade=10)
    >>> ssac_spec = td.SSACAnalysis(freqs=sweep_freqs)
    """


class IsothermalSSACAnalysis(IsothermalSteadyChargeDCAnalysis, AbstractSSACAnalysis):
    """
    Configures Isothermal Small-Signal AC (SSAC) analysis parameters for charge simulation.

    Notes
    -----
    This analysis class provides an interface for SSAC analysis.

    Examples
    --------
    >>> import tidy3d as td
    >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)
    >>> sweep_freqs = freq_range.sweep_decade(num_points_per_decade=10)
    >>> ssac_spec = td.IsothermalSSACAnalysis(freqs=sweep_freqs)
    """
