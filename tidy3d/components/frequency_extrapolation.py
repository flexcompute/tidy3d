"""Extrapolation into low frequencies specification."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pydantic

from tidy3d.components.base import Tidy3dBaseModel


class AbstractLowFrequencySmoothingSpec(Tidy3dBaseModel):
    """Abstract base class for low frequency smoothing specifications."""

    min_sampling_time: pydantic.NonNegativeFloat = pydantic.Field(
        1.0,
        title="Minimum Sampling Time (periods)",
        description="The minimum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results below this threshold will be completely discarded.",
    )

    max_sampling_time: pydantic.NonNegativeFloat = pydantic.Field(
        5.0,
        title="Maximum Sampling Time (periods)",
        description="The maximum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results above this threshold will be not be modified.",
    )

    order: int = pydantic.Field(
        1,
        title="Extrapolation Order",
        description="The order of the polynomial to use for the low frequency extrapolation.",
        ge=0,
        le=3,
    )

    max_deviation: Optional[float] = pydantic.Field(
        0.5,
        title="Maximum Deviation",
        description="The maximum deviation (in fraction of the trusted values) to allow for the low frequency smoothing.",
        ge=0,
    )

    @pydantic.root_validator(skip_on_failure=True)
    def _validate_sampling_times(cls, values):
        min_sampling_time = values.get("min_sampling_time")
        max_sampling_time = values.get("max_sampling_time")
        if min_sampling_time >= max_sampling_time:
            raise ValueError(
                "The minimum sampling time must be less than the maximum sampling time."
            )
        return values


class LowFrequencySmoothingSpec(AbstractLowFrequencySmoothingSpec):
    """Specifies the low frequency smoothing parameters for the simulation.
    This specification affects only results recorded in mode monitors. Specifically, the mode decomposition data
    for frequencies for which the total simulation time in units of the corresponding period (T = 1/f) is less than
    the specified minimum sampling time will be overridden by extrapolation from the data in the trusted frequency range.
    The trusted frequency range is defined in terms of minimum and maximum sampling times (the total simulation time divided by the corresponding period).
    Example
    -------
    >>> low_freq_smoothing = LowFrequencySmoothingSpec(
    ...     min_sampling_time=3,
    ...     max_sampling_time=6,
    ...     order=1,
    ...     max_deviation=0.5,
    ...     monitors=("monitor1", "monitor2"),
    ... )
    """

    monitors: tuple[str, ...] = pydantic.Field(
        ...,
        title="Monitors",
        description="The names of monitors to which low frequency smoothing will be applied.",
    )

    @pydantic.validator("monitors", always=True)
    def _validate_monitors(cls, val, values):
        """Validate the monitors list is not empty."""
        if not val:
            raise ValueError("The monitors list must not be empty.")
        return val
