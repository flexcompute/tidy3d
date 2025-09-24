"""Extrapolation into low frequencies specification."""

from __future__ import annotations

import pydantic.v1 as pydantic
from typing import Optional
from tidy3d.components.base import Tidy3dBaseModel


class LowFrequencySmoothingSpec(Tidy3dBaseModel):
    """Specifies the low frequency smoothing parameters for the terminal component simulation.
    The low frequency smoothing is performed by fitting a polynomial to the data in the trusted frequency range,
    defined by the minimum and maximum sampling times, and then using the polynomial to extrapolate
    the data outside of the trusted frequency range into lower frequencies.

    Example
    -------
    >>> low_freq_smoothing = LowFrequencySmoothingSpec(
    ...     min_sampling_time=3,
    ...     max_sampling_time=6,
    ...     order=1,
    ...     max_deviation=0.5,
    ... )
    """

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

    monitors: tuple[str, ...] = pydantic.Field(
        ...,
        title="Monitors",
        description="The monitors to use for the low frequency smoothing.",
    )

    @pydantic.root_validator(skip_on_failure=True)
    def _validate_sampling_times(cls, values):
        min_sampling_time = values.get("min_sampling_time")
        max_sampling_time = values.get("max_sampling_time")
        if min_sampling_time is not None and max_sampling_time is not None:
            if min_sampling_time >= max_sampling_time:
                raise ValueError(
                    "The minimum sampling time must be less than the maximum sampling time."
                )
        return values

    @pydantic.validator("monitors", always=True)
    def _validate_monitors(cls, val, values):
        """Validate the monitors list is not empty."""
        if len(val) == 0:
            raise ValueError("The monitors list must not be empty.")
        return val
