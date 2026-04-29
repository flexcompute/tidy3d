"""Extrapolation into low frequencies specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, NonNegativeFloat, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel

if TYPE_CHECKING:
    from tidy3d.compat import Self


class AbstractLowFrequencySmoothingSpec(Tidy3dBaseModel):
    """Abstract base class for low frequency smoothing specifications."""

    min_sampling_time: NonNegativeFloat = Field(
        1.0,
        title="Minimum Sampling Time (periods)",
        description="The minimum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results below this threshold will be completely discarded.",
    )

    max_sampling_time: NonNegativeFloat = Field(
        5.0,
        title="Maximum Sampling Time (periods)",
        description="The maximum simulation time in periods of the corresponding frequency for which frequency domain results will be used to fit the polynomial for the low frequency extrapolation. "
        "Results above this threshold will be not be modified.",
    )

    order: int = Field(
        1,
        title="Extrapolation Order",
        description="The order of the polynomial to use for the low frequency extrapolation.",
        ge=0,
        le=3,
    )

    max_deviation: float | None = Field(
        0.5,
        title="Maximum Deviation",
        description="The maximum deviation (in fraction of the trusted values) to allow for the low frequency smoothing.",
        ge=0,
    )

    @model_validator(mode="after")
    def _validate_sampling_times(self) -> Self:
        min_sampling_time = self.min_sampling_time
        max_sampling_time = self.max_sampling_time
        if min_sampling_time >= max_sampling_time:
            raise ValueError(
                "The minimum sampling time must be less than the maximum sampling time."
            )
        return self


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

    monitors: tuple[str, ...] = Field(
        title="Monitors",
        description="The names of monitors to which low frequency smoothing will be applied.",
    )

    @field_validator("monitors")
    @classmethod
    def _validate_monitors(cls, val: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the monitors list is not empty."""
        if not val:
            raise ValueError("The monitors list must not be empty.")
        return val
