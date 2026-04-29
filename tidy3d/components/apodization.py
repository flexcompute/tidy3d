"""Defines specification for apodization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from tidy3d.constants import SECOND

from .base import Tidy3dBaseModel
from .viz import add_ax_if_none

if TYPE_CHECKING:
    from tidy3d.compat import Self

    from .types import ArrayFloat1D, Ax


class ApodizationSpec(Tidy3dBaseModel):
    """Stores specifications for the apodizaton of frequency-domain monitors.

    Example
    -------
    >>> apod_spec = ApodizationSpec(start=1, end=2, width=0.2)


    .. image:: ../../_static/img/apodization.png
        :width: 80%
        :align: center

    """

    start: NonNegativeFloat | None = Field(
        None,
        title="Start Interval",
        description="Defines the time at which the start apodization ends.",
        json_schema_extra={"units": SECOND},
    )

    end: NonNegativeFloat | None = Field(
        None,
        title="End Interval",
        description="Defines the time at which the end apodization begins.",
        json_schema_extra={"units": SECOND},
    )

    width: PositiveFloat | None = Field(
        None,
        title="Apodization Width",
        description="Characteristic decay length of the apodization function, i.e., the width of the ramping up of the scaling function from 0 to 1.",
        json_schema_extra={"units": SECOND},
    )

    @model_validator(mode="after")
    def end_greater_than_start(self) -> Self:
        """Ensure end is greater than or equal to start."""
        if self.end is not None and self.start is not None and self.end < self.start:
            self._raise_validation_error_at_loc(
                "End apodization begins before start apodization ends.", "end"
            )
        return self

    @model_validator(mode="after")
    def width_provided(self) -> Self:
        """Check that width is provided if either start or end apodization is requested."""
        if (self.start is not None or self.end is not None) and self.width is None:
            self._raise_validation_error_at_loc("Apodization width must be set.", "width")
        return self

    @add_ax_if_none
    def plot(self, times: ArrayFloat1D, ax: Ax = None) -> Ax:
        """Plot the apodization function.

        Parameters
        ----------
        times : np.ndarray
            Array of times (seconds) to plot source at.
            To see source time amplitude for a specific :class:`.Simulation`,
            pass ``simulation.tmesh``.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        times = np.array(times)
        amp = np.ones_like(times)

        if self.start is not None:
            start_ind = times < self.start
            time_scaled = (times[start_ind] - self.start) / self.width
            amp[start_ind] *= np.exp(-0.5 * time_scaled**2)

        if self.end is not None:
            end_ind = times > self.end
            time_scaled = (times[end_ind] - self.end) / self.width
            amp[end_ind] *= np.exp(-0.5 * time_scaled**2)

        ax.plot(times, amp, color="blueviolet")
        ax.set_xlabel("time (s)")
        ax.set_title("apodization function")
        ax.set_aspect("auto")
        return ax
