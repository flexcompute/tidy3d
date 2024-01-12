"""Defines specification for apodization."""

import pydantic as pd
import numpy as np

from .base import Tidy3dBaseModel
from ..constants import SECOND
from ..exceptions import SetupError
from .types import ArrayFloat1D, Ax
from .viz import add_ax_if_none


class ApodizationSpec(Tidy3dBaseModel):
    """Stores specifications for the apodizaton of frequency-domain monitors.

    Example
    -------
    >>> apod_spec = ApodizationSpec(start=1, end=2, width=0.5)
    """

    start: pd.NonNegativeFloat = pd.Field(
        None,
        title="Start Interval",
        description="Defines the time at which the start apodization ends.",
        units=SECOND,
    )

    end: pd.NonNegativeFloat = pd.Field(
        None,
        title="End Interval",
        description="Defines the time at which the end apodization begins.",
        units=SECOND,
    )

    width: pd.PositiveFloat = pd.Field(
        None,
        title="Apodization Width",
        description="Characteristic decay length of the apodization function.",
        units=SECOND,
    )

    @pd.validator("end", always=True, allow_reuse=True)
    def end_greater_than_start(cls, val, values):
        """Ensure end is greater than or equal to start."""
        start = values.get("start")
        if val is not None and start is not None and val < start:
            raise SetupError("End apodization begins before start apodization ends.")
        return val

    @pd.validator("width", always=True, allow_reuse=True)
    def width_provided(cls, val, values):
        """Check that width is provided if either start or end apodization is requested."""
        start = values.get("start")
        end = values.get("end")
        if (start is not None or end is not None) and val is None:
            raise SetupError("Apodization width must be set.")
        return val

    @add_ax_if_none
    def plot(self, times: ArrayFloat1D, ax: Ax = None) -> Ax:
        """Plot the apodization function.

        Parameters
        ----------
        times : np.ndarray
            Array of times (seconds) to plot source at.
            To see source time amplitude for a specific :class:`Simulation`,
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
