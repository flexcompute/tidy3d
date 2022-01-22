"""Objects that define how data is recorded from simulation."""
from __future__ import annotations
from abc import ABC
from typing import List, Union

import pydantic
import numpy as np

from .types import Literal, Ax, Direction, EMField, ArrayLike, FieldType
from .geometry import Box
from .validators import assert_plane, validate_name_str
from .mode import ModeSpec
from .viz import add_ax_if_none, MonitorParams
from ..log import SetupError, ValidationError
from ..constants import HERTZ, SECOND


class Monitor(Box, ABC):
    """Abstract base class for monitors."""

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique name for monitor.",
        min_length=1,
    )

    @add_ax_if_none
    def plot(  # pylint:disable=duplicate-code
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:

        kwargs = MonitorParams().update_params(**kwargs)
        ax = self.geometry.plot(x=x, y=y, z=z, ax=ax, **kwargs)
        return ax

    @property
    def geometry(self):
        """:class:`Box` representation of monitor.

        Returns
        -------
        :class:`Box`
            Representation of the monitor geometry as a :class:`Box`.
        """
        return Box(center=self.center, size=self.size)


class FreqMonitor(Monitor, ABC):
    """:class:`Monitor` that records data in the frequency-domain."""

    freqs: Union[List[float], ArrayLike] = pydantic.Field(
        ...,
        title="Frequencies",
        description="Array or list of frequencies stored by the field monitor.",
        units=HERTZ,
        min_items=1,
    )

    # @pydantic.validator("freqs", always=True)
    # def freqs_nonempty(cls, val):
    #     """Ensure freqs has at least one element"""
    #     if len(val) == 0:
    #         raise ValidationError("Monitor 'freqs' should have at least one element.")
    #     return


class TimeMonitor(Monitor, ABC):
    """:class:`Monitor` that records data in the time-domain."""

    start: pydantic.NonNegativeFloat = pydantic.Field(
        0.0,
        title="Start time",
        description="Time at which to start monitor recording.",
        units=SECOND,
    )

    stop: pydantic.NonNegativeFloat = pydantic.Field(
        None,
        title="Stop time",
        description="Time at which to stop monitor recording.  "
        "If not specified, record until end of simulation.",
        units=SECOND,
    )

    interval: pydantic.PositiveInt = pydantic.Field(
        1,
        title="Time interval",
        description="Number of time step intervals between monitor recordings.",
    )

    @pydantic.validator("stop", always=True, allow_reuse=True)
    def stop_greater_than_start(cls, val, values):
        """Ensure sure stop is greater than or equal to start."""
        start = values.get("start")
        if val and val < start:
            raise SetupError("Monitor start time is greater than stop time.")
        return val


class AbstractFieldMonitor(Monitor, ABC):
    """:class:`Monitor` that records electromagnetic field data as a function of x,y,z."""

    fields: List = pydantic.Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor.",
        min_items=1,
    )


class PlanarMonitor(Monitor, ABC):
    """:class:`Monitor` that has a planar geometry."""

    _plane_validator = assert_plane()


class AbstractFluxMonitor(PlanarMonitor, ABC):
    """:class:`Monitor` that records flux through a plane"""


class FieldMonitor(AbstractFieldMonitor, FreqMonitor):
    """:class:`Monitor` that records electromagnetic fields in the frequency domain.

    Example
    -------
    >>> monitor = FieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['Hx'],
    ...     freqs=[250e12, 300e12],
    ...     name='steady_state_monitor')
    """

    _data_type: Literal["ScalarFieldData"] = pydantic.Field("ScalarFieldData")


class FieldTimeMonitor(AbstractFieldMonitor, TimeMonitor):
    """:class:`Monitor` that records electromagnetic fields in the time domain.

    Example
    -------
    >>> monitor = FieldTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['Hx'],
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='movie_monitor')
    """

    _data_type: Literal["ScalarFieldTimeData"] = pydantic.Field("ScalarFieldTimeData")


class FluxMonitor(AbstractFluxMonitor, FreqMonitor):
    """:class:`Monitor` that records power flux through a plane in the frequency domain.

    Example
    -------
    >>> monitor = FluxMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     name='flux_monitor')
    """

    _data_type: Literal["FluxData"] = pydantic.Field("FluxData")


class FluxTimeMonitor(AbstractFluxMonitor, TimeMonitor):
    """:class:`Monitor` that records power flux through a plane in the time domain.

    Example
    -------
    >>> monitor = FluxTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='flux_vs_time')
    """

    _data_type: Literal["FluxTimeData"] = pydantic.Field("FluxTimeData")


class ModeMonitor(PlanarMonitor, FreqMonitor):
    """:class:`Monitor` that records amplitudes from modal decomposition of fields on plane.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3)
    >>> monitor = ModeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')
    """

    direction: List[Direction] = pydantic.Field(
        ["+", "-"],
        title="Direction",
        description="Specifies which direction(s) of mode propagation for the monitor to measure.",
    )

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    _data_type: Literal["ModeData"] = pydantic.Field("ModeData")


# types of monitors that are accepted by simulation
MonitorType = Union[FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor]
