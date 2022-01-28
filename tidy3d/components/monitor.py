"""Objects that define how data is recorded from simulation."""
from abc import ABC
from typing import List, Union

import pydantic

from .types import Literal, Ax, EMField, ArrayLike
from .geometry import Box
from .validators import assert_plane
from .mode import ModeSpec
from .viz import add_ax_if_none, equal_aspect, MonitorParams
from ..log import SetupError
from ..constants import HERTZ, SECOND


class Monitor(Box, ABC):
    """Abstract base class for monitors."""

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique name for monitor.",
        min_length=1,
    )

    @equal_aspect
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

    fields: List[EMField] = pydantic.Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor.",
    )

    def surfaces(self) -> List["AbstractFieldMonitor"]:  # pylint: disable=too-many-locals
        """Returns a list of 6 monitors corresponding to each surface of the field monitor.
        The output monitors are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. Each output monitor will have the same frequency/time
        data as the calling object. Its name will be that of the calling object appended with the
        above symbols. E.g., if the calling object's name is "field", the x+ monitor's name will be
        "field_x+". Does not work when the calling monitor has zero volume.

        Returns
        -------
        List[:class:`AbstractFieldMonitor`]
            List of 6 surface monitors for each side of the field monitor.

        Example
        -------
        >>> volume_monitor = FieldMonitor(center=(0,0,0), size=(1,2,3), freqs=[2e14], name='field')
        >>> surface_monitors = volume_monitor.surfaces()
        """

        if any(s == 0.0 for s in self.size):
            raise SetupError(
                "Can't generate surfaces for the given monitor because it has zero volume."
            )

        self_bmin, self_bmax = self.bounds
        center_x, center_y, center_z = self.center
        size_x, size_y, size_z = self.size

        # Set up geometry data and names for each surface:

        surface_centers = (
            (self_bmin[0], center_y, center_z),  # x-
            (self_bmax[0], center_y, center_z),  # x+
            (center_x, self_bmin[1], center_z),  # y-
            (center_x, self_bmax[1], center_z),  # y+
            (center_x, center_y, self_bmin[2]),  # z-
            (center_x, center_y, self_bmax[2]),  # z+
        )

        surface_sizes = (
            (0.0, size_y, size_z),  # x-
            (0.0, size_y, size_z),  # x+
            (size_x, 0.0, size_z),  # y-
            (size_x, 0.0, size_z),  # y+
            (size_x, size_y, 0.0),  # z-
            (size_x, size_y, 0.0),  # z+
        )

        surface_names = (
            self.name + "_x-",
            self.name + "_x+",
            self.name + "_y-",
            self.name + "_y+",
            self.name + "_z-",
            self.name + "_z+",
        )

        # Create "surface" monitors
        monitors = []
        for center, size, name in zip(surface_centers, surface_sizes, surface_names):
            mon_new = self.copy(deep=True)
            mon_new.center = center
            mon_new.size = size
            mon_new.name = name
            monitors.append(mon_new)

        return monitors


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

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    _data_type: Literal["ModeData"] = pydantic.Field("ModeData")


# types of monitors that are accepted by simulation
MonitorType = Union[FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor]
