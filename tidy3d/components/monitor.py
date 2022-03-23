"""Objects that define how data is recorded from simulation."""
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import pydantic
import numpy as np

from .types import Literal, Ax, EMField, ArrayLike, Array
from .geometry import Box
from .validators import assert_plane
from .mode import ModeSpec
from .viz import add_ax_if_none, equal_aspect, MonitorParams, ARROW_COLOR_MONITOR, ARROW_ALPHA
from ..log import SetupError
from ..constants import HERTZ, SECOND


BYTES_REAL = 4
BYTES_COMPLEX = 8


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

    @abstractmethod
    def storage_size(self, num_cells: int, tmesh: Array) -> int:
        """Size of monitor storage given the number of points after discretization.

        Parameters
        ----------
        num_cells : int
            Number of grid cells within the monitor after discretization by a :class:`Simulation`.
        tmesh : Array
            The discretized time mesh of a :class:`Simulation`.

        Returns
        -------
        int
            Number of bytes to be stored in monitor.
        """


class FreqMonitor(Monitor, ABC):
    """:class:`Monitor` that records data in the frequency-domain."""

    freqs: Union[List[float], ArrayLike] = pydantic.Field(
        ...,
        title="Frequencies",
        description="Array or list of frequencies stored by the field monitor.",
        units=HERTZ,
    )


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

    def time_inds(self, tmesh: Array) -> Tuple[int, int]:
        """Compute the starting and stopping index of the monitor in a given discrete time mesh."""

        tind_beg, tind_end = (0, 0)

        if tmesh.size == 0:
            return (tind_beg, tind_end)

        # If monitor.stop is None, record until the end
        t_stop = self.stop
        if t_stop is None:
            tind_end = int(tmesh.size)
            t_stop = tmesh[-1]
        else:
            tend = np.nonzero(tmesh <= t_stop)[0]
            if tend.size > 0:
                tind_end = int(tend[-1] + 1)

        # Step to compare to in order to handle t_start = t_stop
        if np.array(tmesh).size < 2:
            dt = 1e-20
        else:
            dt = tmesh[1] - tmesh[0]

        # If equal start and stopping time, record one time step
        if np.abs(self.start - t_stop) < dt:
            tind_beg = max(tind_end - 1, 0)
        else:
            tbeg = np.nonzero(tmesh[0:tind_end] >= self.start)[0]
            if tbeg.size > 0:
                tind_beg = tbeg[0]
            else:
                tind_beg = tind_end

        return (tind_beg, tind_end)

    def num_steps(self, tmesh: Array) -> int:
        """Compute number of time steps for a time monitor."""

        tind_beg, tind_end = self.time_inds(tmesh)
        number_of_steps = int((tind_end - tind_beg) / self.interval)
        return number_of_steps


class AbstractFieldMonitor(Monitor, ABC):
    """:class:`Monitor` that records electromagnetic field data as a function of x,y,z."""

    fields: List[EMField] = pydantic.Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor.",
    )


class PlanarMonitor(Monitor, ABC):
    """:class:`Monitor` that has a planar geometry."""

    _plane_validator = assert_plane()


class AbstractFluxMonitor(PlanarMonitor, ABC):
    """:class:`Monitor` that records flux through a plane."""


class AbstractModeMonitor(PlanarMonitor, FreqMonitor):
    """:class:`Monitor` that records mode-related data."""

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:

        # call the monitor.plot() function first
        ax = super().plot(x=x, y=y, z=z, ax=ax, **kwargs)

        # and then add an arrow using the direction comuputed from `_dir_arrow`.
        ax = self._plot_arrow(
            x=x,
            y=y,
            z=z,
            ax=ax,
            direction=self._dir_arrow,
            color=ARROW_COLOR_MONITOR,
            alpha=ARROW_ALPHA,
            both_dirs=True,
        )
        return ax

    @property
    def _dir_arrow(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        dx = np.cos(self.mode_spec.angle_phi) * np.sin(self.mode_spec.angle_theta)
        dy = np.sin(self.mode_spec.angle_phi) * np.sin(self.mode_spec.angle_theta)
        dz = np.cos(self.mode_spec.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self.size.index(0.0))


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

    _data_type: Literal["FieldData"] = pydantic.Field("FieldData")

    def storage_size(self, num_cells: int, tmesh: Array) -> int:
        # stores 1 complex number per grid cell, per frequency, per field
        return BYTES_COMPLEX * num_cells * len(self.freqs) * len(self.fields)

    def surfaces(self) -> List["FieldMonitor"]:  # pylint: disable=too-many-locals
        """Returns a list of 6 monitors corresponding to each surface of the field monitor.
        The output monitors are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. Each output monitor will have the same frequency/time
        data as the calling object. Its name will be that of the calling object appended with the
        above symbols. E.g., if the calling object's name is "field", the x+ monitor's name will be
        "field_x+". Does not work when the calling monitor has zero volume.

        Returns
        -------
        List[:class:`FieldMonitor`]
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

    _data_type: Literal["FieldTimeData"] = pydantic.Field("FieldTimeData")

    def storage_size(self, num_cells: int, tmesh: Array) -> int:
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)


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

    def storage_size(self, num_cells: int, tmesh: Array) -> int:
        # stores 1 real number per frequency
        return BYTES_REAL * len(self.freqs)


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

    def storage_size(self, num_cells: int, tmesh: Array) -> int:
        # stores 1 real number per time tep
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps


class ModeMonitor(AbstractModeMonitor):
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

    _data_type: Literal["ModeData"] = pydantic.Field("ModeData")

    def storage_size(self, num_cells: int, tmesh: int) -> int:
        # stores 3 complex numbers per frequency, per mode.
        return 3 * BYTES_COMPLEX * len(self.freqs) * self.mode_spec.num_modes


class ModeFieldMonitor(AbstractModeMonitor):
    """:class:`Monitor` that stores the mode field profiles returned by the mode solver in the
    monitor plane.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3)
    >>> monitor = ModeFieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')
    """

    _data_type: Literal["ModeFieldData"] = pydantic.Field("ModeFieldData")

    def storage_size(self, num_cells: int, tmesh: int) -> int:
        # fields store 6 complex numbers per grid cell, per frequency, per mode.
        field_size = 6 * BYTES_COMPLEX * num_cells * len(self.freqs) * self.mode_spec.num_modes
        return field_size


# types of monitors that are accepted by simulation
MonitorType = Union[
    FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor, ModeFieldMonitor
]
