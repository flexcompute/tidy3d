"""Abstract bases for classes that define how data is recorded from simulation."""
from abc import ABC, abstractmethod
from typing import Tuple

import pydantic.v1 as pd
import numpy as np

from ..types import ArrayFloat1D, Numpy
from ..types import Direction, Axis, BoxSurface
from ..geometry.base import Box
from ..validators import assert_plane
from ..base import cached_property
from ..viz import PlotParams, plot_params_monitor
from ...constants import SECOND
from ...exceptions import SetupError
from ...log import log


class AbstractMonitor(Box, ABC):
    """Abstract base class for steady-state monitors."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for monitor.",
        min_length=1,
    )

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Monitor object."""
        return plot_params_monitor

    @cached_property
    def geometry(self) -> Box:
        """:class:`Box` representation of monitor.

        Returns
        -------
        :class:`Box`
            Representation of the monitor geometry as a :class:`Box`.
        """
        return Box(center=self.center, size=self.size)

    @abstractmethod
    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
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

    def downsample(self, arr: Numpy, axis: Axis) -> Numpy:
        """Downsample a 1D array making sure to keep the first and last entries, based on the
        spatial interval defined for the ``axis``.

        Parameters
        ----------
        arr : Numpy
            A 1D array of arbitrary type.
        axis : Axis
            Axis for which to select the interval_space defined for the monitor.

        Returns
        -------
        Numpy
            Downsampled array.
        """

        size = len(arr)
        interval = self.interval_space[axis]
        # There should always be at least 3 indices for "surface" monitors. Also, if the
        # size along this dim is already smaller than the interval, then don't downsample.
        if size < 4 or (size - 1) <= interval:
            return arr
        # make sure the last index is always included
        inds = np.arange(0, size, interval)
        if inds[-1] != size - 1:
            inds = np.append(inds, size - 1)
        return arr[inds]

    def downsampled_num_cells(self, num_cells: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Given a tuple of the number of cells spanned by the monitor along each dimension,
        return the number of cells one would have after downsampling based on ``interval_space``.
        """
        arrs = [np.arange(ncells) for ncells in num_cells]
        return tuple((self.downsample(arr, axis=dim).size for dim, arr in enumerate(arrs)))


class AbstractTimeMonitor(AbstractMonitor, ABC):
    """Abstract base class for transient monitors."""

    start: pd.NonNegativeFloat = pd.Field(
        0.0,
        title="Start time",
        description="Time at which to start monitor recording.",
        units=SECOND,
    )

    stop: pd.NonNegativeFloat = pd.Field(
        None,
        title="Stop time",
        description="Time at which to stop monitor recording.  "
        "If not specified, record until end of simulation.",
        units=SECOND,
    )

    interval: pd.PositiveInt = pd.Field(
        1,
        title="Time interval",
        description="Number of time step intervals between monitor recordings.",
    )

    @pd.validator("stop", always=True, allow_reuse=True)
    def stop_greater_than_start(cls, val, values):
        """Ensure sure stop is greater than or equal to start."""
        start = values.get("start")
        if val and val < start:
            raise SetupError("Monitor start time is greater than stop time.")
        return val

    def time_inds(self, tmesh: ArrayFloat1D) -> Tuple[int, int]:
        """Compute the starting and stopping index of the monitor in a given discrete time mesh."""

        tmesh = np.array(tmesh)
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
        dt = 1e-20 if np.array(tmesh).size < 2 else tmesh[1] - tmesh[0]
        # If equal start and stopping time, record one time step
        if np.abs(self.start - t_stop) < dt and self.start <= tmesh[-1]:
            tind_beg = max(tind_end - 1, 0)
        else:
            tbeg = np.nonzero(tmesh[:tind_end] >= self.start)[0]
            tind_beg = tbeg[0] if tbeg.size > 0 else tind_end
        return (tind_beg, tind_end)

    def num_steps(self, tmesh: ArrayFloat1D) -> int:
        """Compute number of time steps for a time monitor."""

        tind_beg, tind_end = self.time_inds(tmesh)
        return int((tind_end - tind_beg) / self.interval)


class AbstractPlanarMonitor(AbstractMonitor, ABC):
    """:class:`AbstractMonitor` that has a planar geometry."""

    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the monitor's plane."""
        return self.size.index(0.0)


class AbstractSurfaceIntegrationMonitor(AbstractMonitor, ABC):
    """Abstract class for monitors that perform surface integrals during the solver run."""

    normal_dir: Direction = pd.Field(
        None,
        title="Normal vector orientation",
        description="Direction of the surface monitor's normal vector w.r.t. "
        "the positive x, y or z unit vectors. Must be one of ``'+'`` or ``'-'``. "
        "Applies to surface monitors only, and defaults to ``'+'`` if not provided.",
    )

    exclude_surfaces: Tuple[BoxSurface, ...] = pd.Field(
        None,
        title="Excluded surfaces",
        description="Surfaces to exclude in the integration, if a volume monitor.",
    )

    @property
    def integration_surfaces(self):
        """Surfaces of the monitor where fields will be recorded for subsequent integration."""
        if self.size.count(0.0) == 0:
            return self.surfaces_with_exclusion(**self.dict())
        return [self]

    @pd.root_validator(skip_on_failure=True)
    def normal_dir_exists_for_surface(cls, values):
        """If the monitor is a surface, set default ``normal_dir`` if not provided.
        If the monitor is a box, warn that ``normal_dir`` is relevant only for surfaces."""
        normal_dir = values.get("normal_dir")
        name = values.get("name")
        size = values.get("size")
        if size.count(0.0) != 1:
            if normal_dir is not None:
                log.warning(
                    "The ``normal_dir`` field is relevant only for surface monitors "
                    f"and will be ignored for monitor {name}, which is a box."
                )
        else:
            if normal_dir is None:
                values["normal_dir"] = "+"
        return values

    @pd.root_validator(skip_on_failure=True)
    def check_excluded_surfaces(cls, values):
        """Error if ``exclude_surfaces`` is provided for a surface monitor."""
        exclude_surfaces = values.get("exclude_surfaces")
        if exclude_surfaces is None:
            return values
        name = values.get("name")
        size = values.get("size")
        if size.count(0.0) > 0:
            raise SetupError(
                f"Can't specify ``exclude_surfaces`` for surface monitor {name}; "
                "valid for box monitors only."
            )
        return values
