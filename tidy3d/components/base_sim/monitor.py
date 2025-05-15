"""Abstract bases for classes that define how data is recorded from simulation."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pydantic.v1 as pd

from ..base import cached_property
from ..geometry.base import Box
from ..types import ArrayFloat1D, Axis, Numpy
from ..validators import _warn_unsupported_traced_argument
from ..viz import PlotParams, plot_params_monitor


class AbstractMonitor(Box, ABC):
    """Abstract base class for steady-state monitors."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for monitor.",
        min_length=1,
    )

    _warn_traced_center = _warn_unsupported_traced_argument("center")
    _warn_traced_size = _warn_unsupported_traced_argument("size")

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
