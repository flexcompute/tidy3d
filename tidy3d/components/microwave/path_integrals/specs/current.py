"""Module containing specifications for current path integrals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from pydantic import Field, field_validator

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.bound_ops import bounds_union
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.base import (
    AbstractAxesRH,
    AxisAlignedPathIntegralSpec,
    Custom2DPathIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.viz import ARROW_CURRENT, plot_params_current_path
from tidy3d.components.types import ArrayFloat2D
from tidy3d.components.types.base import Direction
from tidy3d.components.validators import assert_plane
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import MICROMETER, fp_eps
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from typing import Optional

    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.types import Ax, Bound
    from tidy3d.components.types.base import Axis


class AxisAlignedCurrentIntegralSpec(AbstractAxesRH, Box):
    """Class for specifying the computation of conduction current via Ampère's circuital law on an axis-aligned loop.

    Example
    -------
    >>> current_spec = AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0),
    ...     size=(1, 1, 0),
    ...     sign="+",
    ...     snap_contour_to_grid=True
    ... )
    """

    _plane_validator = assert_plane()

    sign: Direction = Field(
        title="Direction of Contour Integral",
        description="Positive indicates current flowing in the positive normal axis direction.",
    )

    extrapolate_to_endpoints: bool = Field(
        False,
        title="Extrapolate to Endpoints",
        description="This parameter is passed to :class:`AxisAlignedPathIntegral` objects when computing the contour integral.",
    )

    snap_contour_to_grid: bool = Field(
        False,
        title="Snap Contour to Grid",
        description="This parameter is passed to :class:`AxisAlignedPathIntegral` objects when computing the contour integral.",
    )

    @cached_property
    def main_axis(self) -> Axis:
        """Axis normal to loop"""
        for index, value in enumerate(self.size):
            if value == 0:
                return index
        raise SetupError("AxisAlignedCurrentIntegralSpec requires a zero-sized dimension.")

    def _to_path_integral_specs(
        self,
        h_horizontal: Optional[DataArray] = None,
        h_vertical: Optional[DataArray] = None,
    ) -> tuple[AxisAlignedPathIntegralSpec, ...]:
        """Returns four ``AxisAlignedPathIntegralSpec`` instances, which represent a contour
        integral around the surface defined by ``self.size``."""
        ax1 = self.remaining_axes[0]
        ax2 = self.remaining_axes[1]

        horizontal_passed = h_horizontal is not None
        vertical_passed = h_vertical is not None
        if self.snap_contour_to_grid and horizontal_passed and vertical_passed:
            (coord1, coord2) = self.remaining_dims

            # Locations where horizontal paths will be snapped
            v_bounds = [
                self.center[ax2] - self.size[ax2] / 2,
                self.center[ax2] + self.size[ax2] / 2,
            ]
            h_snaps = h_horizontal.sel({coord2: v_bounds}, method="nearest").coords[coord2].values
            # Locations where vertical paths will be snapped
            h_bounds = [
                self.center[ax1] - self.size[ax1] / 2,
                self.center[ax1] + self.size[ax1] / 2,
            ]
            v_snaps = h_vertical.sel({coord1: h_bounds}, method="nearest").coords[coord1].values

            bottom_bound = h_snaps[0]
            top_bound = h_snaps[1]
            left_bound = v_snaps[0]
            right_bound = v_snaps[1]
        else:
            bottom_bound = self.bounds[0][ax2]
            top_bound = self.bounds[1][ax2]
            left_bound = self.bounds[0][ax1]
            right_bound = self.bounds[1][ax1]

        # Horizontal paths
        path_size = list(self.size)
        path_size[ax1] = right_bound - left_bound
        path_size[ax2] = 0
        path_center = list(self.center)
        path_center[ax2] = bottom_bound

        bottom = AxisAlignedPathIntegralSpec(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )
        path_center[ax2] = top_bound
        top = AxisAlignedPathIntegralSpec(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )

        # Vertical paths
        path_size = list(self.size)
        path_size[ax1] = 0
        path_size[ax2] = top_bound - bottom_bound
        path_center = list(self.center)

        path_center[ax1] = left_bound
        left = AxisAlignedPathIntegralSpec(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )
        path_center[ax1] = right_bound
        right = AxisAlignedPathIntegralSpec(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )

        return (bottom, right, top, left)

    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        plot_arrow: bool = True,
        **path_kwargs: Any,
    ) -> Ax:
        """Plot path integral at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        plot_arrow : bool = True
            Whether to plot the arrow indicating current direction. Default is ``True``.
        **path_kwargs
            Optional keyword arguments passed to the matplotlib plotting of the line.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/36marrat>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis != self.main_axis or not np.isclose(position, self.center[axis], rtol=fp_eps):
            return ax

        plot_params = plot_params_current_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        path_integrals = self._to_path_integral_specs()
        # Plot the path
        for path in path_integrals:
            (xs, ys) = path._vertices_2D(axis)
            ax.plot(xs, ys, **plot_kwargs)

        # Add arrow to bottom path, unless right path is longer
        if plot_arrow:
            (ax1, ax2) = self.remaining_axes

            arrow_path = path_integrals[0]
            if self.size[ax2] > self.size[ax1]:
                arrow_path = path_integrals[1]

            (xs, ys) = arrow_path._vertices_2D(axis)
            X = (xs[0] + xs[1]) / 2
            Y = (ys[0] + ys[1]) / 2
            center = np.array([X, Y])
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            direction = np.array([dx, dy])
            segment_length = np.linalg.norm(direction)
            unit_dir = direction / segment_length

            # Change direction of arrow depending on sign of current definition
            if self.sign == "-":
                unit_dir *= -1.0
            # Change direction of arrow when the "y" axis is dropped,
            # since the plotted coordinate system will be left-handed (x, z)
            if self.main_axis == 1:
                unit_dir *= -1.0

            start = center - unit_dir * segment_length
            end = center
            ax.annotate(
                "",
                xytext=(start[0], start[1]),
                xy=(end[0], end[1]),
                arrowprops=ARROW_CURRENT,
            )
        return ax


class Custom2DCurrentIntegralSpec(Custom2DPathIntegralSpec):
    """Class for specifying the computation of conduction current via Ampère's circuital law on a custom path.
    To compute the current flowing in the positive ``axis`` direction, the vertices should be
    ordered in a counterclockwise direction.

    Example
    -------
    >>> import numpy as np
    >>> vertices = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
    >>> current_spec = Custom2DCurrentIntegralSpec(
    ...     axis=2,
    ...     position=0,
    ...     vertices=vertices
    ... )
    """

    vertices: ArrayFloat2D = Field(
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the path. "
        "The index of dimension should be in the ascending order, which means "
        "if the axis corresponds with ``y``, the coordinates of the vertices should be (x, z). "
        "The path must form a closed contour, i.e., ``vertices[-1] == vertices[0]``.",
        json_schema_extra={"units": MICROMETER},
    )

    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        plot_arrow: bool = True,
        **path_kwargs: Any,
    ) -> Ax:
        """Plot path integral at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        plot_arrow : bool = True
            Whether to plot the arrow indicating current direction. Default is ``True``.
        **path_kwargs
            Optional keyword arguments passed to the matplotlib plotting of the line.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/36marrat>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis != self.main_axis or not np.isclose(position, self.position, rtol=fp_eps):
            return ax

        plot_params = plot_params_current_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        xs = self.vertices[:, 0]
        ys = self.vertices[:, 1]
        ax.plot(xs, ys, **plot_kwargs)

        # Add arrow at start of contour
        if plot_arrow:
            ax.annotate(
                "",
                xytext=(xs[0], ys[0]),
                xy=(xs[1], ys[1]),
                arrowprops=ARROW_CURRENT,
            )
        return ax


class CompositeCurrentIntegralSpec(MicrowaveBaseModel):
    """Specification for a composite current integral.

    Notes
    -----
        This class is used to set up a ``CompositeCurrentIntegral``, which combines
        multiple current integrals. It does not perform any integration itself.

    Example
    -------
    >>> spec1 = AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(1, 1, 0), sign="+"
    ... )
    >>> spec2 = AxisAlignedCurrentIntegralSpec(
    ...     center=(2, 0, 0), size=(1, 1, 0), sign="+"
    ... )
    >>> composite_spec = CompositeCurrentIntegralSpec(
    ...     path_specs=(spec1, spec2),
    ...     sum_spec="sum"
    ... )
    """

    path_specs: tuple[Union[AxisAlignedCurrentIntegralSpec, Custom2DCurrentIntegralSpec], ...] = (
        Field(
            title="Path Specifications",
            description="Definition of the disjoint path specifications for each isolated contour integral.",
        )
    )

    sum_spec: Literal["sum", "split"] = Field(
        title="Sum Specification",
        description="Determines the method used to combine the currents calculated by the different "
        "current integrals defined by ``path_specs``. ``sum`` simply adds all currents, while ``split`` "
        "keeps contributions with opposite phase separate, which allows for isolating the current "
        "flowing in opposite directions. In ``split`` version, the current returned is the maximum "
        "of the two contributions.",
    )

    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        plot_arrow: bool = True,
        **path_kwargs: Any,
    ) -> Ax:
        """Plot path integral at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        plot_arrow : bool = True
            Whether to plot the arrow indicating current direction. Default is ``True``.
        **path_kwargs
            Optional keyword arguments passed to the matplotlib plotting of the line.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/36marrat>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        for path_spec in self.path_specs:
            ax = path_spec.plot(x=x, y=y, z=z, ax=ax, plot_arrow=plot_arrow, **path_kwargs)
        return ax

    @field_validator("path_specs")
    @classmethod
    def _path_specs_not_empty(
        cls, val: tuple[Union[AxisAlignedCurrentIntegralSpec, Custom2DCurrentIntegralSpec], ...]
    ) -> tuple[Union[AxisAlignedCurrentIntegralSpec, Custom2DCurrentIntegralSpec], ...]:
        """Makes sure at least one path spec has been supplied"""
        # overall shape of vertices
        if len(val) < 1:
            raise SetupError(
                "'CompositeCurrentIntegralSpec.path_specs' must be a list of one or more current integrals. "
            )
        return val

    @cached_property
    def bounds(self) -> Bound:
        """Return the overall bounding box of all path specifications.

        Computed by taking the union of bounds from all path specs.

        Returns
        -------
        Bound
            Tuple of (rmin, rmax) where rmin and rmax are tuples of (x, y, z) coordinates
            representing the minimum and maximum corners of the bounding box.
        """
        # Start with bounds of first path spec
        overall_bounds = self.path_specs[0].bounds

        # Union with bounds of remaining path specs
        for path_spec in self.path_specs[1:]:
            overall_bounds = bounds_union(overall_bounds, path_spec.bounds)

        return overall_bounds
