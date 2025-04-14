"""Module containing specifications for path integrals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pydantic.v1 as pd
import shapely
import xarray as xr
from typing_extensions import Self

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.types import ArrayFloat2D, Bound, Coordinate, Coordinate2D
from tidy3d.components.types.base import Axis, Direction
from tidy3d.components.validators import assert_line
from tidy3d.constants import MICROMETER, fp_eps
from tidy3d.exceptions import SetupError


class AbstractAxesRH(MicrowaveBaseModel, ABC):
    """Represents an axis-aligned right-handed coordinate system with one axis preferred.
    Typically `main_axis` would refer to the normal axis of a plane.
    """

    @cached_property
    @abstractmethod
    def main_axis(self) -> Axis:
        """Get the preferred axis."""

    @cached_property
    def remaining_axes(self) -> tuple[Axis, Axis]:
        """Get in-plane axes, ordered to maintain a right-handed coordinate system."""
        axes: list[Axis] = [0, 1, 2]
        axes.pop(self.main_axis)
        if self.main_axis == 1:
            return (axes[1], axes[0])
        else:
            return (axes[0], axes[1])

    @cached_property
    def remaining_dims(self) -> tuple[str, str]:
        """Get in-plane dimensions, ordered to maintain a right-handed coordinate system."""
        dim1 = "xyz"[self.remaining_axes[0]]
        dim2 = "xyz"[self.remaining_axes[1]]
        return (dim1, dim2)

    @cached_property
    def local_dims(self) -> tuple[str, str, str]:
        """Get in-plane dimensions with in-plane dims first, followed by the `main_axis` dimension."""
        dim3 = "xyz"[self.main_axis]
        return self.remaining_dims + tuple(dim3)


class AxisAlignedPathIntegralSpec(AbstractAxesRH, Box):
    """Class for defining the simplest type of path integral, which is aligned with Cartesian axes.

    Example
    -------
    >>> path_spec = AxisAlignedPathIntegralSpec(
    ...     center=(0, 0, 1),
    ...     size=(0, 0, 2),
    ...     extrapolate_to_endpoints=True
    ... )
    """

    _line_validator = assert_line()

    extrapolate_to_endpoints: bool = pd.Field(
        False,
        title="Extrapolate to Endpoints",
        description="If the endpoints of the path integral terminate at or near a material interface, "
        "the field is likely discontinuous. When this field is ``True``, fields that are outside and on the bounds "
        "of the integral are ignored. Should be enabled when computing voltage between two conductors.",
    )

    snap_path_to_grid: bool = pd.Field(
        False,
        title="Snap Path to Grid",
        description="It might be desirable to integrate exactly along the Yee grid associated with "
        "a field. When this field is ``True``, the integration path will be snapped to the grid.",
    )

    @cached_property
    def main_axis(self) -> Axis:
        """Axis for performing integration."""
        for index, value in enumerate(self.size):
            if value != 0:
                return index

    def _vertices_2D(self, axis: Axis) -> tuple[Coordinate2D, Coordinate2D]:
        """Returns the two vertices of this path in the plane defined by ``axis``."""
        min = self.bounds[0]
        max = self.bounds[1]
        _, min = Box.pop_axis(min, axis)
        _, max = Box.pop_axis(max, axis)

        u = [min[0], max[0]]
        v = [min[1], max[1]]
        return (u, v)


class Custom2DPathIntegralSpec(AbstractAxesRH):
    """Class for specifying a custom path integral defined as a curve on an axis-aligned plane.

    Example
    -------
    >>> import numpy as np
    >>> vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> path_spec = Custom2DPathIntegralSpec(
    ...     axis=2,
    ...     position=0.5,
    ...     vertices=vertices
    ... )

    Notes
    -----

    Given a set of vertices :math:`\\vec{r}_i`, this class approximates path integrals over
    vector fields of the form :math:`\\int{\\vec{F} \\cdot \\vec{dl}}`
    as :math:`\\sum_i{\\vec{F}(\\vec{r}_i) \\cdot \\vec{dl}_i}`,
    where the differential length :math:`\\vec{dl}` is approximated using central differences
    :math:`\\vec{dl}_i = \\frac{\\vec{r}_{i+1} - \\vec{r}_{i-1}}{2}`.
    If the path is not closed, forward and backward differences are used at the endpoints.
    """

    axis: Axis = pd.Field(
        ..., title="Axis", description="Specifies dimension of the planar axis (0,1,2) -> (x,y,z)."
    )

    position: float = pd.Field(
        ...,
        title="Position",
        description="Position of the plane along the ``axis``.",
    )

    vertices: ArrayFloat2D = pd.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the path. "
        "The index of dimension should be in the ascending order, which means "
        "if the axis corresponds with ``y``, the coordinates of the vertices should be (x, z). "
        "If you wish to indicate a closed contour, the final vertex should be made "
        "equal to the first vertex, i.e., ``vertices[-1] == vertices[0]``",
        units=MICROMETER,
    )

    @staticmethod
    def _compute_dl_component(coord_array: xr.DataArray, closed_contour=False) -> np.ndarray:
        """Computes the differential length element along the integration path."""
        dl = np.gradient(coord_array)
        if closed_contour:
            # If the contour is closed, we can use central difference on the starting/end point
            # which will be more accurate than the default forward/backward choice in np.gradient
            grad_end = np.gradient([coord_array[-2], coord_array[0], coord_array[1]])
            dl[0] = dl[-1] = grad_end[1]
        return dl

    @classmethod
    def from_circular_path(
        cls, center: Coordinate, radius: float, num_points: int, normal_axis: Axis, clockwise: bool
    ) -> Self:
        """Creates a ``Custom2DPathIntegralSpec`` from a circular path given a desired number of points
        along the perimeter.

        Parameters
        ----------
        center : Coordinate
            The center of the circle.
        radius : float
            The radius of the circle.
        num_points : int
            The number of equidistant points to use along the perimeter of the circle.
        normal_axis : Axis
            The axis normal to the defined circle.
        clockwise : bool
            When ``True``, the points will be ordered clockwise with respect to the positive
            direction of the ``normal_axis``.

        Returns
        -------
        :class:`.Custom2DPathIntegralSpec`
            A path integral defined on a circular path.
        """

        def generate_circle_coordinates(radius: float, num_points: int, clockwise: bool):
            """Helper for generating x,y vertices around a circle in the local coordinate frame."""
            sign = 1.0
            if clockwise:
                sign = -1.0
            angles = np.linspace(0, sign * 2 * np.pi, num_points, endpoint=True)
            xt = radius * np.cos(angles)
            yt = radius * np.sin(angles)
            return (xt, yt)

        # Get transverse axes
        normal_center, trans_center = Geometry.pop_axis(center, normal_axis)

        # These x,y coordinates in the local coordinate frame
        if normal_axis == 1:
            # Handle special case when y is the axis that is popped
            clockwise = not clockwise
        xt, yt = generate_circle_coordinates(radius, num_points, clockwise)
        xt += trans_center[0]
        yt += trans_center[1]
        circle_vertices = np.column_stack((xt, yt))
        # Close the contour exactly
        circle_vertices[-1, :] = circle_vertices[0, :]
        return cls(axis=normal_axis, position=normal_center, vertices=circle_vertices)

    @cached_property
    def is_closed_contour(self) -> bool:
        """Returns ``true`` when the first vertex equals the last vertex."""
        return np.isclose(
            self.vertices[0, :],
            self.vertices[-1, :],
            rtol=fp_eps,
            atol=np.finfo(np.float32).smallest_normal,
        ).all()

    @cached_property
    def main_axis(self) -> Axis:
        """Axis for performing integration."""
        return self.axis

    @pd.validator("vertices", always=True)
    def _correct_shape(cls, val):
        """Makes sure vertices size is correct."""
        # overall shape of vertices
        if val.shape[1] != 2:
            raise SetupError(
                "'Custom2DPathIntegralSpec.vertices' must be a 2 dimensional array shaped (N, 2). "
                f"Given array with shape of '{val.shape}'."
            )
        return val

    @cached_property
    def bounds(self) -> Bound:
        """Helper to get the geometric bounding box of the path integral."""
        path_min = np.amin(self.vertices, axis=0)
        path_max = np.amax(self.vertices, axis=0)
        min_bound = Geometry.unpop_axis(self.position, path_min, self.axis)
        max_bound = Geometry.unpop_axis(self.position, path_max, self.axis)
        return (min_bound, max_bound)

    @cached_property
    def sign(self) -> Direction:
        """Uses the ordering of the vertices to determine the direction of the current flow."""
        linestr = shapely.LineString(coordinates=self.vertices)
        is_ccw = shapely.is_ccw(linestr)
        # Invert statement when the vertices are given as (x, z)
        if self.axis == 1:
            is_ccw = not is_ccw
        if is_ccw:
            return "+"
        else:
            return "-"
