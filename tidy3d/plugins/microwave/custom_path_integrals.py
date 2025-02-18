"""Helper classes for performing custom path integrals with fields on the Yee grid"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pydantic.v1 as pd
import shapely
import xarray as xr

from ...components.base import cached_property
from ...components.geometry.base import Geometry
from ...components.types import ArrayFloat2D, Ax, Axis, Bound, Coordinate, Direction
from ...components.viz import add_ax_if_none
from ...constants import MICROMETER, fp_eps
from ...exceptions import SetupError
from .path_integrals import (
    AbstractAxesRH,
    AxisAlignedPathIntegral,
    CurrentIntegralAxisAligned,
    IntegralResultTypes,
    MonitorDataTypes,
    VoltageIntegralAxisAligned,
)
from .viz import (
    ARROW_CURRENT,
    plot_params_current_path,
    plot_params_voltage_minus,
    plot_params_voltage_path,
    plot_params_voltage_plus,
)

FieldParameter = Literal["E", "H"]


class CustomPathIntegral2D(AbstractAxesRH):
    """Class for defining a custom path integral defined as a curve on an axis-aligned plane.

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
        2, title="Axis", description="Specifies dimension of the planar axis (0,1,2) -> (x,y,z)."
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

    def compute_integral(
        self, field: FieldParameter, em_field: MonitorDataTypes
    ) -> IntegralResultTypes:
        """Computes the path integral defined by ``vertices`` given the input ``em_field``.

        Parameters
        ----------
        field : :class:`.FieldParameter`
            Can take the value of ``"E"`` or ``"H"``. Determines whether to perform the integral
            over electric or magnetic field.
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.IntegralResultTypes`
            Result of integral over remaining dimensions (frequency, time, mode indices).
        """

        (dim1, dim2, dim3) = self.local_dims

        h_field_name = f"{field}{dim1}"
        v_field_name = f"{field}{dim2}"

        # Validate that fields are present
        em_field._check_fields_stored([h_field_name, v_field_name])

        # Select fields lying on the plane
        plane_indexer = {dim3: self.position}
        field1 = em_field.field_components[h_field_name].sel(plane_indexer, method="nearest")
        field2 = em_field.field_components[v_field_name].sel(plane_indexer, method="nearest")

        # Although for users we use the convention that an axis is simply `popped`
        # internally we prefer a right-handed coordinate system where dimensions
        # keep a proper order. The only change is to swap 'x' and 'z' when the
        # normal axis is along  `y`
        # Dim 's' represents the parameterization of the line
        # 't' is likely used for time
        if self.main_axis == 1:
            x_path = xr.DataArray(self.vertices[:, 1], dims="s")
            y_path = xr.DataArray(self.vertices[:, 0], dims="s")
        else:
            x_path = xr.DataArray(self.vertices[:, 0], dims="s")
            y_path = xr.DataArray(self.vertices[:, 1], dims="s")

        path_indexer = {dim1: x_path, dim2: y_path}
        field1_interp = field1.interp(path_indexer, method="linear")
        field2_interp = field2.interp(path_indexer, method="linear")

        # Determine the differential length elements along the path
        dl_x = self._compute_dl_component(x_path, self.is_closed_contour)
        dl_y = self._compute_dl_component(y_path, self.is_closed_contour)
        dl_x = xr.DataArray(dl_x, dims="s")
        dl_y = xr.DataArray(dl_y, dims="s")

        # Compute the dot product between differential length element and vector field
        integrand = field1_interp * dl_x + field2_interp * dl_y
        # Integrate along the path
        result = integrand.integrate(coord="s")
        result = result.reset_coords(drop=True)
        return AxisAlignedPathIntegral._make_result_data_array(result)

    @staticmethod
    def _compute_dl_component(coord_array: xr.DataArray, closed_contour=False) -> np.array:
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
    ) -> CustomPathIntegral2D:
        """Creates a ``CustomPathIntegral2D`` from a circular path given a desired number of points
        along the perimeter.

        Parameters
        ----------
        center : Coordinate
            The center of the circle.
        radius : float
            The radius of the circle.
        num_points : int
            THe number of equidistant points to use along the perimeter of the circle.
        normal_axis : Axis
            The axis normal to the defined circle.
        clockwise : bool
            When ``True``, the points will be ordered clockwise with respect to the positive
            direction of the ``normal_axis``.

        Returns
        -------
        :class:`.CustomPathIntegral2D`
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
                "'CustomPathIntegral2D.vertices' must be a 2 dimensional array shaped (N, 2). "
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


class CustomVoltageIntegral2D(CustomPathIntegral2D):
    """Class for computing the voltage between two points defined by a custom path.
    Computed voltage is :math:`V=V_b-V_a`, where position b is the final vertex in the supplied path.

    Notes
    -----

    Use :class:`.VoltageIntegralAxisAligned` if possible, since interpolation
    near conductors will not be accurate.

    .. TODO Improve by including extrapolate_to_endpoints field, non-trivial extension."""

    def compute_voltage(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute voltage along path defined by a line.

        Parameters
        ----------
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.IntegralResultTypes`
            Result of voltage computation over remaining dimensions (frequency, time, mode indices).
        """
        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        voltage = -1.0 * self.compute_integral(field="E", em_field=em_field)
        voltage = VoltageIntegralAxisAligned._set_data_array_attributes(voltage)
        return voltage

    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **path_kwargs,
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

        plot_params = plot_params_voltage_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        xs = self.vertices[:, 0]
        ys = self.vertices[:, 1]
        ax.plot(xs, ys, markevery=[0, -1], **plot_kwargs)

        # Plot special end points
        end_kwargs = plot_params_voltage_plus.include_kwargs(**path_kwargs).to_kwargs()
        start_kwargs = plot_params_voltage_minus.include_kwargs(**path_kwargs).to_kwargs()
        ax.plot(xs[0], ys[0], **start_kwargs)
        ax.plot(xs[-1], ys[-1], **end_kwargs)

        return ax


class CustomCurrentIntegral2D(CustomPathIntegral2D):
    """Class for computing conduction current via Ampère's circuital law on a custom path.
    To compute the current flowing in the positive ``axis`` direction, the vertices should be
    ordered in a counterclockwise direction."""

    def compute_current(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute current flowing in a custom loop.

        Parameters
        ----------
        em_field : :class:`.MonitorDataTypes`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.IntegralResultTypes`
            Result of current computation over remaining dimensions (frequency, time, mode indices).
        """
        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        current = self.compute_integral(field="H", em_field=em_field)
        current = CurrentIntegralAxisAligned._set_data_array_attributes(current)
        return current

    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **path_kwargs,
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
        ax.annotate(
            "",
            xytext=(xs[0], ys[0]),
            xy=(xs[1], ys[1]),
            arrowprops=ARROW_CURRENT,
        )
        return ax

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
