"""Helper classes for performing path integrals with fields on the Yee grid"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pydantic.v1 as pd
import shapely as shapely
import xarray as xr

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.data.data_array import (
    FreqDataArray,
    FreqModeDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    TimeDataArray,
)
from ...components.data.monitor_data import FieldData, FieldTimeData, ModeData, ModeSolverData
from ...components.geometry.base import Box, Geometry
from ...components.types import Ax, Axis, Coordinate2D, Direction
from ...components.validators import assert_line, assert_plane
from ...components.viz import add_ax_if_none
from ...constants import AMP, VOLT, fp_eps
from ...exceptions import DataError, Tidy3dError
from ...log import log
from .viz import (
    ARROW_CURRENT,
    plot_params_current_path,
    plot_params_voltage_minus,
    plot_params_voltage_path,
    plot_params_voltage_plus,
)

MonitorDataTypes = Union[FieldData, FieldTimeData, ModeData, ModeSolverData]
EMScalarFieldType = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]
IntegralResultTypes = Union[FreqDataArray, FreqModeDataArray, TimeDataArray]


class AbstractAxesRH(Tidy3dBaseModel, ABC):
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

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


class AxisAlignedPathIntegral(AbstractAxesRH, Box):
    """Class for defining the simplest type of path integral, which is aligned with Cartesian axes."""

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
        description="It might be desireable to integrate exactly along the Yee grid associated with "
        "a field. When this field is ``True``, the integration path will be snapped to the grid.",
    )

    def compute_integral(self, scalar_field: EMScalarFieldType) -> IntegralResultTypes:
        """Computes the defined integral given the input ``scalar_field``."""

        if not scalar_field.does_cover(self.bounds):
            raise DataError("Scalar field does not cover the integration domain.")
        coord = "xyz"[self.main_axis]

        scalar_field = self._get_field_along_path(scalar_field)
        # Get the boundaries
        min_bound = self.bounds[0][self.main_axis]
        max_bound = self.bounds[1][self.main_axis]

        if self.extrapolate_to_endpoints:
            # Remove field outside the boundaries
            scalar_field = scalar_field.sel({coord: slice(min_bound, max_bound)})
            # Ignore values on the boundary (sel is inclusive)
            scalar_field = scalar_field.drop_sel({coord: (min_bound, max_bound)}, errors="ignore")
            coordinates = scalar_field.coords[coord].values
        else:
            coordinates = scalar_field.coords[coord].sel({coord: slice(min_bound, max_bound)})

        # Integration is along the original coordinates plus ensure that
        # endpoints corresponding to the precise bounds of the port are included
        coords_interp = np.array([min_bound])
        coords_interp = np.concatenate((coords_interp, coordinates))
        coords_interp = np.concatenate((coords_interp, [max_bound]))
        coords_interp = {coord: coords_interp}

        # Use extrapolation for the 2 additional endpoints, unless there is only a single sample point
        method = "linear"
        if len(coordinates) == 1 and self.extrapolate_to_endpoints:
            method = "nearest"
        scalar_field = scalar_field.interp(
            coords_interp, method=method, kwargs={"fill_value": "extrapolate"}
        )
        result = scalar_field.integrate(coord=coord)
        return self._make_result_data_array(result)

    def _get_field_along_path(self, scalar_field: EMScalarFieldType) -> EMScalarFieldType:
        """Returns a selection of the input ``scalar_field`` ready for integration."""
        (axis1, axis2) = self.remaining_axes
        (coord1, coord2) = self.remaining_dims

        if self.snap_path_to_grid:
            # Coordinates that are not integrated
            remaining_coords = {
                coord1: self.center[axis1],
                coord2: self.center[axis2],
            }
            # Select field nearest to center of integration line
            scalar_field = scalar_field.sel(
                remaining_coords,
                method="nearest",
                drop=False,
            )
        else:
            # Try to interpolate unless there is only a single coordinate
            coord1dict = {coord1: self.center[axis1]}
            if scalar_field.sizes[coord1] == 1:
                scalar_field = scalar_field.sel(coord1dict, method="nearest")
            else:
                scalar_field = scalar_field.interp(
                    coord1dict, method="linear", kwargs={"bounds_error": True}
                )
            coord2dict = {coord2: self.center[axis2]}
            if scalar_field.sizes[coord2] == 1:
                scalar_field = scalar_field.sel(coord2dict, method="nearest")
            else:
                scalar_field = scalar_field.interp(
                    coord2dict, method="linear", kwargs={"bounds_error": True}
                )
        # Remove unneeded coordinates
        scalar_field = scalar_field.reset_coords(drop=True)
        return scalar_field

    @cached_property
    def main_axis(self) -> Axis:
        """Axis for performing integration."""
        for index, value in enumerate(self.size):
            if value != 0:
                return index
        raise Tidy3dError("Failed to identify axis.")

    def _vertices_2D(self, axis: Axis) -> tuple[Coordinate2D, Coordinate2D]:
        """Returns the two vertices of this path in the plane defined by ``axis``."""
        min = self.bounds[0]
        max = self.bounds[1]
        _, min = Box.pop_axis(min, axis)
        _, max = Box.pop_axis(max, axis)

        u = [min[0], max[0]]
        v = [min[1], max[1]]
        return (u, v)

    @staticmethod
    def _check_monitor_data_supported(em_field: MonitorDataTypes):
        """Helper for validating that monitor data is supported."""
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeData, ModeSolverData)):
            supported_types = list(MonitorDataTypes.__args__)
            raise DataError(
                f"'em_field' type {type(em_field)} not supported. Supported types are "
                f"{supported_types}"
            )

    @staticmethod
    def _make_result_data_array(result: xr.DataArray) -> IntegralResultTypes:
        """Helper for creating the proper result type."""
        if "t" in result.coords:
            return TimeDataArray(data=result.data, coords=result.coords)
        elif "f" in result.coords and "mode_index" in result.coords:
            return FreqModeDataArray(data=result.data, coords=result.coords)
        else:
            return FreqDataArray(data=result.data, coords=result.coords)


class VoltageIntegralAxisAligned(AxisAlignedPathIntegral):
    """Class for computing the voltage between two points defined by an axis-aligned line."""

    sign: Direction = pd.Field(
        ...,
        title="Direction of Path Integral",
        description="Positive indicates V=Vb-Va where position b has a larger coordinate along the axis of integration.",
    )

    def compute_voltage(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute voltage along path defined by a line."""
        self._check_monitor_data_supported(em_field=em_field)
        e_component = "xyz"[self.main_axis]
        field_name = f"E{e_component}"
        # Validate that fields are present
        em_field._check_fields_stored([field_name])
        e_field = em_field.field_components[field_name]

        voltage = self.compute_integral(e_field)

        if self.sign == "+":
            voltage *= -1

        voltage = VoltageIntegralAxisAligned._set_data_array_attributes(voltage)
        # Return data array of voltage while keeping coordinates of frequency|time|mode index
        return voltage

    @staticmethod
    def _set_data_array_attributes(data_array: IntegralResultTypes) -> IntegralResultTypes:
        """Add explanatory attributes to the data array."""
        data_array.name = "V"
        return data_array.assign_attrs(units=VOLT, long_name="voltage")

    @staticmethod
    def from_terminal_positions(
        plus_terminal: float,
        minus_terminal: float,
        x: float = None,
        y: float = None,
        z: float = None,
        extrapolate_to_endpoints: bool = True,
        snap_path_to_grid: bool = True,
    ) -> VoltageIntegralAxisAligned:
        """Helper to create a :class:`VoltageIntegralAxisAligned` from two coordinates that
        define a line and two positions indicating the endpoints of the path integral.

        Parameters
        ----------
        plus_terminal : float
            Position along the voltage axis of the positive terminal.
        minus_terminal : float
            Position along the voltage axis of the negative terminal.
        x : float = None
            Position in x direction, only two of x,y,z can be specified to define line.
        y : float = None
            Position in y direction, only two of x,y,z can be specified to define line.
        z : float = None
            Position in z direction, only two of x,y,z can be specified to define line.
        extrapolate_to_endpoints: bool = True
            Passed directly to :class:`VoltageIntegralAxisAligned`
        snap_path_to_grid: bool = True
            Passed directly to :class:`VoltageIntegralAxisAligned`

        Returns
        -------
        VoltageIntegralAxisAligned
            The created path integral for computing voltage between the two terminals.
        """
        axis_positions = Geometry.parse_two_xyz_kwargs(x=x, y=y, z=z)
        # Calculate center and size of the future box
        midpoint = (plus_terminal + minus_terminal) / 2
        length = np.abs(plus_terminal - minus_terminal)
        center = [midpoint, midpoint, midpoint]
        size = [length, length, length]
        for axis, position in axis_positions:
            size[axis] = 0
            center[axis] = position

        direction = "+"
        if plus_terminal < minus_terminal:
            direction = "-"

        return VoltageIntegralAxisAligned(
            center=center,
            size=size,
            extrapolate_to_endpoints=extrapolate_to_endpoints,
            snap_path_to_grid=snap_path_to_grid,
            sign=direction,
        )

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
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis == self.main_axis or not np.isclose(position, self.center[axis], rtol=fp_eps):
            return ax

        (xs, ys) = self._vertices_2D(axis)
        # Plot the path
        plot_params = plot_params_voltage_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        ax.plot(xs, ys, markevery=[0, -1], **plot_kwargs)

        # Plot special end points
        end_kwargs = plot_params_voltage_plus.include_kwargs(**path_kwargs).to_kwargs()
        start_kwargs = plot_params_voltage_minus.include_kwargs(**path_kwargs).to_kwargs()

        if self.sign == "-":
            start_kwargs, end_kwargs = end_kwargs, start_kwargs

        ax.plot(xs[0], ys[0], **start_kwargs)
        ax.plot(xs[1], ys[1], **end_kwargs)
        return ax


class CurrentIntegralAxisAligned(AbstractAxesRH, Box):
    """Class for computing conduction current via Ampère's circuital law on an axis-aligned loop."""

    _plane_validator = assert_plane()

    sign: Direction = pd.Field(
        ...,
        title="Direction of Contour Integral",
        description="Positive indicates current flowing in the positive normal axis direction.",
    )

    extrapolate_to_endpoints: bool = pd.Field(
        False,
        title="Extrapolate to Endpoints",
        description="This parameter is passed to :class:`AxisAlignedPathIntegral` objects when computing the contour integral.",
    )

    snap_contour_to_grid: bool = pd.Field(
        False,
        title="Snap Contour to Grid",
        description="This parameter is passed to :class:`AxisAlignedPathIntegral` objects when computing the contour integral.",
    )

    def compute_current(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute current flowing in loop defined by the outer edge of a rectangle."""
        AxisAlignedPathIntegral._check_monitor_data_supported(em_field=em_field)
        ax1 = self.remaining_axes[0]
        ax2 = self.remaining_axes[1]
        h_component = "xyz"[ax1]
        v_component = "xyz"[ax2]
        h_field_name = f"H{h_component}"
        v_field_name = f"H{v_component}"
        # Validate that fields are present
        em_field._check_fields_stored([h_field_name, v_field_name])
        h_horizontal = em_field.field_components[h_field_name]
        h_vertical = em_field.field_components[v_field_name]

        # Decompose contour into path integrals
        (bottom, right, top, left) = self._to_path_integrals(h_horizontal, h_vertical)

        current = 0
        # Compute and add contributions from each part of the contour
        current += bottom.compute_integral(h_horizontal)
        current += right.compute_integral(h_vertical)
        current -= top.compute_integral(h_horizontal)
        current -= left.compute_integral(h_vertical)

        if self.sign == "-":
            current *= -1
        current = CurrentIntegralAxisAligned._set_data_array_attributes(current)
        return current

    @cached_property
    def main_axis(self) -> Axis:
        """Axis normal to loop"""
        for index, value in enumerate(self.size):
            if value == 0:
                return index
        raise Tidy3dError("Failed to identify axis.")

    def _to_path_integrals(
        self, h_horizontal=None, h_vertical=None
    ) -> tuple[AxisAlignedPathIntegral, ...]:
        """Returns four ``AxisAlignedPathIntegral`` instances, which represent a contour
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

        bottom = AxisAlignedPathIntegral(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )
        path_center[ax2] = top_bound
        top = AxisAlignedPathIntegral(
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
        left = AxisAlignedPathIntegral(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )
        path_center[ax1] = right_bound
        right = AxisAlignedPathIntegral(
            center=path_center,
            size=path_size,
            extrapolate_to_endpoints=self.extrapolate_to_endpoints,
            snap_path_to_grid=self.snap_contour_to_grid,
        )

        return (bottom, right, top, left)

    @staticmethod
    def _set_data_array_attributes(data_array: IntegralResultTypes) -> IntegralResultTypes:
        """Add explanatory attributes to the data array."""
        data_array.name = "I"
        return data_array.assign_attrs(units=AMP, long_name="current")

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
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis != self.main_axis or not np.isclose(position, self.center[axis], rtol=fp_eps):
            return ax

        plot_params = plot_params_current_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        path_integrals = self._to_path_integrals()
        # Plot the path
        for path in path_integrals:
            (xs, ys) = path._vertices_2D(axis)
            ax.plot(xs, ys, **plot_kwargs)

        (ax1, ax2) = self.remaining_axes

        # Add arrow to bottom path, unless right path is longer
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
