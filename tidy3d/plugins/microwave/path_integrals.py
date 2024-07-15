"""Helper classes for performing path integrals with fields on the Yee grid"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.data.data_array import (
    FreqDataArray,
    FreqModeDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    TimeDataArray,
)
from ...components.data.monitor_data import FieldData, FieldTimeData, ModeSolverData
from ...components.geometry.base import Box
from ...components.types import Axis, Direction
from ...components.validators import assert_line, assert_plane
from ...exceptions import DataError, Tidy3dError

MonitorDataTypes = Union[FieldData, FieldTimeData, ModeSolverData]
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
        if isinstance(scalar_field, ScalarFieldDataArray):
            return FreqDataArray(data=result.data, coords=result.coords)
        elif isinstance(scalar_field, ScalarFieldTimeDataArray):
            return TimeDataArray(data=result.data, coords=result.coords)
        else:
            if not isinstance(scalar_field, ScalarModeFieldDataArray):
                raise TypeError(
                    f"Unsupported 'scalar_field' type: {type(scalar_field)}. "
                    "Expected one of 'ScalarFieldDataArray', 'ScalarFieldTimeDataArray', "
                    "'ScalarModeFieldDataArray'."
                )
            return FreqModeDataArray(data=result.data, coords=result.coords)

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


class VoltageIntegralAxisAligned(AxisAlignedPathIntegral):
    """Class for computing the voltage between two points defined by an axis-aligned line."""

    sign: Direction = pd.Field(
        ...,
        title="Direction of Path Integral",
        description="Positive indicates V=Vb-Va where position b has a larger coordinate along the axis of integration.",
    )

    def compute_voltage(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute voltage along path defined by a line."""
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeSolverData)):
            raise DataError("'em_field' type not supported.")
        e_component = "xyz"[self.main_axis]
        field_name = f"E{e_component}"
        # Validate that the field is present
        if field_name not in em_field.field_components:
            raise DataError(f"'field_name' '{field_name}' not found.")
        e_field = em_field.field_components[field_name]
        # V = -integral(E)
        voltage = self.compute_integral(e_field)

        if self.sign == "+":
            voltage *= -1
        # Return data array of voltage while keeping coordinates of frequency|time|mode index
        return voltage


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
        description="This parameter is passed to ``AxisAlignedPathIntegral`` objects when computing the contour integral.",
    )

    snap_contour_to_grid: bool = pd.Field(
        False,
        title="Snap Contour to Grid",
        description="This parameter is passed to ``AxisAlignedPathIntegral`` objects when computing the contour integral.",
    )

    def compute_current(self, em_field: MonitorDataTypes) -> IntegralResultTypes:
        """Compute current flowing in loop defined by the outer edge of a rectangle."""
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeSolverData)):
            raise DataError("'em_field' type not supported.")
        ax1 = self.remaining_axes[0]
        ax2 = self.remaining_axes[1]
        h_component = "xyz"[ax1]
        v_component = "xyz"[ax2]
        h_field_name = f"H{h_component}"
        v_field_name = f"H{v_component}"
        # Validate that fields are present
        if h_field_name not in em_field.field_components:
            raise DataError(f"'field_name' '{h_field_name}' not found.")
        if v_field_name not in em_field.field_components:
            raise DataError(f"'field_name' '{v_field_name}' not found.")
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

        return current

    @cached_property
    def main_axis(self) -> Axis:
        """Axis normal to loop"""
        for index, value in enumerate(self.size):
            if value == 0:
                return index
        raise Tidy3dError("Failed to identify axis.")

    def _to_path_integrals(self, h_horizontal, h_vertical) -> tuple[AxisAlignedPathIntegral, ...]:
        """Returns four ``AxisAlignedPathIntegral`` instances, which represent a contour
        integral around the surface defined by ``self.size``."""
        ax1 = self.remaining_axes[0]
        ax2 = self.remaining_axes[1]

        if self.snap_contour_to_grid:
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
