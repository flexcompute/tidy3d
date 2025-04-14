"""Helper classes for performing path integrals with fields on the Yee grid"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from tidy3d.components.data.data_array import (
    CurrentIntegralResultTypes,
    IntegralResultTypes,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    VoltageIntegralResultTypes,
    _make_base_result_data_array,
    _make_current_data_array,
    _make_voltage_data_array,
)
from tidy3d.components.data.monitor_data import FieldData, FieldTimeData, ModeData, ModeSolverData
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.microwave.path_integrals.base_spec import AxisAlignedPathIntegralSpec
from tidy3d.components.microwave.path_integrals.current_spec import (
    CurrentIntegralAxisAlignedSpec,
)
from tidy3d.components.microwave.path_integrals.voltage_spec import VoltageIntegralAxisAlignedSpec
from tidy3d.constants import fp_eps
from tidy3d.exceptions import DataError

MonitorDataTypes = Union[FieldData, FieldTimeData, ModeData, ModeSolverData]
EMScalarFieldType = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]


class AxisAlignedPathIntegral(AxisAlignedPathIntegralSpec):
    """Class for defining the simplest type of path integral, which is aligned with Cartesian axes."""

    def compute_integral(self, scalar_field: EMScalarFieldType) -> IntegralResultTypes:
        """Computes the defined integral given the input ``scalar_field``."""

        if not scalar_field.does_cover(self.bounds, fp_eps, np.finfo(np.float32).smallest_normal):
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
        return _make_base_result_data_array(result)

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

    @staticmethod
    def _check_monitor_data_supported(em_field: MonitorDataTypes):
        """Helper for validating that monitor data is supported."""
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeData, ModeSolverData)):
            supported_types = list(MonitorDataTypes.__args__)
            raise DataError(
                f"'em_field' type {type(em_field)} not supported. Supported types are "
                f"{supported_types}"
            )


class VoltageIntegralAxisAligned(AxisAlignedPathIntegral, VoltageIntegralAxisAlignedSpec):
    """Class for computing the voltage between two points defined by an axis-aligned line."""

    def compute_voltage(self, em_field: MonitorDataTypes) -> VoltageIntegralResultTypes:
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

        return _make_voltage_data_array(voltage)

    @staticmethod
    def from_terminal_positions(
        plus_terminal: float,
        minus_terminal: float,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
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


class CurrentIntegralAxisAligned(CurrentIntegralAxisAlignedSpec):
    """Class for computing conduction current via Ampère's circuital law on an axis-aligned loop."""

    def compute_current(self, em_field: MonitorDataTypes) -> CurrentIntegralResultTypes:
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
        return _make_current_data_array(current)

    def _to_path_integrals(
        self, h_horizontal=None, h_vertical=None
    ) -> tuple[AxisAlignedPathIntegral, ...]:
        """Returns four ``AxisAlignedPathIntegral`` instances, which represent a contour
        integral around the surface defined by ``self.size``."""
        path_specs = self._to_path_integral_specs(h_horizontal=h_horizontal, h_vertical=h_vertical)
        path_integrals = tuple(
            AxisAlignedPathIntegral(**path_spec.dict(exclude={"type"})) for path_spec in path_specs
        )
        return path_integrals
