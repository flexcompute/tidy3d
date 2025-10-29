"""Base classes for performing path integrals with fields on the Yee grid"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import xarray as xr

from tidy3d.components.data.data_array import (
    IntegralResultType,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    _make_base_result_data_array,
)
from tidy3d.components.data.monitor_data import FieldData, FieldTimeData, ModeData, ModeSolverData
from tidy3d.components.microwave.path_integrals.specs.base import (
    AxisAlignedPathIntegralSpec,
    Custom2DPathIntegralSpec,
)
from tidy3d.constants import fp_eps
from tidy3d.exceptions import DataError

IntegrableMonitorDataType = Union[FieldData, FieldTimeData, ModeData, ModeSolverData]
EMScalarFieldType = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]
FieldParameter = Literal["E", "H"]


class AxisAlignedPathIntegral(AxisAlignedPathIntegralSpec):
    """Class for defining the simplest type of path integral, which is aligned with Cartesian axes.

    Example
    -------
    >>> path = AxisAlignedPathIntegral(
    ...     center=(0, 0, 1),
    ...     size=(0, 0, 2),
    ...     extrapolate_to_endpoints=True,
    ...     snap_path_to_grid=False,
    ... )
    """

    def compute_integral(self, scalar_field: EMScalarFieldType) -> IntegralResultType:
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
    def _check_monitor_data_supported(em_field: IntegrableMonitorDataType) -> None:
        """Helper for validating that monitor data is supported."""
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeData, ModeSolverData)):
            supported_types = list(IntegrableMonitorDataType.__args__)
            raise DataError(
                f"'em_field' type {type(em_field)} not supported. Supported types are "
                f"{supported_types}"
            )


class Custom2DPathIntegral(Custom2DPathIntegralSpec):
    """Class for defining a custom path integral defined as a curve on an axis-aligned plane.

    Notes
    -----

    Given a set of vertices :math:`\\vec{r}_i`, this class approximates path integrals over
    vector fields of the form :math:`\\int{\\vec{F} \\cdot \\vec{dl}}`
    as :math:`\\sum_i{\\vec{F}(\\vec{r}_i) \\cdot \\vec{dl}_i}`,
    where the differential length :math:`\\vec{dl}` is approximated using central differences
    :math:`\\vec{dl}_i = \\frac{\\vec{r}_{i+1} - \\vec{r}_{i-1}}{2}`.
    If the path is not closed, forward and backward differences are used at the endpoints.

    Example
    -------
    >>> import numpy as np
    >>> vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> path = Custom2DPathIntegral(
    ...     axis=2,
    ...     position=0.5,
    ...     vertices=vertices,
    ... )
    """

    def compute_integral(
        self, field: FieldParameter, em_field: IntegrableMonitorDataType
    ) -> IntegralResultType:
        """Computes the path integral defined by ``vertices`` given the input ``em_field``.

        Parameters
        ----------
        field : :class:`.FieldParameter`
            Can take the value of ``"E"`` or ``"H"``. Determines whether to perform the integral
            over electric or magnetic field.
        em_field : :class:`.IntegrableMonitorDataType`
            The electromagnetic field data that will be used for integrating.

        Returns
        -------
        :class:`.IntegralResultType`
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
        dl_x = Custom2DPathIntegralSpec._compute_dl_component(x_path, self.is_closed_contour)
        dl_y = Custom2DPathIntegralSpec._compute_dl_component(y_path, self.is_closed_contour)
        dl_x = xr.DataArray(dl_x, dims="s")
        dl_y = xr.DataArray(dl_y, dims="s")

        # Compute the dot product between differential length element and vector field
        integrand = field1_interp * dl_x + field2_interp * dl_y
        # Integrate along the path
        result = integrand.integrate(coord="s")
        result = result.reset_coords(drop=True)
        return _make_base_result_data_array(result)
