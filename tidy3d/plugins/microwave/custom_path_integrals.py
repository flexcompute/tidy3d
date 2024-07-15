"""Helper classes for performing custom path integrals with fields on the Yee grid"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ...components.base import cached_property
from ...components.data.data_array import FreqDataArray, FreqModeDataArray, TimeDataArray
from ...components.data.monitor_data import FieldData, FieldTimeData, ModeSolverData
from ...components.types import ArrayFloat2D, Axis
from ...constants import MICROMETER, fp_eps
from ...exceptions import DataError, SetupError
from .path_integrals import AbstractAxesRH, IntegralResultTypes, MonitorDataTypes

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
        if not isinstance(em_field, (FieldData, FieldTimeData, ModeSolverData)):
            raise DataError("'em_field' type not supported.")

        (dim1, dim2, dim3) = self.local_dims

        h_field_name = f"{field}{dim1}"
        v_field_name = f"{field}{dim2}"
        # Validate that fields are present
        if h_field_name not in em_field.field_components:
            raise DataError(f"'field_name' '{h_field_name}' not found.")
        if v_field_name not in em_field.field_components:
            raise DataError(f"'field_name' '{v_field_name}' not found.")

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

        if isinstance(em_field, FieldData):
            return FreqDataArray(data=result.data, coords=result.coords)
        elif isinstance(em_field, FieldTimeData):
            return TimeDataArray(data=result.data, coords=result.coords)
        else:
            if not isinstance(em_field, ModeSolverData):
                raise TypeError(
                    f"Unsupported 'em_field' type: {type(em_field)}. "
                    "Expected one of 'FieldData', 'FieldTimeData', 'ModeSolverData'."
                )
            return FreqModeDataArray(data=result.data, coords=result.coords)

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

    @cached_property
    def is_closed_contour(self) -> bool:
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
        voltage = -1.0 * self.compute_integral(field="E", em_field=em_field)
        return voltage


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
        current = self.compute_integral(field="H", em_field=em_field)
        return current
