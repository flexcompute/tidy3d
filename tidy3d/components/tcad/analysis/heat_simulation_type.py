"""Dealing with time specifications for DeviceSimulation"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.constants import KELVIN, SECOND

# 'PositiveFloat' admits infinity, which reaches the solver as the literal expression "inf".
FinitePositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class UnsteadySpec(Tidy3dBaseModel):
    """Defines an unsteady specification

    Example
    --------
    >>> import tidy3d as td
    >>> time_spec = td.UnsteadySpec(
    ...     time_step=0.01,
    ...     total_time_steps=200,
    ... )
    """

    time_step: PositiveFloat = Field(
        ...,
        title="Time-step",
        description="Time step taken for each iteration of the time integration loop.",
        json_schema_extra={"units": SECOND},
    )

    total_time_steps: PositiveInt = Field(
        ...,
        title="Total time steps",
        description="Specifies the total number of time steps run during the simulation.",
    )


class UnsteadyHeatAnalysis(Tidy3dBaseModel):
    """
    Configures relevant unsteady-state heat simulation parameters.

    Notes
    -----
        The initial temperature can be uniform (a scalar) or spatially varying (a
        :class:`.SpatialDataArray`, e.g. the result of a previous steady or unsteady heat
        simulation converted with ``to_spatial_data_array``). A spatial field is sampled at
        every mesh node inside its bounding box; nodes outside the box start at
        ``background_temperature``.

    Example
    -------
    >>> import tidy3d as td
    >>> time_spec = td.UnsteadyHeatAnalysis(
    ...     initial_temperature=300,
    ...     unsteady_spec=td.UnsteadySpec(
    ...         time_step=0.01,
    ...         total_time_steps=200,
    ...     ),
    ... )

    A cooling analysis starting from a heater's steady-state profile:

    >>> import numpy as np
    >>> T0 = td.SpatialDataArray(
    ...     350 + np.zeros((2, 2, 2)), coords=dict(x=[0, 1], y=[0, 1], z=[0, 1])
    ... )
    >>> cooling = td.UnsteadyHeatAnalysis(
    ...     initial_temperature=T0,
    ...     background_temperature=300,
    ...     unsteady_spec=td.UnsteadySpec(time_step=0.01, total_time_steps=200),
    ... )
    """

    initial_temperature: FinitePositiveFloat | SpatialDataArray = Field(
        ...,
        title="Initial temperature.",
        description="Initial value for the temperature field. Either a uniform scalar or a "
        "spatially varying :class:`.SpatialDataArray`. A field is interpolated at every mesh "
        "node within its bounding box; nodes outside it take ``background_temperature``. An "
        "axis holding a single coordinate is invariant rather than zero-thickness: the field "
        "is extruded along it. That is only meaningful along the zero-size dimension of a 2D "
        "simulation, whose mesh is one cell thick there, so it is rejected on any other axis.",
        json_schema_extra={"units": KELVIN},
    )

    background_temperature: FinitePositiveFloat = Field(
        default=300,
        title="Background temperature.",
        description="Initial temperature of mesh nodes outside the bounding box of a spatially "
        "varying ``initial_temperature``. Ignored when ``initial_temperature`` is a scalar.",
        json_schema_extra={"units": KELVIN},
    )

    unsteady_spec: UnsteadySpec = Field(
        ...,
        title="Unsteady specification",
        description="Time step and total time steps for the unsteady simulation.",
    )

    @model_validator(mode="after")
    def _check_initial_temperature_field_is_usable(self) -> UnsteadyHeatAnalysis:
        """A spatial initial temperature must be non-empty, finite, positive and on ascending
        finite axes.

        Same contract as the prescribed lattice temperature of
        :class:`.SteadyChargeDCAnalysis`: the solver samples the field by bisecting each
        axis and clamping against its ends, so a descending axis would resolve every node to
        one plane. Non-positive values are rejected here because an absolute temperature of
        0 K would only surface as a nonsensical solve rather than an error.

        The empty and non-finite-coordinate cases are checked first because the value checks
        below pass vacuously on an empty array and every comparison against NaN is false.
        """
        temperature = self.initial_temperature
        # A scalar is already constrained by 'FinitePositiveFloat'.
        if not isinstance(temperature, SpatialDataArray):
            return self

        values = np.asarray(temperature.values)
        if values.size == 0:
            self._raise_validation_error_at_loc(
                "'initial_temperature' is empty: it holds no values to interpolate from. A "
                "'sel' or 'interp' that matched nothing along an axis is the usual cause.",
                "initial_temperature",
            )
        if np.iscomplexobj(values):
            self._raise_validation_error_at_loc(
                "'initial_temperature' contains complex values. An initial temperature field "
                "must be real.",
                "initial_temperature",
            )
        if not np.all(np.isfinite(values)):
            self._raise_validation_error_at_loc(
                "'initial_temperature' contains non-finite values (NaN or infinity). An "
                "initial temperature field must be finite everywhere.",
                "initial_temperature",
            )
        if np.any(values <= 0):
            hint = ""
            if np.isclose(values.min(), 0.0):
                # 'to_spatial_data_array' defaults to fill_value=0.0: right for a source
                # term, 0 K for a temperature.
                hint = (
                    " If this field came from 'to_spatial_data_array', pass a physical "
                    "'fill_value' (e.g. the background temperature): its default of 0.0 is "
                    "meant for source terms, not temperatures."
                )
            self._raise_validation_error_at_loc(
                f"'initial_temperature' must be strictly positive everywhere, but its minimum "
                f"is {values.min():.4g} K. An initial temperature is an absolute temperature "
                f"in Kelvin.{hint}",
                "initial_temperature",
            )

        for axis in "xyz":
            coords = np.atleast_1d(np.asarray(temperature.coords[axis].values))
            if not np.all(np.isfinite(coords)):
                self._raise_validation_error_at_loc(
                    f"'initial_temperature' has a non-finite '{axis}' coordinate (NaN or "
                    f"infinity). Mesh nodes are placed on each axis by comparing against its "
                    f"coordinates, which no comparison against NaN can do.",
                    "initial_temperature",
                )
            if coords.size > 1 and np.any(coords[1:] <= coords[:-1]):
                self._raise_validation_error_at_loc(
                    f"'initial_temperature' has a '{axis}' coordinate that is not strictly "
                    f"ascending. Sort the field along every axis before using it, "
                    f"e.g. 'T.sortby(['x', 'y', 'z'])'.",
                    "initial_temperature",
                )
        return self
