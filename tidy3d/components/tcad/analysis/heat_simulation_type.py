"""Dealing with time specifications for DeviceSimulation"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, PositiveFloat, PositiveInt

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
