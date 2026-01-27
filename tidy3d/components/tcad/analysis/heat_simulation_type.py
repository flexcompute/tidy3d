"""Dealing with time specifications for DeviceSimulation"""

from __future__ import annotations

from pydantic import Field, PositiveFloat, PositiveInt

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import KELVIN, SECOND


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
    """

    initial_temperature: PositiveFloat = Field(
        ...,
        title="Initial temperature.",
        description="Initial value for the temperature field.",
        json_schema_extra={"units": KELVIN},
    )

    unsteady_spec: UnsteadySpec = Field(
        ...,
        title="Unsteady specification",
        description="Time step and total time steps for the unsteady simulation.",
    )
