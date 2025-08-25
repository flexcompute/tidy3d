"""Dealing with time specifications for DeviceSimulation"""

from __future__ import annotations

import pydantic.v1 as pd

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

    time_step: pd.PositiveFloat = pd.Field(
        ...,
        title="Time-step",
        description="Time step taken for each iteration of the time integration loop.",
        units=SECOND,
    )

    total_time_steps: pd.PositiveInt = pd.Field(
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

    initial_temperature: pd.PositiveFloat = pd.Field(
        ...,
        title="Initial temperature.",
        description="Initial value for the temperature field.",
        units=KELVIN,
    )

    unsteady_spec: UnsteadySpec = pd.Field(
        ...,
        title="Unsteady specification",
        description="Time step and total time steps for the unsteady simulation.",
    )
