"""Dealing with time specifications for DeviceSimulation"""

from abc import ABC
from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import KELVIN, SECOND, VOLT


class TimeBaseSpec(Tidy3dBaseModel, ABC):
    """Base class to deal with time specifications for DeviceSimulations"""


class SteadySpec(TimeBaseSpec):
    """Defines a steady-state specification

    Example
    --------
    >>> import tidy3d as td
    >>> time_spec = td.SteadySpec()
    """


class UnsteadySpec(TimeBaseSpec):
    """Defines an unsteady specification

    Example
    --------
    >>> import tidy3d as td
    >>> time_spec = td.UnsteadySpec(
    ...     time_step=0.01,
    ...     total_time_steps=200,
    ...     initial_value=300,
    ...     output_fr=50,
    ... )
    """

    time_step: pd.PositiveFloat = pd.Field(
        title="Time-step",
        description="Time step taken for each iteration of the time integration loop.",
        units=SECOND,
    )

    total_time_steps: pd.PositiveInt = pd.Field(
        title="Total time steps",
        description="Specifies the total number of time steps run during the simulation.",
    )

    initial_temperature: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Initial temperature.",
        description="Initial value for the temperature field.",
        units=KELVIN,
    )

    initial_voltage: Optional[float] = pd.Field(
        None,
        title="Initial voltage.",
        description="Initial value for the voltage field.",
        units=VOLT,
    )

    output_fr: Optional[pd.PositiveInt] = pd.Field(
        default=1,
        title="Output frequency",
        description="Determines how often output files will be written. I.e., an output "
        "file will be written every 'output_fr' time steps.",
    )


TimeSpecType = Union[SteadySpec, UnsteadySpec]
