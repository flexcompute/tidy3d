"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

import pydantic as pd
from typing_extensions import Literal

from .log import set_logging_level, DEFAULT_LEVEL


class Tidy3dConfig(pd.BaseModel):
    """configuration of tidy3d"""

    class Config:
        """Config of the config."""

        arbitrary_types_allowed = False
        validate_all = True
        extra = "forbid"
        validate_assignment = True
        allow_population_by_field_name = True
        frozen = False

    logging_level: Literal["debug", "info", "warning", "error"] = pd.Field(
        DEFAULT_LEVEL.lower(),
        title="Logging Level",
        description="The lowest level of logging output that will be displayed. "
        'Can be "debug", "info", "warning", "error".',
    )

    @pd.validator("logging_level", always=True)
    def _set_logging_level(cls, val):
        """Set the logging level if logging_level is changed."""
        set_logging_level(val)
        return val


# instance of the config that can be modified.
config = Tidy3dConfig()
