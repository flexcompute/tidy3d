"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

import pydantic.v1 as pd

from .log import DEFAULT_LEVEL, LogLevel, set_log_suppression, set_logging_level


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

    logging_level: LogLevel = pd.Field(
        DEFAULT_LEVEL,
        title="Logging Level",
        description="The lowest level of logging output that will be displayed. "
        'Can be "DEBUG", "SUPPORT", "USER", INFO", "WARNING", "ERROR", or "CRITICAL". '
        'Note: "SUPPORT" and "USER" levels are only used in backend solver logging.',
    )

    log_suppression: bool = pd.Field(
        True,
        title="Log suppression",
        description="Enable or disable suppression of certain log messages when they are repeated "
        "for several elements.",
    )

    @pd.validator("logging_level", pre=True, always=True)
    def _set_logging_level(cls, val):
        """Set the logging level if logging_level is changed."""
        set_logging_level(val)
        return val

    @pd.validator("log_suppression", pre=True, always=True)
    def _set_log_suppression(cls, val):
        """Control log suppression when log_suppression is changed."""
        set_log_suppression(val)
        return val


# instance of the config that can be modified.
config = Tidy3dConfig()
