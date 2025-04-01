"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .log import DEFAULT_LEVEL, LogLevel, set_log_suppression, set_logging_level


class Tidy3dConfig(BaseModel):
    """configuration of tidy3d"""

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        validate_default=True,
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        frozen=False,
    )

    logging_level: LogLevel = Field(
        DEFAULT_LEVEL,
        title="Logging Level",
        description="The lowest level of logging output that will be displayed. "
        'Can be "DEBUG", "SUPPORT", "USER", INFO", "WARNING", "ERROR", or "CRITICAL". '
        'Note: "SUPPORT" and "USER" levels are only used in backend solver logging.',
    )

    log_suppression: bool = Field(
        True,
        title="Log suppression",
        description="Enable or disable suppression of certain log messages when they are repeated "
        "for several elements.",
    )

    use_local_subpixel: Optional[bool] = Field(
        None,
        title="Whether to use local subpixel averaging. If 'None', local subpixel "
        "averaging will be used if 'tidy3d-extras' is installed and not used otherwise. "
        "NOTE: This feature is not yet supported.",
    )

    @field_validator("logging_level")
    def _set_logging_level(val):
        """Set the logging level if logging_level is changed."""
        set_logging_level(val)
        return val

    @field_validator("log_suppression")
    def _set_log_suppression(val):
        """Control log suppression when log_suppression is changed."""
        set_log_suppression(val)
        return val


# instance of the config that can be modified.
config = Tidy3dConfig()
