"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

import pydantic as pd

from .log import DEFAULT_LEVEL, LogLevel, set_logging_level, log


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
        'Can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".',
    )

    @pd.validator("logging_level", pre=True, always=True)
    def _set_logging_level(cls, val):
        """Set the logging level if logging_level is changed."""
        val_upper = val.upper()
        if val_upper != val:
            log.warning(
                f"'{val}' provided to 'td.config.logging_level'. "
                "In the future, only upper-case logging levels may be specified. "
                f"This value will be converted to upper case '{val_upper}'."
            )
        set_logging_level(val_upper)
        return val_upper


# instance of the config that can be modified.
config = Tidy3dConfig()
