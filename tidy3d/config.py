"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

import os

import pydantic as pd
from typing_extensions import Literal

from .log import set_logging_level, DEFAULT_LEVEL, Tidy3dKeyError
from .components.base import Tidy3dBaseModel
from .web.config import DEFAULT_CONFIG, WEB_CONFIGS

# set the default web config based on environment variable, if present
env_config = os.environ.get("TIDY3D_ENV")
DEFAULT_WEB_CONFIG = "prod" if env_config is None else env_config


class Tidy3dConfig(pd.BaseModel):
    """configuration of tidy3d"""

    class Config:
        """Config of the config."""

        arbitrary_types_allowed = False
        validate_all = True
        extra = "forbid"
        validate_assignment = True
        allow_population_by_field_name = True

    logging_level: Literal["debug", "info", "warning", "error"] = pd.Field(
        DEFAULT_LEVEL.lower(),
        title="Logging Level",
        description="The lowest level of logging output that will be displayed. "
        'Can be "debug", "info", "warning", "error".',
    )

    web_config: Literal["prod", "preprod", "dev", "uat"] = pd.Field(
        DEFAULT_WEB_CONFIG,
        title="Web Configuration",
        description="Default configuration that webapi uses.",
    )

    # default_folder: str = pd.Field(
    #     "default",
    #     title="Default Folder",
    #     description="Default name of folder where tasks go if ``folder_name`` is not supplied.",
    # )

    frozen: bool = pd.Field(
        False, title="Frozen", description="Whether all tidy3d components are immutable."
    )

    @pd.validator("logging_level", always=True)
    def _set_logging_level(cls, val):
        """Set the logging level if logging_level is changed."""
        set_logging_level(val)
        return val

    @pd.validator("web_config", always=True)
    def _set_web_config(cls, val):
        """Set the default web config."""
        if val not in WEB_CONFIGS:
            raise Tidy3dKeyError(
                f"web config '{val}' not found. " f"Must be one of {list(WEB_CONFIGS.keys())}"
            )
        new_config = WEB_CONFIGS[val]
        for key, value in new_config.dict().items():
            setattr(DEFAULT_CONFIG, key, value)

        return val

    @pd.validator("frozen", always=True)
    def _change_mutability(cls, val):
        """Set whether tidy3d compoennts are mutable."""
        Tidy3dBaseModel.__config__.frozen = val
        return val


# instance of the config that can be modified.
config = Tidy3dConfig()
