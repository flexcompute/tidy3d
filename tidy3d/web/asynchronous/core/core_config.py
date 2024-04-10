"""Tidy3d core log, need init config from Tidy3d api"""

import logging as log

# default setting
config_setting = {
    "logger": log,
    "logger_console": None,
    "version": "",
}


def set_config(logger, logger_console, version: str):
    """Init tidy3d core logger and logger console.

    Parameters
    ----------
    logger : :class:`.Logger`
        Tidy3d log Logger.
    logger_console : :class:`.Console`
        Get console from logging handlers.
    version : str
        tidy3d version
    """
    config_setting["logger"] = logger
    config_setting["logger_console"] = logger_console
    config_setting["version"] = version


def get_logger():
    """Get logging handlers."""
    return config_setting["logger"]


def get_logger_console():
    """Get console from logging handlers."""
    return config_setting["logger_console"]


def get_version():
    """Get version from cache."""
    return config_setting["version"]
