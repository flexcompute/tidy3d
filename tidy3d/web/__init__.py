""" imports interfaces for interacting with server """
from .api.container import Job, Batch, BatchData
from .cli.migrate import migrate
from .cli.app import configure_fn as configure
from .api.webapi import (
    run,
    upload,
    get_info,
    start,
    monitor,
    delete,
    download,
    load,
    estimate_cost,
    abort,
    get_tasks,
    delete_old,
    download_log,
    download_json,
    load_simulation,
    real_cost,
    test,
)
from .cli import tidy3d_cli
from .api.asynchronous import run_async
from .core import core_config
from ..log import log, get_logging_console
from ..version import __version__

migrate()

core_config.set_config(log, get_logging_console(), __version__)

__all__ = [
    "run",
    "upload",
    "get_info",
    "start",
    "monitor",
    "delete",
    "abort",
    "download",
    "load",
    "estimate_cost",
    "get_tasks",
    "delete_old",
    "download_json",
    "download_log",
    "load_simulation",
    "real_cost",
    "Job",
    "Batch",
    "BatchData",
    "tidy3d_cli",
    "configure",
    "run_async",
    "test",
]
