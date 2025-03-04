# ruff: noqa: E402
"""imports interfaces for interacting with server"""

from ..log import get_logging_console, log
from ..version import __version__
from .core import core_config

# set logger to tidy3d.log before it's invoked in other imports
core_config.set_config(log, get_logging_console(), __version__)

# from .api.asynchronous import run_async # NOTE: we use autograd one now (see below)
# autograd compatible wrappers for run and run_async
from .api.autograd.autograd import run, run_async
from .api.container import Batch, BatchData, Job
from .api.webapi import (
    abort,
    account,
    delete,
    delete_old,
    download,
    download_json,
    download_log,
    estimate_cost,
    get_info,
    get_tasks,
    load,
    load_simulation,
    monitor,
    real_cost,
    start,
    test,
    # run, # NOTE: use autograd one now (see below)
    upload,
)
from .cli import tidy3d_cli
from .cli.app import configure_fn as configure
from .cli.migrate import migrate

migrate()

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
    "account",
]
