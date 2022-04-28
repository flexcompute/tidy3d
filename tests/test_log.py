import pytest
import tidy3d as td
from tidy3d.log import Tidy3dError

log = td.log


def test_log():
    log.debug("test")
    log.info("test")
    log.warning("test")
    log.error("test")


def _test_log_config():
    """these are deprecated."""
    td.set_logging_level("debug")
    td.set_logging_file("test.log")
