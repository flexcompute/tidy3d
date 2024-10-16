import numpy as np
import pytest
import tidy3d as td


class CaptureHandler:
    """Log handler used to store log records during tests."""

    def __init__(self):
        self.level = 0
        self.records = []

    def handle(self, level, level_name, message):
        self.records.append((level, message))


@pytest.fixture
def log_capture(monkeypatch):
    """Captures log records and makes them available as a list of tuples with
    the log level and message.
    """
    log_capture = CaptureHandler()
    monkeypatch.setitem(td.log.handlers, "pytest_capture", log_capture)
    return log_capture.records


@pytest.fixture
def rng():
    seed = 36523525
    return np.random.default_rng(seed)
