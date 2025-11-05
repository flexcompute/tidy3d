from __future__ import annotations

from collections.abc import Generator

import pytest
from tidy3d_frontend.tidy3d.web.core.task_core import TaskFactory


@pytest.fixture(autouse=True)
def clear_task_factory_registry() -> Generator[None, None, None]:
    """Ensure TaskFactory registry is empty for each test."""
    TaskFactory.reset()
    TaskFactory.reset()
    yield
    TaskFactory.reset()
