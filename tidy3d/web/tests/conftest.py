from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tidy3d_frontend.tidy3d.web.core.task_core import TaskFactory

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def clear_task_factory_registry() -> Generator[None, None, None]:
    """Ensure TaskFactory registry is empty for each test."""
    TaskFactory.reset()
    TaskFactory.reset()
    yield
    TaskFactory.reset()
