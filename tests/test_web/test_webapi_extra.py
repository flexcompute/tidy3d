from __future__ import annotations

from datetime import datetime

import pytest
import responses

from tidy3d.web.api.webapi import delete, get_info, get_tasks, real_cost, start


@responses.activate
def test_get_info_not_found(monkeypatch):
    """Tests that get_info raises a ValueError when the task is not found."""
    monkeypatch.setattr(
        "tidy3d.web.core.task_core.SimulationTask.get", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="Task not found."):
        get_info("non_existent_task_id")


@responses.activate
def test_start_not_found(monkeypatch):
    """Tests that start raises a ValueError when the task is not found."""
    monkeypatch.setattr(
        "tidy3d.web.core.task_core.SimulationTask.get", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="Task not found."):
        start("non_existent_task_id")


def test_delete_not_found():
    """Tests that delete raises a ValueError when the task id is not found."""
    with pytest.raises(ValueError, match="Task id not found."):
        delete("")


@responses.activate
def test_get_tasks_empty(monkeypatch):
    """Tests that get_tasks returns an empty list when there are no tasks."""

    class MockFolder:
        def list_tasks(self):
            return []

    monkeypatch.setattr(
        "tidy3d.web.core.task_core.Folder.get", lambda *args, **kwargs: MockFolder()
    )
    assert get_tasks() == []


@responses.activate
def test_get_tasks_order_old(monkeypatch):
    """Tests that get_tasks returns tasks in old order."""

    class MockTask:
        def __init__(self, created_at, task_id):
            self.created_at = created_at
            self.task_id = task_id

        def dict(self):
            return {"task_id": self.task_id, "created_at": self.created_at}

        def model_dump(self):
            return self.dict()

    class MockFolder:
        def list_tasks(self):
            return [
                MockTask(datetime(2023, 1, 2), "2"),
                MockTask(datetime(2023, 1, 1), "1"),
                MockTask(datetime(2023, 1, 3), "3"),
            ]

    monkeypatch.setattr(
        "tidy3d.web.core.task_core.Folder.get", lambda *args, **kwargs: MockFolder()
    )
    tasks = get_tasks(order="old")
    assert [t["task_id"] for t in tasks] == ["1", "2", "3"]


@responses.activate
def test_real_cost_no_flex_unit(monkeypatch):
    """Tests that real_cost returns None when flex_unit is not available."""

    class MockTaskInfo:
        realFlexUnit = None
        oriRealFlexUnit = None
        taskType = "FDTD"

    monkeypatch.setattr("tidy3d.web.api.webapi.get_info", lambda *args, **kwargs: MockTaskInfo())
    assert real_cost("task_id") is None
