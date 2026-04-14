"""Tests for BatchData in-memory caching."""

from __future__ import annotations

from pathlib import Path

import tidy3d as td
from tidy3d.web.api import container as web_container


def _write_bytes(path: Path, size: int) -> None:
    path.write_bytes(b"0" * size)


def test_batch_data_caches_small_files(monkeypatch, tmp_path):
    task_paths = {
        "task1": str(tmp_path / "task1.hdf5"),
        "task2": str(tmp_path / "task2.hdf5"),
    }
    task_ids = {"task1": "task-1", "task2": "task-2"}
    _write_bytes(Path(task_paths["task1"]), 1)
    _write_bytes(Path(task_paths["task2"]), 2)

    monkeypatch.setattr(td.config.batch_data_cache, "enabled", True)
    monkeypatch.setattr(td.config.batch_data_cache, "max_total_size_gb", 1.0)

    calls = {"load": 0}
    sentinels = [object(), object()]

    def fake_load(*args, **kwargs):
        result = sentinels[calls["load"]]
        calls["load"] += 1
        return result

    monkeypatch.setattr(web_container.web, "load", fake_load)

    batch_data = td.web.BatchData(
        task_paths=task_paths,
        task_ids=task_ids,
        is_downloaded=True,
    )

    first = batch_data["task1"]
    second = batch_data["task1"]

    assert first is second
    assert calls["load"] == 1


def test_batch_data_caches_multiple_tasks_independently(monkeypatch, tmp_path):
    task_paths = {
        "task1": str(tmp_path / "task1.hdf5"),
        "task2": str(tmp_path / "task2.hdf5"),
    }
    task_ids = {"task1": "task-1", "task2": "task-2"}
    _write_bytes(Path(task_paths["task1"]), 1)
    _write_bytes(Path(task_paths["task2"]), 2)

    monkeypatch.setattr(td.config.batch_data_cache, "enabled", True)
    monkeypatch.setattr(td.config.batch_data_cache, "max_total_size_gb", 1.0)

    calls = {"load": 0}
    sentinels = {
        "task1.hdf5": object(),
        "task2.hdf5": object(),
    }

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        return sentinels[Path(kwargs["path"]).name]

    monkeypatch.setattr(web_container.web, "load", fake_load)

    batch_data = td.web.BatchData(
        task_paths=task_paths,
        task_ids=task_ids,
        is_downloaded=True,
    )

    first_task1 = batch_data["task1"]
    first_task2 = batch_data["task2"]
    second_task1 = batch_data["task1"]
    second_task2 = batch_data["task2"]

    assert first_task1 is second_task1
    assert first_task2 is second_task2
    assert calls["load"] == 2


def test_batch_data_skips_cache_when_any_file_is_large(monkeypatch, tmp_path):
    task_paths = {
        "task1": str(tmp_path / "task1.hdf5"),
        "task2": str(tmp_path / "task2.hdf5"),
    }
    task_ids = {"task1": "task-1", "task2": "task-2"}
    _write_bytes(Path(task_paths["task1"]), 1)
    _write_bytes(Path(task_paths["task2"]), 2)

    threshold_gb = 2 / (1024**3)
    monkeypatch.setattr(td.config.batch_data_cache, "enabled", True)
    monkeypatch.setattr(td.config.batch_data_cache, "max_total_size_gb", threshold_gb)

    calls = {"load": 0}
    sentinels = [object(), object()]

    def fake_load(*args, **kwargs):
        result = sentinels[calls["load"]]
        calls["load"] += 1
        return result

    monkeypatch.setattr(web_container.web, "load", fake_load)

    batch_data = td.web.BatchData(
        task_paths=task_paths,
        task_ids=task_ids,
        is_downloaded=True,
    )

    first = batch_data["task1"]
    second = batch_data["task1"]

    assert first is not second
    assert calls["load"] == 2
