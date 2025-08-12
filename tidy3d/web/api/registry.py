"""Runtime registry for plugin simulation types and loaders.

This allows ``tidy3d.web.api`` to remain decoupled from plugin packages by
letting plugins self-register at import time.
"""

from __future__ import annotations

from typing import Any, Callable


class _Registry:
    """Holds mappings used by the web API to handle external types.

    - ``class_to_task_type``: map Python classes to server-side task types.
    - ``json_type_to_sim_loader``: map JSON ``type`` strings to simulation loaders.
    - ``json_type_to_data_loader``: map JSON ``type`` strings to data loaders.
    - ``task_type_to_remote_files``: map task types to (sim_file, data_file) names.
    """

    class_to_task_type: dict[type[Any], str] = {}
    json_type_to_sim_loader: dict[str, Callable[[str], Any]] = {}
    json_type_to_data_loader: dict[str, Callable[[str], Any]] = {}
    # Optional loaders that construct directly from in-memory dicts/json strings
    json_type_to_sim_loader_from_dict: dict[str, Callable[[dict], Any]] = {}
    json_type_to_data_loader_from_dict: dict[str, Callable[[dict], Any]] = {}
    task_type_to_remote_files: dict[str, tuple[str, str]] = {}


def register_simulation_type(py_class: type[Any], task_type: str) -> None:
    """Register a Python class as a given task type."""

    _Registry.class_to_task_type[py_class] = task_type


def register_sim_loader(json_type: str, loader: Callable[[str], Any]) -> None:
    """Register a JSON ``type`` string to a simulation loader callable.

    The loader should accept a file path and return the decoded simulation object.
    """

    _Registry.json_type_to_sim_loader[json_type] = loader


def register_sim_loader_from_dict(json_type: str, loader: Callable[[dict], Any]) -> None:
    """Register a JSON ``type`` string to a simulation loader from dict."""

    _Registry.json_type_to_sim_loader_from_dict[json_type] = loader


def register_data_loader(json_type: str, loader: Callable[[str], Any]) -> None:
    """Register a JSON ``type`` string to a data loader callable.

    The loader should accept a file path and return the decoded data object.
    """

    _Registry.json_type_to_data_loader[json_type] = loader


def register_data_loader_from_dict(json_type: str, loader: Callable[[dict], Any]) -> None:
    """Register a JSON ``type`` string to a data loader from dict."""

    _Registry.json_type_to_data_loader_from_dict[json_type] = loader


def register_remote_files(task_type: str, sim_file: str, data_file: str) -> None:
    """Register custom remote file names for a given task type."""

    _Registry.task_type_to_remote_files[task_type] = (sim_file, data_file)


def get_task_type_for_instance(obj: Any) -> str | None:
    """Return server task type for an object if registered, else ``None``."""

    for cls, task_type in _Registry.class_to_task_type.items():
        if isinstance(obj, cls):
            return task_type
    return None


def get_registered_sim_loader(json_type: str) -> Callable[[str], Any] | None:
    return _Registry.json_type_to_sim_loader.get(json_type)


def get_registered_data_loader(json_type: str) -> Callable[[str], Any] | None:
    return _Registry.json_type_to_data_loader.get(json_type)


def get_registered_sim_loader_from_dict(json_type: str) -> Callable[[dict], Any] | None:
    return _Registry.json_type_to_sim_loader_from_dict.get(json_type)


def get_registered_data_loader_from_dict(json_type: str) -> Callable[[dict], Any] | None:
    return _Registry.json_type_to_data_loader_from_dict.get(json_type)


def get_remote_files_for_task_type(task_type: str) -> tuple[str, str] | None:
    return _Registry.task_type_to_remote_files.get(task_type)
