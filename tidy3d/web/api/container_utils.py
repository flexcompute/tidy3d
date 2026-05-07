"""Shared helpers for flattening and reconstructing nested workflow containers."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .container_types import BatchTaskTree


def _sanitize_task_name_component(value: Any) -> str:
    """Convert a path component into a readable task-name fragment."""
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._")
    return text or "item"


def validate_mapping_key(key: object) -> None:
    """Ensure nested container mapping keys are hashable."""
    if not isinstance(key, Hashable):
        raise ValueError(
            "Container mapping keys must be hashable. "
            f"Got key {key!r} of type {type(key).__name__!r}."
        )


def _make_task_name_factory(flat: dict[str, Any]) -> Callable[[tuple[Any, ...]], str]:
    """Build a stable task-name generator for a single flattening pass."""
    name_counts: dict[str, int] = {}

    def _next_task_name(path: tuple[Any, ...]) -> str:
        base_parts = [_sanitize_task_name_component(part) for part in path]
        base_name = "__".join(base_parts) or "task"
        count = name_counts.get(base_name, 0)
        candidate = base_name if count == 0 else f"{base_name}__dup{count + 1}"
        while candidate in flat:
            count += 1
            candidate = f"{base_name}__dup{count + 1}"
        name_counts[base_name] = count + 1
        return candidate

    return _next_task_name


def _flatten_container_internal(
    node: Any,
    *,
    is_leaf: Callable[[Any], bool],
    validate_dict_key: Callable[[Any], None] | None,
    leaf_id: Callable[[tuple[Any, ...], Any], str] | None,
    sequence_builder: Callable[[list[Any]], Any],
    preserve_list_kind: bool,
) -> tuple[dict[str, Any], Any]:
    """Shared recursive flattener used by generic and batch-specific container walkers."""

    flat: dict[str, Any] = {}
    next_task_name = _make_task_name_factory(flat)

    def _recur(value: Any, path: tuple[Any, ...]) -> Any:
        if is_leaf(value):
            task_name = leaf_id(path, value) if leaf_id is not None else next_task_name(path)
            flat[task_name] = value
            return task_name
        if isinstance(value, tuple):
            items = [_recur(item, (*path, index)) for index, item in enumerate(value)]
            return sequence_builder(items)
        if isinstance(value, list):
            items = [_recur(item, (*path, index)) for index, item in enumerate(value)]
            if preserve_list_kind:
                return items
            return sequence_builder(items)
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if validate_dict_key is not None:
                    validate_dict_key(key)
                result[key] = _recur(item, (*path, key))
            return result
        raise TypeError(f"Unsupported element in container: {type(value)!r}")

    return flat, _recur(node, ())


def flatten_container(
    node: Any,
    *,
    is_leaf: Callable[[Any], bool],
    validate_dict_key: Callable[[Any], None] | None = None,
    leaf_id: Callable[[tuple[Any, ...], Any], str] | None = None,
) -> dict[str, Any]:
    """Flatten a nested container into a flat leaf mapping."""
    flat, _ = _flatten_container_internal(
        node,
        is_leaf=is_leaf,
        validate_dict_key=validate_dict_key,
        leaf_id=leaf_id,
        sequence_builder=tuple,
        preserve_list_kind=True,
    )
    return flat


def flatten_task_container(
    node: Any,
    *,
    is_leaf: Callable[[Any], bool],
    validate_dict_key: Callable[[Any], None] | None = None,
    leaf_id: Callable[[tuple[Any, ...], Any], str] | None = None,
) -> tuple[dict[str, Any], BatchTaskTree]:
    """Flatten a batch container and return a tuple-backed task-name tree.

    Lists and tuples are both normalized to tuples in the returned tree so batch internals
    do not need to preserve a separate sequence-kind marker.
    """

    flat, task_tree = _flatten_container_internal(
        node,
        is_leaf=is_leaf,
        validate_dict_key=validate_dict_key,
        leaf_id=leaf_id,
        sequence_builder=tuple,
        preserve_list_kind=False,
    )
    return flat, task_tree


def reconstruct_task_container(
    task_tree: BatchTaskTree, get_leaf_value: Callable[[str], Any]
) -> Any:
    """Rebuild a tuple-backed batch container from a task-name tree."""
    if isinstance(task_tree, str):
        return get_leaf_value(task_tree)
    if isinstance(task_tree, (tuple, list)):
        return tuple(reconstruct_task_container(item, get_leaf_value) for item in task_tree)
    return {
        key: reconstruct_task_container(item, get_leaf_value) for key, item in task_tree.items()
    }
