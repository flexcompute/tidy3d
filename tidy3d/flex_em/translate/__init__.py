"""Convert backend objects between public Tidy3D and flex-em."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.flex_em.translate.map import (
    DATA_CONVERTERS,
    DATA_PUBLIC_CLASSES,
    DATA_SCHEMA_CLASSES,
    TASK_CONVERTERS,
    TASK_PUBLIC_CLASSES,
    TASK_SCHEMA_CLASSES,
    resolve,
)

if TYPE_CHECKING:
    from tidy3d.flex_em.translate.map import Tidy3DSolverDataMap, Tidy3DSolverTaskMap

__all__ = [
    "Tidy3DSolverDataMap",
    "Tidy3DSolverTaskMap",
    "data_from_tidy3d",
    "data_to_tidy3d",
    "from_tidy3d",
    "to_em_schema_data_type",
    "to_em_schema_type",
    "to_tidy3d",
    "to_tidy3d_data_type",
    "to_tidy3d_type",
]


def __getattr__(name: str) -> Any:
    """Lazily expose translator family enums without importing type-only names."""

    if name == "Tidy3DSolverDataMap":
        from tidy3d.flex_em.translate.map import Tidy3DSolverDataMap

        return Tidy3DSolverDataMap
    if name == "Tidy3DSolverTaskMap":
        from tidy3d.flex_em.translate.map import Tidy3DSolverTaskMap

        return Tidy3DSolverTaskMap
    raise AttributeError(name)


def _classify(obj: Any, classes: dict[Any, str]) -> Any:
    for family, class_path in classes.items():
        if isinstance(obj, resolve(class_path)):
            return family
    return None


def from_tidy3d(public_task: Any) -> Any:
    """Convert a supported public Tidy3D task object into the schema model."""

    task_type = to_em_schema_type(public_task)
    return resolve(TASK_CONVERTERS[task_type][0])(public_task)


def to_tidy3d(schema_task: Any) -> Any:
    """Convert a supported schema task object into the public Tidy3D model."""

    task_type = to_tidy3d_type(schema_task)
    return resolve(TASK_CONVERTERS[task_type][1])(schema_task)


def data_from_tidy3d(public_data: Any) -> Any:
    """Convert supported public Tidy3D result data into the schema model."""

    data_type = to_em_schema_data_type(public_data)
    return resolve(DATA_CONVERTERS[data_type][0])(public_data)


def data_to_tidy3d(schema_data: Any) -> Any:
    """Convert supported schema result data into the public Tidy3D model."""

    data_type = to_tidy3d_data_type(schema_data)
    return resolve(DATA_CONVERTERS[data_type][1])(schema_data)


def to_em_schema_type(public_obj: Any) -> Tidy3DSolverTaskMap:
    """Classify a public Tidy3D task object for schema conversion."""

    task_type = _classify(public_obj, TASK_PUBLIC_CLASSES)
    if task_type is not None:
        return task_type
    raise TypeError(
        "Unsupported public Tidy3D task object: "
        f"{type(public_obj).__module__}.{type(public_obj).__name__}"
    )


def to_tidy3d_type(schema_obj: Any) -> Tidy3DSolverTaskMap:
    """Classify a schema task object for public Tidy3D conversion."""

    task_type = _classify(schema_obj, TASK_SCHEMA_CLASSES)
    if task_type is not None:
        return task_type
    raise TypeError(
        "Unsupported flex-em task object: "
        f"{type(schema_obj).__module__}.{type(schema_obj).__name__}"
    )


def to_em_schema_data_type(public_data: Any) -> Tidy3DSolverDataMap:
    """Classify public Tidy3D result data for schema conversion."""

    data_type = _classify(public_data, DATA_PUBLIC_CLASSES)
    if data_type is not None:
        return data_type
    raise TypeError(
        "Unsupported public Tidy3D data object: "
        f"{type(public_data).__module__}.{type(public_data).__name__}"
    )


def to_tidy3d_data_type(schema_data: Any) -> Tidy3DSolverDataMap:
    """Classify schema result data for public Tidy3D conversion."""

    data_type = _classify(schema_data, DATA_SCHEMA_CLASSES)
    if data_type is not None:
        return data_type
    raise TypeError(
        "Unsupported flex-em data object: "
        f"{type(schema_data).__module__}.{type(schema_data).__name__}"
    )
