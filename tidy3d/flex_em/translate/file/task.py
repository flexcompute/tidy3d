"""Task-file boundary helpers for public Tidy3D and flex-em files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.flex_em.translate import from_tidy3d, to_tidy3d

if TYPE_CHECKING:
    from os import PathLike


def from_tidy3d_file(path: str | PathLike[str]) -> object:
    """Load a public Tidy3D task file and convert it to a schema task object."""

    from tidy3d import Tidy3dBaseModel

    return from_tidy3d(Tidy3dBaseModel.from_file(path))


def to_tidy3d_file(schema_task: object, path: str | PathLike[str]) -> str | PathLike[str]:
    """Convert a schema task object and write a public Tidy3D task file."""

    to_tidy3d(schema_task).to_file(path)
    return path


def from_em_schema_file(path: str | PathLike[str]) -> object:
    """Load a backend schema task file."""

    from flex_em.schema.tidy3d import Tidy3dBaseModel

    return Tidy3dBaseModel.from_file(path)


def to_em_schema_file(schema_task: object, path: str | PathLike[str]) -> str | PathLike[str]:
    """Write a backend schema task file."""

    schema_task.to_file(path)
    return path
