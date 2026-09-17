"""Result-data file boundary helpers for public Tidy3D and the migrated EM schema files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.em.translate import data_from_tidy3d, data_to_tidy3d

if TYPE_CHECKING:
    from os import PathLike


def from_tidy3d_file(path: str | PathLike[str]) -> object:
    """Load a public Tidy3D result-data file and convert it to schema data."""

    from tidy3d import Tidy3dBaseModel
    from tidy3d.plugins.smatrix.data import modal as _modal  # noqa: F401

    return data_from_tidy3d(Tidy3dBaseModel.from_file(path))


def to_tidy3d_file(schema_data: object, path: str | PathLike[str]) -> str | PathLike[str]:
    """Convert schema result data and write a public Tidy3D result-data file."""

    data_to_tidy3d(schema_data).to_file(path)
    return path


def to_tidy3d_file_from_em_schema_file(
    schema_path: str | PathLike[str], path: str | PathLike[str]
) -> str | PathLike[str]:
    """Convert a schema result-data file and write a public Tidy3D result-data file."""

    schema_data = from_em_schema_file(schema_path)
    return to_tidy3d_file(schema_data, path)


def from_em_schema_file(path: str | PathLike[str]) -> object:
    """Load backend schema result data."""

    from flexcompute.core._migration.em.schema.tidy3d import Tidy3dBaseModel

    return Tidy3dBaseModel.from_file(path)


def to_em_schema_file(schema_data: object, path: str | PathLike[str]) -> str | PathLike[str]:
    """Write backend schema result data."""

    schema_data.to_file(path)
    return path
