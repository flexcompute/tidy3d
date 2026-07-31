"""Volume mesher translation."""

from __future__ import annotations

from typing import Any

from flex_em.schema.tidy3d.components.tcad.mesher import VolumeMesher as VolumeMesherTaskInput

from tidy3d.flex_em.translate.base import dump_for_public, dump_for_schema


def from_task(mesher: Any) -> VolumeMesherTaskInput:
    """Convert a public Tidy3D volume mesher to the schema model."""

    return VolumeMesherTaskInput.model_validate(dump_for_schema(mesher, type_name="VolumeMesher"))


def to_task(mesher: Any) -> Any:
    """Convert a schema volume mesher to the public Tidy3D model."""

    from tidy3d.components.tcad.mesher import VolumeMesher

    return VolumeMesher.model_validate(dump_for_public(mesher, type_name="VolumeMesher"))
