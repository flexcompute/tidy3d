"""Adjoint surface sample-set contract types, owned by ``flexcompute.core._migration.em.schema``.

The sample-set artifact is a client<->solver wire contract: the client generates
and uploads it with the forward task, and the solver consumes it during adjoint
postprocessing. To keep that contract single-source (no drifting public twin), the
classes are defined once in ``flexcompute.core._migration.em.schema`` and re-exported here rather than
translated into duplicate public models. This module is the owned translator
boundary for the sample-set contract (see ``flex/public/flexcompute-core/README.md``): the
rest of the public package imports these names from here, never from
``flexcompute.core._migration.em.schema`` directly, and the functions below translate schema exceptions
into public tidy3d exceptions.

Import-time note: importing this module loads the ``flexcompute.core._migration.em.schema.tidy3d`` model
tree (~1s). Modules loaded by ``import tidy3d`` must not import it at module
level — use ``TYPE_CHECKING`` for annotations and function-local imports at
runtime call sites. Modules in the lazily-loaded web tree may import it normally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from flexcompute.core._migration.em.schema.tidy3d.components.autograd.sample_sets import (
    CircularCrossSectionMetadataBase,  # noqa: TC — runtime re-export consumed across the client package
    CircularCrossSectionSidewallAngleMetadata,
    CircularCrossSectionSidewallMetadata,
    CircularCrossSectionSlabFaceMetadata,
    GeometrySampleSets,  # noqa: TC — runtime re-export consumed across the client package
    PolySlabSidewallAngleMetadata,
    PolySlabSidewallMetadata,
    PolySlabSlabFaceMetadata,
    SampleSetEntry,
    SamplingContext,
    StructureSampleSets,
    StructureSampleSetsEntry,
    SurfaceSampleSet,  # noqa: TC — runtime re-export consumed across the client package
    TriangleMeshSurfaceMetadata,
    encode_traced_keys,
    shape_paths_by_structure,
)
from flexcompute.core._migration.em.schema.tidy3d.components.autograd.sample_sets import (
    circular_cross_section_metadata as _schema_circular_cross_section_metadata,
)
from flexcompute.core._migration.em.schema.tidy3d.components.autograd.sample_sets import (
    typed_sample_set_metadata as _schema_typed_sample_set_metadata,
)
from flexcompute.core._migration.em.schema.tidy3d.components.autograd.sample_sets import (
    validate_sample_sets_coverage as _schema_validate_sample_sets_coverage,
)
from flexcompute.core._migration.em.schema.tidy3d.components.data.data_array import (
    CellDataArray,
    IndexedDataArray,
    PointDataArray,
)

# The schema package has its own exception hierarchy (an M0 copy); contract functions
# like its validate_sample_sets_coverage raise ITS AdjointError, which client code
# cannot catch as tidy3d.exceptions.AdjointError. The schema exception stays private
# to this module: the boundary functions below translate it, so client code only
# ever sees tidy3d exceptions.
from flexcompute.core._migration.em.schema.tidy3d.exceptions import (
    AdjointError as _SchemaAdjointError,
)

from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from flexcompute.core._migration.em.schema.tidy3d.components.autograd.sample_sets import (
        SampleSetMetadataType,
    )

MetadataT = TypeVar("MetadataT")

__all__ = [
    "CellDataArray",
    "CircularCrossSectionMetadataBase",
    "CircularCrossSectionSidewallAngleMetadata",
    "CircularCrossSectionSidewallMetadata",
    "CircularCrossSectionSlabFaceMetadata",
    "GeometrySampleSets",
    "IndexedDataArray",
    "PointDataArray",
    "PolySlabSidewallAngleMetadata",
    "PolySlabSidewallMetadata",
    "PolySlabSlabFaceMetadata",
    "SampleSetEntry",
    "SamplingContext",
    "StructureSampleSets",
    "StructureSampleSetsEntry",
    "SurfaceSampleSet",
    "TriangleMeshSurfaceMetadata",
    "circular_cross_section_metadata",
    "encode_traced_keys",
    "shape_paths_by_structure",
    "typed_sample_set_metadata",
    "validate_sample_sets_coverage",
]


def circular_cross_section_metadata(
    metadata: SampleSetMetadataType | None, num_pts_circumference: int
) -> CircularCrossSectionMetadataBase:
    """Stamp polyslab metadata as its circular-cross-section variant at this boundary.

    Delegates to the schema helper and translates the schema package's
    ``AdjointError`` into ``tidy3d.exceptions.AdjointError``, so public-side
    generation raises only tidy3d exceptions.
    """
    try:
        return _schema_circular_cross_section_metadata(metadata, num_pts_circumference)
    except _SchemaAdjointError as exc:
        raise AdjointError(str(exc)) from exc


def typed_sample_set_metadata(
    sample_set: SurfaceSampleSet, expected_type: type[MetadataT], consumer: str
) -> MetadataT:
    """Return a sample set's metadata as ``expected_type`` at this boundary.

    Delegates to the schema accessor and translates the schema package's
    ``AdjointError`` into ``tidy3d.exceptions.AdjointError``, so public-side
    consumption raises only tidy3d exceptions.
    """
    try:
        return _schema_typed_sample_set_metadata(sample_set, expected_type, consumer)
    except _SchemaAdjointError as exc:
        raise AdjointError(str(exc)) from exc


def validate_sample_sets_coverage(
    sample_sets: GeometrySampleSets, required: Mapping[int, Collection[tuple]]
) -> None:
    """Validate artifact coverage against the backward request at this boundary.

    Delegates to the schema helper and translates the schema package's
    ``AdjointError`` into ``tidy3d.exceptions.AdjointError``, so public-side
    callers only ever see tidy3d exceptions.
    """
    try:
        _schema_validate_sample_sets_coverage(sample_sets, required)
    except _SchemaAdjointError as exc:
        raise AdjointError(str(exc)) from exc
