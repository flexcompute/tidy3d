"""Translate Tidy3D geometries to the core geometries that validate them.

Each translator builds the core's schema record from the geometry's fields, with
validation, and shares the geometry's arrays rather than copying them. A composite's
record nests the records its members' cores hold, so no member data is copied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flexcompute.geometry.base import BoxCore
from flexcompute.geometry.composites import (
    ClipOperationCore,
    GeometryArrayCore,
    GeometryGroupCore,
    TransformedCore,
)
from flexcompute.geometry.mesh import TriangleMeshCore
from flexcompute.geometry.polyslab import PolySlabCore
from flexcompute.geometry.primitives import CylinderCore, SphereCore
from flexcompute.geometry.schema import (
    BoxSchema,
    ClipOperationSchema,
    ComplexPolySlabSchema,
    CylinderSchema,
    GeometryArraySchema,
    GeometryGroupSchema,
    PolySlabSchema,
    SphereSchema,
    TransformedSchema,
    TriangleMeshSchema,
)

# The modules, not their classes: a module such as ``primitives`` builds a geometry while it
# loads, before its later classes exist, and a case below names a class only when it is tried.
# ``slabs`` leaves the name ``polyslab`` to its translator.
from tidy3d.components.geometry import base, mesh, primitives
from tidy3d.components.geometry import polyslab as slabs

if TYPE_CHECKING:
    from flexcompute.geometry.base import GeometryCore


def to_core(geometry: base.Geometry) -> GeometryCore:
    """The core geometry of ``geometry``. A subclass, such as a monitor that is a ``Box``,
    translates as the geometry it extends."""
    match geometry:
        case base.Box():
            return box(geometry)
        case primitives.Sphere():
            return sphere(geometry)
        case primitives.Cylinder():
            return cylinder(geometry)
        case slabs.ComplexPolySlabBase():  # before PolySlab, which it extends
            return complex_polyslab(geometry)
        case slabs.PolySlab():
            return polyslab(geometry)
        case mesh.TriangleMesh():
            return triangle_mesh(geometry)
        case base.Transformed():
            return transformed(geometry)
        case base.ClipOperation():
            return clip_operation(geometry)
        case base.GeometryGroup():
            return geometry_group(geometry)
        case base.GeometryArray():
            return geometry_array(geometry)
    raise NotImplementedError(f"'{type(geometry).__name__}' has no core geometry.")


def box(geometry: base.Box) -> BoxCore:
    return BoxCore(BoxSchema(center=geometry.center, size=geometry.size))


def sphere(geometry: primitives.Sphere) -> SphereCore:
    return SphereCore(SphereSchema(center=geometry.center, radius=geometry.radius))


def cylinder(geometry: primitives.Cylinder) -> CylinderCore:
    return CylinderCore(
        CylinderSchema(
            center=geometry.center,
            radius=geometry.radius,
            length=geometry.length,
            axis=geometry.axis,
            sidewall_angle=geometry.sidewall_angle,
            reference_plane=geometry.reference_plane,
        )
    )


def polyslab(geometry: slabs.PolySlab) -> PolySlabCore:
    return PolySlabCore(PolySlabSchema(**_polyslab_fields(geometry)))


def complex_polyslab(geometry: slabs.ComplexPolySlabBase) -> PolySlabCore:
    return PolySlabCore(ComplexPolySlabSchema(**_polyslab_fields(geometry)))


def _polyslab_fields(geometry: slabs.PolySlab) -> dict[str, Any]:
    return {
        "axis": geometry.axis,
        "slab_bounds": geometry.slab_bounds,
        "sidewall_angle": geometry.sidewall_angle,
        "reference_plane": geometry.reference_plane,
        "dilation": geometry.dilation,
        "vertices": geometry.vertices,
        "bulges": geometry.bulges,
    }


def triangle_mesh(geometry: mesh.TriangleMesh) -> TriangleMeshCore:
    dataset = geometry.mesh_dataset
    triangles = None if dataset is None else dataset.surface_mesh.data
    return TriangleMeshCore(TriangleMeshSchema(triangles=triangles))


def transformed(geometry: base.Transformed) -> TransformedCore:
    # The geometry package refuses a transform of a transform, so a nested one folds here.
    member, transform = geometry.geometry._core.data, geometry.transform
    if isinstance(member, TransformedSchema):
        member, transform = member.geometry, np.dot(transform, member.transform)
    return TransformedCore(TransformedSchema(geometry=member, transform=transform))


def clip_operation(geometry: base.ClipOperation) -> ClipOperationCore:
    record = ClipOperationSchema(
        operation=geometry.operation,
        geometry_a=geometry.geometry_a._core.data,
        geometry_b=geometry.geometry_b._core.data,
    )
    return ClipOperationCore(record)


def geometry_group(geometry: base.GeometryGroup) -> GeometryGroupCore:
    members = tuple(member._core.data for member in geometry.geometries)
    return GeometryGroupCore(GeometryGroupSchema(geometries=members))


def geometry_array(geometry: base.GeometryArray) -> GeometryArrayCore:
    record = GeometryArraySchema(
        geometry=geometry.geometry._core.data,
        offsets=geometry.offsets,
        transforms=geometry.transforms,
    )
    return GeometryArrayCore(record)
