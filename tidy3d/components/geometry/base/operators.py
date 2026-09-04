"""Boolean operators and geometry autograd hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.components.autograd.path_utils import validate_traced_path
from tidy3d.constants import inf

if TYPE_CHECKING:
    from typing_extensions import Self

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.path_utils import AutogradRoute

    from .clip_operation import ClipOperation
    from .core import Geometry
    from .geometry_group import GeometryGroup


def _is_geometry(obj: Any) -> bool:
    from .core import Geometry

    return isinstance(obj, Geometry)


def _compute_derivatives(
    self,  # pyrefly: ignore[implicit-any-parameter]
    derivative_info: DerivativeInfo,
) -> AutogradFieldMap:
    """Compute the adjoint derivatives for this object."""
    raise NotImplementedError(f"Can't compute derivative for 'Geometry': '{type(self)}'.")


def _resolve_autograd_route(
    self,  # pyrefly: ignore[implicit-any-parameter]
    field_path: tuple[Any, ...],
) -> AutogradRoute:
    """Resolve and validate one traced geometry path for adjoint routing."""
    return validate_traced_path(
        parameter_kind="geometry",
        owner_kind="geometry type",
        owner_name=type(self).__name__,
        field_path=field_path,
        supported_paths=self._traced_supported_paths,
        supported_parameters=type(self)._traced_autograd_supported_parameters(),
    )


def _as_union(self) -> list[Geometry]:  # pyrefly: ignore[implicit-any-parameter]
    """Return a list of geometries that, united, make up the given geometry."""
    from .clip_operation import ClipOperation
    from .geometry_group import GeometryGroup

    if isinstance(self, GeometryGroup):
        return self.geometries

    if isinstance(self, ClipOperation) and self.operation == "union":
        return (self.geometry_a, self.geometry_b)
    return (self,)


def __add__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: int | Geometry,
) -> Self | GeometryGroup:
    """Union of geometries"""
    from .geometry_group import GeometryGroup

    # This allows the user to write sum(geometries...) with the default start=0
    if isinstance(other, int):
        return self
    if not _is_geometry(other):
        return NotImplemented
    return GeometryGroup(geometries=self._as_union() + other._as_union())


def __radd__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: int | Geometry,
) -> Self | GeometryGroup:
    """Union of geometries"""
    from .geometry_group import GeometryGroup

    # This allows the user to write sum(geometries...) with the default start=0
    if isinstance(other, int):
        return self
    if not _is_geometry(other):
        return NotImplemented
    return GeometryGroup(geometries=other._as_union() + self._as_union())


def __or__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: Geometry,
) -> GeometryGroup:
    """Union of geometries"""
    from .geometry_group import GeometryGroup

    if not _is_geometry(other):
        return NotImplemented
    return GeometryGroup(geometries=self._as_union() + other._as_union())


def __mul__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: Geometry,
) -> ClipOperation:
    """Intersection of geometries"""
    from .clip_operation import ClipOperation

    if not _is_geometry(other):
        return NotImplemented
    return ClipOperation(operation="intersection", geometry_a=self, geometry_b=other)


def __and__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: Geometry,
) -> ClipOperation:
    """Intersection of geometries"""
    from .clip_operation import ClipOperation

    if not _is_geometry(other):
        return NotImplemented
    return ClipOperation(operation="intersection", geometry_a=self, geometry_b=other)


def __sub__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: Geometry,
) -> ClipOperation:
    """Difference of geometries"""
    from .clip_operation import ClipOperation

    if not _is_geometry(other):
        return NotImplemented
    return ClipOperation(operation="difference", geometry_a=self, geometry_b=other)


def __xor__(
    self,  # pyrefly: ignore[implicit-any-parameter]
    other: Geometry,
) -> ClipOperation:
    """Symmetric difference of geometries"""
    from .clip_operation import ClipOperation

    if not _is_geometry(other):
        return NotImplemented
    return ClipOperation(operation="symmetric_difference", geometry_a=self, geometry_b=other)


def __pos__(self) -> Self:  # pyrefly: ignore[implicit-any-parameter]
    """No op"""
    return self


def __neg__(self) -> ClipOperation:  # pyrefly: ignore[implicit-any-parameter]
    """Opposite of a geometry"""
    from .box import Box
    from .clip_operation import ClipOperation

    return ClipOperation(
        operation="difference", geometry_a=Box(size=(inf, inf, inf)), geometry_b=self
    )


def __invert__(self) -> ClipOperation:  # pyrefly: ignore[implicit-any-parameter]
    """Opposite of a geometry"""
    from .box import Box
    from .clip_operation import ClipOperation

    return ClipOperation(
        operation="difference", geometry_a=Box(size=(inf, inf, inf)), geometry_b=self
    )
