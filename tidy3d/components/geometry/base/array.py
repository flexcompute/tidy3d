"""Geometry array base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import Field, field_validator, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.types import (
    Coordinate,
    MatrixReal4x4,
)
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import (
    ValidationError,
)
from tidy3d.log import log

from .constants import LINEAR_TRANSFORM_TOL
from .core import Geometry, assert_geometry_finite, check_transform_invertible
from .geometry_group import GeometryGroup
from .transformed import Transformed

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import (
        Axis,
        Bound,
        Shapely,
    )


class GeometryArray(Geometry):
    """A geometry representing an array of copies of a base geometry, with optional offsets
    and/or linear transformations applied to each copy.

    Notes
    -----
    This class provides an efficient way to represent arrays of repeated geometries,
    avoiding the need to create many individual geometry objects.

    The instance pose for each copy is defined as: ``T(offsets[i]) @ L(transforms[i])``,
    where ``T`` is a translation matrix and ``L`` is the linear transform. In other words,
    the transform is applied first, then the translation.

    - ``offsets`` represent all per-instance translation.
    - ``transforms`` represent linear transforms only (rotation/reflection/scale/shear)
      and must not contain translation. Use ``offsets`` for translations.
    - If both ``offsets`` and ``transforms`` are ``None``, the array contains a single
      instance of the base geometry at the origin.
    - If both are provided, they must have the same length.
    - Adjoint/autodiff is not currently supported for ``GeometryArray``.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> box = td.Box(size=(1, 1, 1))
    >>> # Using offsets only:
    >>> offsets = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]]
    >>> array = td.GeometryArray(geometry=box, offsets=offsets)
    >>> # Or use the convenience method:
    >>> array = box.array(offsets=offsets)
    >>> # Using linear transforms only (rotation around z-axis):
    >>> rot_0 = td.Transformed.rotation(0, 2)  # no rotation
    >>> rot_90 = td.Transformed.rotation(np.pi/2, 2)  # 90 degree rotation
    >>> array = td.GeometryArray(geometry=box, transforms=[rot_0, rot_90])
    >>> # Both None gives single instance of base geometry:
    >>> array = td.GeometryArray(geometry=box)
    """

    geometry: discriminated_union(GeometryType) = Field(
        ...,
        title="Geometry",
        description="Base geometry to be repeated in the array.",
    )

    offsets: tuple[Coordinate, ...] | None = Field(
        default=None,
        title="Offsets",
        description="A tuple of 3D coordinate offsets. Each offset translates the base "
        "geometry (after any transform is applied) to create a copy. If not provided, no "
        "additional translation is applied beyond any transforms.",
    )

    transforms: tuple[MatrixReal4x4, ...] | None = Field(
        default=None,
        title="Transforms",
        description="A tuple of 4x4 linear-only transformation matrices "
        "(rotation/reflection/scale/shear, no translation). Typical transforms can be "
        "created using ``Transformed.rotation``, ``Transformed.reflection``, or ``Transformed.scaling``. "
        "Each transform is applied to the base geometry before the corresponding offset translation. "
        "If not provided, only translations from offsets are applied.",
    )

    _geometry_is_finite = assert_geometry_finite("geometry")

    @field_validator("transforms")
    @classmethod
    def _validate_transforms(
        cls, val: tuple[MatrixReal4x4, ...] | None
    ) -> tuple[MatrixReal4x4, ...] | None:
        """Validate that transforms are invertible, linear-only, and non-empty if provided."""
        if val is None:
            return val

        # Must not be empty if provided
        if len(val) < 1:
            raise ValidationError("'transforms' must have at least one transform when provided.")

        # Check each transform
        for i, transform in enumerate(val):
            # Check invertibility
            check_transform_invertible(transform, index=i)

            # Check linear-only (no translation)
            transform_array = np.asarray(transform)

            # Check translation column: transform[:3, 3] should be zero
            translation = transform_array[:3, 3]
            if not np.allclose(translation, 0, atol=LINEAR_TRANSFORM_TOL):
                idx_msg = f"at index {i}"
                raise ValidationError(
                    f"Transform {idx_msg} contains translation in [:3, 3] = {translation.tolist()}. "
                    "GeometryArray transforms must be linear-only (rotation/reflection/scale/shear). "
                    "Use the 'offsets' parameter for translations."
                )

            # Check bottom row: transform[3, :] should be [0, 0, 0, 1]
            bottom_row = transform_array[3, :]
            expected_bottom = np.array([0, 0, 0, 1])
            if not np.allclose(bottom_row, expected_bottom, atol=LINEAR_TRANSFORM_TOL):
                idx_msg = f"at index {i}"
                raise ValidationError(
                    f"Transform {idx_msg} has invalid homogeneous form: [3, :] = {bottom_row.tolist()}. "
                    "Expected [0, 0, 0, 1]."
                )

        return val

    @model_validator(mode="after")
    def _validate_offsets_and_transforms(self) -> Self:
        """Validate offsets and transforms are consistent."""
        offsets = self.offsets
        transforms = self.transforms

        # If offsets provided, must not be empty
        if offsets is not None and len(offsets) < 1:
            self._raise_validation_error_at_loc(
                ValidationError("'offsets' must have at least one offset when provided."), "offsets"
            )

        # If both provided, lengths must match
        if offsets is not None and transforms is not None:
            if len(offsets) != len(transforms):
                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Number of transforms ({len(transforms)}) must match "
                        f"number of offsets ({len(offsets)}) when both are provided."
                    ),
                    "transforms",
                )

        return self

    @cached_property
    def num_geometries(self) -> int:
        """Number of geometries in the array."""
        if self.offsets is not None:
            return len(self.offsets)
        if self.transforms is not None:
            return len(self.transforms)
        # Both None means single geometry (base geometry at origin)
        return 1

    @cached_property
    def _all_transforms(self) -> np.ndarray:
        """Compute all 4x4 transforms for all geometries in a vectorized way.

        Returns
        -------
        numpy.ndarray
            Array of shape (num_geometries, 4, 4) containing the full transform
            (rotation/scale + translation) for each geometry in the array.
        """
        n = self.num_geometries
        shape = (n, 4, 4)

        # Get all transforms, defaulting to identity if not provided
        if self.transforms is not None:
            transforms = np.array(self.transforms)
        else:
            transforms = np.broadcast_to(Transformed.identity(), shape)

        # Build translation matrices for all offsets
        # translation matrix: [[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]]
        translations = np.broadcast_to(Transformed.identity(), shape).copy()
        if self.offsets is not None:
            translations[:, :3, 3] = self.offsets

        # Apply transform, then translation: result = translation @ transform
        return np.matmul(translations, transforms)

    def _get_full_transform(self, index: int) -> MatrixReal4x4:
        """Get the full 4x4 transform for a geometry at given index (transform + translation)."""
        return self._all_transforms[index]

    @cached_property
    def _transformed_geometries(self) -> list[Transformed]:
        """List of transformed geometries in the array."""
        return [
            Transformed(geometry=self.geometry, transform=transform)
            for transform in self._all_transforms
        ]

    @cached_property
    def _geometry_group(self) -> GeometryGroup:
        """Return a GeometryGroup containing all transformed geometries in the array."""
        return GeometryGroup(geometries=tuple(self._transformed_geometries))

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        return self._geometry_group.bounds

    def intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_tilted_plane`.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        return self._geometry_group.intersections_tilted_plane(
            normal,
            origin,
            to_2D,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )

    def intersections_plane(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Returns list of shapely geometries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_plane`.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        return self._geometry_group.intersections_plane(
            x=x,
            y=y,
            z=z,
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
        )

    def intersects_axis_position(
        self, axis: int, position: float, section_tolerance_2d: bool = False
    ) -> bool:
        """Whether self intersects plane specified by a given position along a normal axis.

        Parameters
        ----------
        axis : int = None
            Axis normal to the plane.
        position : float = None
            Position of plane along the normal axis.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersects_axis_position`.

        Returns
        -------
        bool
            Whether this geometry intersects the plane.
        """
        return self._geometry_group.intersects_axis_position(
            axis, position, section_tolerance_2d=section_tolerance_2d
        )

    def inside(self, x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> NDArray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        return self._geometry_group.inside(x, y, z)

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        return self._geometry_group._volume(bounds)

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        # Surface area cannot be reliably computed when non-trivial transforms are present
        if self.transforms is not None:
            log.warning("Surface area of transformed elements cannot be calculated.")
            return None
        # For pure translations, sum surface areas using local bounds for base geometry
        total_area = 0.0
        for geom in self._transformed_geometries:
            # Transform bounds to local coordinate system
            vertices = np.dot(geom.inverse, Transformed._vertices_from_bounds(bounds))[:3]
            local_bounds = (tuple(vertices.min(axis=1)), tuple(vertices.max(axis=1)))
            instance_area = self.geometry.surface_area(local_bounds)
            if instance_area is None:
                return None
            total_area += instance_area
        return total_area

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        return self._geometry_group._normal_2dmaterial

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> GeometryGroup:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        return self._geometry_group._update_from_bounds(bounds=bounds, axis=axis)

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object.

        Raises
        ------
        NotImplementedError
            Adjoint/autodiff is not currently supported for GeometryArray.
        """
        raise NotImplementedError(
            "Adjoint is not currently supported for 'GeometryArray'.",
        )
