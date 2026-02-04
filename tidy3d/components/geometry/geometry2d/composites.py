"""Composite 2D geometry types: Transformed2D and Array2D."""

from __future__ import annotations

import math
from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
import pydantic.v1 as pydantic
import shapely
import shapely.affinity

from tidy3d.components.geometry.base import Geometry, GeometryGroup, Transformed
from tidy3d.components.geometry.geometry2d.base import Bound2D, Geometry2D
from tidy3d.components.types import Axis, Coordinate2D, Shapely, annotate_type
from tidy3d.exceptions import SetupError, ValidationError

# Type alias for 3x3 homogeneous transformation matrix
MatrixReal3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class Transformed2D(Geometry2D):
    """A 2D geometry with an affine transformation applied.

    Applies a 3x3 homogeneous transformation matrix to any Geometry2D.
    Supports translation, rotation, scaling, shearing, and reflection.

    The transformation is applied lazily - the base geometry is stored
    unchanged, and the transform is applied when needed (bounds, to_shapely,
    to_3d_geometry).

    Example
    -------
    >>> from tidy3d.components.geometry.geometry2d import Rectangle2D, Transformed2D
    >>>
    >>> # Create a rotated rectangle
    >>> rect = Rectangle2D(center=(0, 0), size=(2, 1))
    >>> rotated = Transformed2D(
    ...     geometry=rect,
    ...     transform=Transformed2D.rotation(math.pi / 4),  # 45 degrees
    ... )
    >>>
    >>> # Or use the convenience method
    >>> rotated = rect.rotated(math.pi / 4)
    >>>
    >>> # Chain transformations (rotate then translate)
    >>> transformed = Transformed2D(
    ...     geometry=rect,
    ...     transform=Transformed2D.translation(5, 0) @ Transformed2D.rotation(math.pi / 2),
    ... )
    """

    geometry: annotate_type("Geometry2DType") = pydantic.Field(
        ...,
        title="Geometry",
        description="Base 2D geometry to be transformed.",
    )

    transform: MatrixReal3x3 = pydantic.Field(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        title="Transform",
        description="3x3 homogeneous transformation matrix. "
        "Format: [[a, b, tx], [c, d, ty], [0, 0, 1]]",
    )

    @pydantic.validator("transform")
    def _transform_is_invertible(cls, val: MatrixReal3x3) -> MatrixReal3x3:
        """Validate that the transform matrix is invertible."""
        mat = np.array(val)
        det = np.linalg.det(mat)
        if np.isclose(det, 0.0):
            raise ValidationError(
                "Transformation matrix must be invertible (non-zero determinant). "
                f"Got determinant = {det}"
            )
        return val

    @pydantic.validator("geometry")
    def _geometry_is_finite(cls, val: "Geometry2DType") -> "Geometry2DType":
        """Validate that the geometry has finite bounds."""
        bounds = val.bounds_2d
        if not np.isfinite(bounds).all():
            raise ValidationError(
                "Transformations are only supported on geometries with finite bounds. "
                f"Got bounds = {bounds}"
            )
        return val

    @pydantic.root_validator(skip_on_failure=True)
    def _flatten_nested_transforms(cls, values: dict) -> dict:
        """Flatten nested Transformed2D by composing matrices."""
        geometry = values.get("geometry")
        transform = values.get("transform")

        while isinstance(geometry, Transformed2D):
            inner = geometry
            geometry = inner.geometry
            # Compose transforms: outer @ inner
            transform = tuple(
                tuple(row)
                for row in np.dot(np.array(transform), np.array(inner.transform))
            )

        values["geometry"] = geometry
        values["transform"] = transform
        return values

    @cached_property
    def _transform_array(self) -> np.ndarray:
        """Transform matrix as numpy array."""
        return np.array(self.transform)

    @cached_property
    def _inverse(self) -> np.ndarray:
        """Inverse of the transform matrix."""
        return np.linalg.inv(self._transform_array)

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box of the transformed geometry.

        Computed by transforming the corner vertices of the inner
        geometry's bounding box.
        """
        (min_x, min_y), (max_x, max_y) = self.geometry.bounds_2d

        # 4 corner vertices in homogeneous coordinates
        corners = np.array([
            [min_x, min_x, max_x, max_x],
            [min_y, max_y, min_y, max_y],
            [1.0, 1.0, 1.0, 1.0],
        ])

        # Transform corners
        transformed = np.dot(self._transform_array, corners)

        # Extract x, y (discard homogeneous coordinate)
        xs = transformed[0, :]
        ys = transformed[1, :]

        return ((float(xs.min()), float(ys.min())), (float(xs.max()), float(ys.max())))

    def to_shapely(self) -> Shapely:
        """Convert to shapely geometry with transform applied.

        Uses shapely.affinity.affine_transform to apply the 2D
        affine transformation.
        """
        base_shapely = self.geometry.to_shapely()

        # Extract affine parameters from 3x3 matrix
        # shapely.affinity.affine_transform uses (a, b, d, e, xoff, yoff)
        # where: x' = a*x + b*y + xoff
        #        y' = d*x + e*y + yoff
        mat = self._transform_array
        a, b, tx = mat[0, 0], mat[0, 1], mat[0, 2]
        c, d, ty = mat[1, 0], mat[1, 1], mat[1, 2]

        return shapely.affinity.affine_transform(
            base_shapely,
            [a, b, c, d, tx, ty],
        )

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to 3D geometry with transform applied.

        The 2D transform is expanded to a 3D transform that leaves
        the extrusion axis unchanged.

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane. Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Default is 0.0.

        Returns
        -------
        Transformed
            A 3D Transformed geometry wrapping the extruded base geometry.
        """
        # Convert inner geometry to 3D
        inner_3d = self.geometry.to_3d_geometry(slab_bounds, axis, sidewall_angle)

        # Expand 2D transform to 3D (4x4 homogeneous matrix)
        transform_3d = self._expand_to_3d_transform(axis)

        return Transformed(geometry=inner_3d, transform=transform_3d.tolist())

    def _expand_to_3d_transform(self, axis: Axis) -> np.ndarray:
        """Expand 2D (3x3) transform to 3D (4x4) transform.

        The 2D transform operates in the plane perpendicular to the given axis.
        The third dimension is left unchanged (identity).

        Parameters
        ----------
        axis : Axis
            The axis perpendicular to the 2D plane (0=x, 1=y, 2=z).

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix.
        """
        mat_2d = self._transform_array

        # Start with identity 4x4
        mat_4x4 = np.eye(4)

        # Determine which 2D indices map to which 3D indices
        # For axis=2 (XY plane): 2D (0,1) -> 3D (0,1)
        # For axis=1 (XZ plane): 2D (0,1) -> 3D (0,2)
        # For axis=0 (YZ plane): 2D (0,1) -> 3D (1,2)
        if axis == 2:
            idx = [0, 1]  # X, Y
        elif axis == 1:
            idx = [0, 2]  # X, Z
        else:  # axis == 0
            idx = [1, 2]  # Y, Z

        # Copy 2x2 rotation/scale block
        mat_4x4[idx[0], idx[0]] = mat_2d[0, 0]
        mat_4x4[idx[0], idx[1]] = mat_2d[0, 1]
        mat_4x4[idx[1], idx[0]] = mat_2d[1, 0]
        mat_4x4[idx[1], idx[1]] = mat_2d[1, 1]

        # Copy translation
        mat_4x4[idx[0], 3] = mat_2d[0, 2]
        mat_4x4[idx[1], 3] = mat_2d[1, 2]

        return mat_4x4

    # --- Static factory methods for common transforms ---

    @staticmethod
    def translation(dx: float, dy: float) -> MatrixReal3x3:
        """Create a translation matrix.

        Parameters
        ----------
        dx : float
            Translation in x.
        dy : float
            Translation in y.

        Returns
        -------
        MatrixReal3x3
            3x3 transformation matrix.
        """
        return (
            (1.0, 0.0, dx),
            (0.0, 1.0, dy),
            (0.0, 0.0, 1.0),
        )

    @staticmethod
    def rotation(angle: float, origin: Coordinate2D = (0.0, 0.0)) -> MatrixReal3x3:
        """Create a rotation matrix.

        Parameters
        ----------
        angle : float
            Rotation angle in radians (counter-clockwise positive).
        origin : Coordinate2D
            Center of rotation. Default is (0, 0).

        Returns
        -------
        MatrixReal3x3
            3x3 transformation matrix.
        """
        c = math.cos(angle)
        s = math.sin(angle)
        ox, oy = origin

        # Rotation around origin: T(ox,oy) @ R @ T(-ox,-oy)
        # Combined: [[c, -s, ox - c*ox + s*oy],
        #            [s,  c, oy - s*ox - c*oy],
        #            [0,  0, 1]]
        tx = ox - c * ox + s * oy
        ty = oy - s * ox - c * oy

        return (
            (c, -s, tx),
            (s, c, ty),
            (0.0, 0.0, 1.0),
        )

    @staticmethod
    def scaling(
        sx: float,
        sy: Optional[float] = None,
        origin: Coordinate2D = (0.0, 0.0),
    ) -> MatrixReal3x3:
        """Create a scaling matrix.

        Parameters
        ----------
        sx : float
            Scale factor in x.
        sy : float, optional
            Scale factor in y. If None, uses sx (uniform scaling).
        origin : Coordinate2D
            Center of scaling. Default is (0, 0).

        Returns
        -------
        MatrixReal3x3
            3x3 transformation matrix.
        """
        if sy is None:
            sy = sx

        if np.isclose(sx, 0.0) or np.isclose(sy, 0.0):
            raise SetupError("Scale factors cannot be zero.")

        ox, oy = origin

        # Scaling around origin: T(ox,oy) @ S @ T(-ox,-oy)
        tx = ox * (1 - sx)
        ty = oy * (1 - sy)

        return (
            (sx, 0.0, tx),
            (0.0, sy, ty),
            (0.0, 0.0, 1.0),
        )

    @staticmethod
    def reflection(axis: Union[Literal["x", "y"], Coordinate2D]) -> MatrixReal3x3:
        """Create a reflection matrix.

        Parameters
        ----------
        axis : "x", "y", or Coordinate2D
            Axis of reflection:
            - "x": reflect across x-axis (y -> -y)
            - "y": reflect across y-axis (x -> -x)
            - Coordinate2D: reflect across line through origin with this direction

        Returns
        -------
        MatrixReal3x3
            3x3 transformation matrix.
        """
        if axis == "x":
            # Reflect across x-axis: (x, y) -> (x, -y)
            return (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        elif axis == "y":
            # Reflect across y-axis: (x, y) -> (-x, y)
            return (
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        else:
            # Reflect across arbitrary line through origin
            # Direction vector (dx, dy)
            dx, dy = axis
            length = math.hypot(dx, dy)
            if length < 1e-12:
                raise SetupError("Reflection axis vector cannot be zero.")
            dx /= length
            dy /= length

            # Reflection matrix: I - 2 * n * n^T, where n is normal to line
            # Normal to line (dx, dy) is (-dy, dx)
            # Actually: reflect across line => use line direction
            # Householder: R = I - 2 * n * n^T where n is perpendicular to line
            # For line direction (dx, dy), perpendicular is (-dy, dx)
            nx, ny = -dy, dx
            # R = [[1-2*nx^2, -2*nx*ny], [-2*nx*ny, 1-2*ny^2]]
            return (
                (1 - 2 * nx * nx, -2 * nx * ny, 0.0),
                (-2 * nx * ny, 1 - 2 * ny * ny, 0.0),
                (0.0, 0.0, 1.0),
            )


class Array2D(Geometry2D):
    """A base geometry repeated at multiple positions with optional rotations.

    Efficiently represents patterns like via arrays, pad grids, and BGA balls
    where the same shape is placed at many locations.

    Example
    -------
    >>> from tidy3d.components.geometry.geometry2d import Circle2D, Array2D
    >>>
    >>> # Via array with 100 vias
    >>> via = Circle2D(center=(0, 0), radius=0.15)
    >>> via_array = Array2D(
    ...     base_shape=via,
    ...     positions=[(i * 0.5, j * 0.5) for i in range(10) for j in range(10)],
    ... )
    >>>
    >>> # Connector pads with individual rotations
    >>> pad = Rectangle2D(center=(0, 0), size=(2, 4))
    >>> edge_pads = Array2D(
    ...     base_shape=pad,
    ...     positions=[(-10, 0), (10, 0), (0, -10), (0, 10)],
    ...     rotations=[90.0, 270.0, 0.0, 180.0],  # degrees
    ... )
    >>>
    >>> # Create a regular grid
    >>> grid = Array2D.grid(
    ...     base_shape=Circle2D(center=(0, 0), radius=0.1),
    ...     columns=5,
    ...     rows=5,
    ...     spacing=(0.5, 0.5),
    ... )
    """

    base_shape: annotate_type("Geometry2DType") = pydantic.Field(
        ...,
        title="Base Shape",
        description="The 2D geometry to be repeated at each position.",
    )

    positions: tuple[Coordinate2D, ...] = pydantic.Field(
        ...,
        title="Positions",
        description="List of (x, y) positions where the base shape is placed. "
        "Each position is an offset from (0, 0).",
        units="um",
    )

    rotations: Optional[tuple[float, ...]] = pydantic.Field(
        None,
        title="Rotations",
        description="Optional per-instance rotation angles in degrees. "
        "If provided, length must match positions. "
        "Rotation is applied around each instance's position. "
        "Counter-clockwise is positive.",
    )

    @pydantic.validator("positions")
    def _positions_not_empty(cls, val: tuple[Coordinate2D, ...]) -> tuple[Coordinate2D, ...]:
        """Ensure positions is not empty."""
        if len(val) == 0:
            raise SetupError("Array2D.positions must not be empty.")
        return val

    @pydantic.validator("rotations", always=True)
    def _rotations_length_matches(
        cls,
        val: Optional[tuple[float, ...]],
        values: dict,
    ) -> Optional[tuple[float, ...]]:
        """Ensure rotations length matches positions if provided."""
        if val is None:
            return None

        positions = values.get("positions")
        if positions is not None and len(val) != len(positions):
            raise SetupError(
                f"len(rotations)={len(val)} must equal len(positions)={len(positions)}"
            )

        return val

    @pydantic.validator("base_shape")
    def _base_shape_is_finite(cls, val: "Geometry2DType") -> "Geometry2DType":
        """Validate that base_shape has finite bounds."""
        bounds = val.bounds_2d
        if not np.isfinite(bounds).all():
            raise ValidationError(
                "Array2D.base_shape must have finite bounds. "
                f"Got bounds = {bounds}"
            )
        return val

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box containing all instances.

        Accounts for translations and rotations of the base shape.
        """
        base_bounds = self.base_shape.bounds_2d
        (base_min_x, base_min_y), (base_max_x, base_max_y) = base_bounds

        # If no rotations, bounds is simpler
        if self.rotations is None:
            # Just offset base bounds by each position
            all_min_x = min(pos[0] + base_min_x for pos in self.positions)
            all_max_x = max(pos[0] + base_max_x for pos in self.positions)
            all_min_y = min(pos[1] + base_min_y for pos in self.positions)
            all_max_y = max(pos[1] + base_max_y for pos in self.positions)
            return ((all_min_x, all_min_y), (all_max_x, all_max_y))

        # With rotations, compute transformed bounds for each instance
        # Base shape corners (relative to origin)
        corners = np.array([
            [base_min_x, base_min_x, base_max_x, base_max_x],
            [base_min_y, base_max_y, base_min_y, base_max_y],
        ])

        all_min_x = float("inf")
        all_max_x = float("-inf")
        all_min_y = float("inf")
        all_max_y = float("-inf")

        for i, (px, py) in enumerate(self.positions):
            angle = math.radians(self.rotations[i])
            c, s = math.cos(angle), math.sin(angle)

            # Rotate corners around origin, then translate
            rot_matrix = np.array([[c, -s], [s, c]])
            rotated = rot_matrix @ corners

            xs = rotated[0, :] + px
            ys = rotated[1, :] + py

            all_min_x = min(all_min_x, xs.min())
            all_max_x = max(all_max_x, xs.max())
            all_min_y = min(all_min_y, ys.min())
            all_max_y = max(all_max_y, ys.max())

        return ((all_min_x, all_min_y), (all_max_x, all_max_y))

    def to_shapely(self) -> Shapely:
        """Convert to shapely geometry containing all instances.

        Returns a GeometryCollection or MultiPolygon of all
        translated (and optionally rotated) base shapes.
        """
        base_shapely = self.base_shape.to_shapely()
        geometries = []

        for i, (px, py) in enumerate(self.positions):
            # Start with base shape
            instance = base_shapely

            # Apply rotation if specified
            if self.rotations is not None:
                angle = self.rotations[i]
                if not np.isclose(angle, 0.0):
                    # Rotate around origin (base shape's center)
                    instance = shapely.affinity.rotate(
                        instance, angle, origin=(0, 0), use_radians=False
                    )

            # Translate to position
            instance = shapely.affinity.translate(instance, px, py)
            geometries.append(instance)

        # Combine into single geometry
        if len(geometries) == 1:
            return geometries[0]

        # Try to create MultiPolygon if all are polygons
        if all(g.geom_type == "Polygon" for g in geometries):
            return shapely.MultiPolygon(geometries)

        return shapely.GeometryCollection(geometries)

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to 3D geometry group containing all instances.

        Each instance is wrapped in a Transformed to apply its
        translation and rotation.

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane. Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Default is 0.0.

        Returns
        -------
        GeometryGroup
            A group containing Transformed copies of the base 3D geometry.
        """
        # Convert base shape to 3D once
        base_3d = self.base_shape.to_3d_geometry(slab_bounds, axis, sidewall_angle)

        geometries = []

        for i, (px, py) in enumerate(self.positions):
            # Build 2D transform
            if self.rotations is not None and not np.isclose(self.rotations[i], 0.0):
                angle = math.radians(self.rotations[i])
                # Rotate around origin, then translate
                transform_2d = np.dot(
                    np.array(Transformed2D.translation(px, py)),
                    np.array(Transformed2D.rotation(angle)),
                )
            else:
                # Just translation
                transform_2d = np.array(Transformed2D.translation(px, py))

            # Expand to 3D transform
            transform_3d = self._expand_2d_to_3d_transform(transform_2d, axis)

            # Wrap base geometry in Transformed
            geometries.append(
                Transformed(geometry=base_3d, transform=transform_3d.tolist())
            )

        return GeometryGroup(geometries=tuple(geometries))

    @staticmethod
    def _expand_2d_to_3d_transform(mat_2d: np.ndarray, axis: Axis) -> np.ndarray:
        """Expand 2D (3x3) transform to 3D (4x4) transform.

        Parameters
        ----------
        mat_2d : np.ndarray
            3x3 homogeneous 2D transformation matrix.
        axis : Axis
            The axis perpendicular to the 2D plane.

        Returns
        -------
        np.ndarray
            4x4 homogeneous 3D transformation matrix.
        """
        mat_4x4 = np.eye(4)

        # Determine which 2D indices map to which 3D indices
        if axis == 2:
            idx = [0, 1]  # X, Y
        elif axis == 1:
            idx = [0, 2]  # X, Z
        else:  # axis == 0
            idx = [1, 2]  # Y, Z

        # Copy 2x2 rotation/scale block
        mat_4x4[idx[0], idx[0]] = mat_2d[0, 0]
        mat_4x4[idx[0], idx[1]] = mat_2d[0, 1]
        mat_4x4[idx[1], idx[0]] = mat_2d[1, 0]
        mat_4x4[idx[1], idx[1]] = mat_2d[1, 1]

        # Copy translation
        mat_4x4[idx[0], 3] = mat_2d[0, 2]
        mat_4x4[idx[1], 3] = mat_2d[1, 2]

        return mat_4x4

    @classmethod
    def grid(
        cls,
        base_shape: "Geometry2DType",
        columns: int,
        rows: int,
        spacing: tuple[float, float],
        origin: Coordinate2D = (0.0, 0.0),
        rotation: float = 0.0,
    ) -> "Array2D":
        """Create a regular grid array.

        Parameters
        ----------
        base_shape : Geometry2DType
            The shape to repeat.
        columns : int
            Number of columns (x direction).
        rows : int
            Number of rows (y direction).
        spacing : tuple[float, float]
            Spacing between instances as (x_spacing, y_spacing).
        origin : Coordinate2D
            Origin of the grid (position of instance [0, 0]).
        rotation : float
            Uniform rotation for all instances (degrees).

        Returns
        -------
        Array2D
            Grid array with columns * rows instances.

        Example
        -------
        >>> grid = Array2D.grid(
        ...     base_shape=Circle2D(center=(0, 0), radius=0.1),
        ...     columns=10,
        ...     rows=10,
        ...     spacing=(1.0, 1.0),
        ...     origin=(-4.5, -4.5),
        ... )
        """
        if columns <= 0 or rows <= 0:
            raise SetupError("columns and rows must be positive integers.")

        ox, oy = origin
        sx, sy = spacing

        positions = tuple(
            (ox + i * sx, oy + j * sy)
            for j in range(rows)
            for i in range(columns)
        )

        rotations = None
        if not np.isclose(rotation, 0.0):
            rotations = tuple(rotation for _ in range(len(positions)))

        return cls(
            base_shape=base_shape,
            positions=positions,
            rotations=rotations,
        )

