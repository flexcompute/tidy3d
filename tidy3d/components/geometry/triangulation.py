from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.errors import ShapelyError

from tidy3d.exceptions import Tidy3dError

if TYPE_CHECKING:
    from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D


TRIANGULATION_ERROR = "Impossible to triangulate polygon. Verify that the polygon is valid."


def cross2(a: ArrayFloat1D, b: ArrayFloat1D) -> float:
    """Return the scalar 2D cross product."""
    return a[0] * b[1] - a[1] * b[0]


def _coordinate_key(coordinate: ArrayFloat1D) -> tuple[float, float]:
    """Return a hashable coordinate key."""
    return (float(coordinate[0]), float(coordinate[1]))


def _orient_ccw(triangle: tuple[int, int, int], coordinates: np.ndarray) -> tuple[int, int, int]:
    """Orient a triangle counter-clockwise."""
    if cross2(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]) < 0:
        return (triangle[0], triangle[2], triangle[1])
    return triangle


def _triangulate_shapely(vertices: ArrayFloat2D) -> list[tuple[int, int, int]]:
    """Triangulate using GEOS constrained Delaunay triangulation."""
    index_by_coordinate = {_coordinate_key(vertex): index for index, vertex in enumerate(vertices)}
    if len(index_by_coordinate) != len(vertices):
        raise ValueError("Polygon has repeated vertices.")

    polygon = shapely.Polygon(vertices)
    if not polygon.is_valid:
        raise ValueError("Polygon is invalid.")

    triangles = []
    for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(polygon)):
        coordinates = np.asarray(triangle.exterior.coords[:-1])
        if len(coordinates) != 3:
            raise ValueError("Triangulation returned a non-triangular polygon.")
        try:
            indices = tuple(
                index_by_coordinate[_coordinate_key(coordinate)] for coordinate in coordinates
            )
        except KeyError as exc:
            raise ValueError(
                f"Triangulation returned a vertex not present in the input polygon: {exc}."
            ) from exc
        triangles.append(_orient_ccw(indices, coordinates))

    if len(triangles) != len(vertices) - 2:
        raise ValueError("Triangulation did not use all polygon vertices.")
    return triangles


def triangulate(vertices: ArrayFloat2D) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon.

    Parameters
    ----------
    vertices : ArrayFloat2D
        Vertices of the polygon.

    Returns
    -------
    list[tuple[int, int, int]]
       List of indices of the vertices of the triangles.
    """
    try:
        return _triangulate_shapely(vertices)
    except (ShapelyError, KeyError, ValueError) as exc:
        raise Tidy3dError(f"{TRIANGULATION_ERROR} Reason: {exc}") from exc
