from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import shapely

from ...exceptions import Tidy3dError
from ..types import ArrayFloat1D, ArrayFloat2D


@dataclass
class Vertex:
    """Simple data class to hold triangulation data structures.

    Parameters
    ----------
    coordinate: ArrayFloat1D
        Vertex coordinate.
    index : int
        Vertex index in the original polygon.
    convexity : float = 0.0
        Value representing the convexity (> 0) or concavity (< 0) of the vertex in the polygon.
    is_ear : bool = False
        Flag indicating whether this is an ear of the polygon.
    """

    coordinate: ArrayFloat1D

    index: int

    convexity: float

    is_ear: bool


def update_convexity(vertices: List[Vertex], i: int) -> int:
    """Update the convexity of a vertex in a polygon.

    Parameters
    ----------
    vertices : List[Vertex]
        Vertices of the polygon.
    i : int
        Index of the vertex to be updated.

    Returns
    -------
    int
        Value indicating vertex convexity change w.r.t. 0. See note below.

    Note
    ----
    Besides updating the vertex, this function returns a value indicating whether the updated vertex
    convexity changed to or from 0 (0 convexity means the vertex is collinear with its neighbors).
    If the convexity changes from zero to non-zero, return -1. If it changes from non-zero to zero,
    return +1. Return 0 in any other case. This allows the main triangulation loop to keep track of
    the total number of collinear vertices in the polygon.

    """
    result = -1 if vertices[i].convexity == 0.0 else 0
    j = (i + 1) % len(vertices)
    vertices[i].convexity = np.cross(
        vertices[i].coordinate - vertices[i - 1].coordinate,
        vertices[j].coordinate - vertices[i].coordinate,
    )
    if vertices[i].convexity == 0.0:
        result += 1
    return result


def is_inside(
    vertex: ArrayFloat1D, triangle: Tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D]
) -> bool:
    """Check if a vertex is inside a triangle.

    Parameters
    ----------
    vertex : ArrayFloat1D
        Vertex coordinates.
    triangle : Tuple[ArrayFloat1D, ArrayFloat1D, ArrayFloat1D]
        Vertices of the triangle in CCW order.

    Returns
    -------
    bool:
        Flag indicating if the vertex is inside the triangle.
    """
    return all(
        np.cross(triangle[i] - triangle[i - 1], vertex - triangle[i - 1]) > 0 for i in range(3)
    )


def update_ear_flag(vertices: List[Vertex], i: int) -> None:
    """Update the ear flag of a vertex in a polygon.

    Parameters
    ----------
    vertices : List[Vertex]
        Vertices of the polygon.
    i : int
        Index of the vertex to be updated.
    """
    h = (i - 1) % len(vertices)
    j = (i + 1) % len(vertices)
    triangle = (vertices[h].coordinate, vertices[i].coordinate, vertices[j].coordinate)
    vertices[i].is_ear = vertices[i].convexity > 0 and not any(
        is_inside(v.coordinate, triangle)
        for k, v in enumerate(vertices)
        if not (v.convexity > 0 or k == h or k == i or k == j)
    )


# TODO: This is an inefficient algorithm that runs in O(n^2). We should use something
# better, and probably as a compiled extension.
def triangulate(vertices: ArrayFloat2D) -> List[Tuple[int, int, int]]:
    """Triangulate a simple polygon.

    Parameters
    ----------
    vertices : ArrayFloat2D
        Vertices of the polygon.

    Returns
    -------
    List[Tuple[int, int, int]]
       List of indices of the vertices of the triangles.
    """
    is_ccw = shapely.LinearRing(vertices).is_ccw

    # Initialize vertices as non-collinear because we will update the actual value below and count
    # the number of collinear vertices.
    vertices = [Vertex(v, i, -1.0, False) for i, v in enumerate(vertices)]
    if not is_ccw:
        vertices.reverse()

    collinears = 0
    for i in range(len(vertices)):
        collinears += update_convexity(vertices, i)

    for i in range(len(vertices)):
        update_ear_flag(vertices, i)

    triangles = []

    ear_found = True
    while len(vertices) > 3:
        if not ear_found:
            raise Tidy3dError(
                "Impossible to triangulate polygon. Verify that the polygon is valid."
            )
        ear_found = False
        i = 0
        while i < len(vertices):
            if vertices[i].is_ear:
                removed = vertices.pop(i)
                h = (i - 1) % len(vertices)
                j = i % len(vertices)
                collinears += update_convexity(vertices, h)
                collinears += update_convexity(vertices, j)
                if collinears == len(vertices):
                    # Undo removal because only collinear vertices remain
                    vertices.insert(i, removed)
                    collinears += update_convexity(vertices, (i - 1) % len(vertices))
                    collinears += update_convexity(vertices, (i + 1) % len(vertices))
                    i += 1
                else:
                    ear_found = True
                    triangles.append((vertices[h].index, removed.index, vertices[j].index))
                    update_ear_flag(vertices, h)
                    update_ear_flag(vertices, j)
                    if len(vertices) == 3:
                        break
            else:
                i += 1

    triangles.append(tuple(v.index for v in vertices))
    return triangles
