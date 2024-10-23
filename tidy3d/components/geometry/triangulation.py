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

    convexity: float = 0.0

    is_ear: bool = False


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
        -1 if vertex was collinear, +1 if it became collinear, 0 otherwise.
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
    vertices = [Vertex(v, i) for i, v in enumerate(vertices)]
    if not is_ccw:
        vertices.reverse()

    collinears = 0
    for i in range(len(vertices)):
        update_convexity(vertices, i)
        if vertices[i].convexity == 0.0:
            collinears += 1

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
