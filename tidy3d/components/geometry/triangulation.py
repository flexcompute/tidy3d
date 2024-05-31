from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import shapely

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
    is_convex : bool = False
        Flag indicating whether this is a convex vertex in the polygon.
    is_ear : bool = False
        Flag indicating whether this is an ear of the polygon.
    """

    coordinate: ArrayFloat1D

    index: int

    is_convex: bool = False

    is_ear: bool = False


def update_convexity(vertices: List[Vertex], i: int) -> None:
    """Update the convexity of a vertex in a polygon.

    Parameters
    ----------
    vertices : List[Vertex]
        Vertices of the polygon.
    i : int
        Index of the vertex to be updated.
    """
    j = (i + 1) % len(vertices)
    vertices[i].is_convex = (
        np.cross(
            vertices[i].coordinate - vertices[i - 1].coordinate,
            vertices[j].coordinate - vertices[i].coordinate,
        )
        > 0
    )


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
    vertices[i].is_ear = vertices[i].is_convex and not any(
        is_inside(v.coordinate, triangle)
        for k, v in enumerate(vertices)
        if not (v.is_convex or k == h or k == i or k == j)
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

    for i in range(len(vertices)):
        update_convexity(vertices, i)

    for i in range(len(vertices)):
        update_ear_flag(vertices, i)

    triangles = []

    while len(vertices) > 3:
        i = 0
        while i < len(vertices):
            if vertices[i].is_ear:
                j = (i + 1) % len(vertices)
                triangles.append((vertices[i - 1].index, vertices[i].index, vertices[j].index))
                vertices.pop(i)
                if len(vertices) == 3:
                    break
                h = (i - 1) % len(vertices)
                j = i % len(vertices)
                if not vertices[h].is_convex:
                    update_convexity(vertices, h)
                if not vertices[j].is_convex:
                    update_convexity(vertices, j)
                update_ear_flag(vertices, h)
                update_ear_flag(vertices, j)
            else:
                i += 1

    triangles.append(tuple(v.index for v in vertices))
    return triangles
