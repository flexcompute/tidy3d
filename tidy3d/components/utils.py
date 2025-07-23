"""Utilities shared by multiple components go here."""

from __future__ import annotations

from typing import Any, Optional

import shapely

from tidy3d.components.types import Shapely
from tidy3d.log import log


def pop_axis_and_swap(
    coord: tuple[Any, Any, Any], axis: int, transpose: bool = False
) -> tuple[Any, tuple[Any, Any]]:
    """
    ``pop_axis_and_swap()`` is identical to ``Geometry.pop_axis()``, except that it accepts
    an additional ``transpose`` argument which reverses the output order.  Examples:

    ``pop_axis_and_swap(("x", "y", "z"), 1, transpose=False)``  ->  ``("y", ("x", "z"))``
    ``pop_axis_and_swap(("x", "y", "z"), 1, transpose=True)``   ->  ``("y", ("z", "x"))``

    Parameters
    ----------
    coord : Tuple[Any, Any, Any]
        Tuple of three values in original coordinate system.
    axis : int
        Integer index into 'xyz' (0,1,2).
    transpose : bool = False
        Optional: Swap the order of the data from the two remaining axes in the output tuple.

    Returns
    -------
    Any, Tuple[Any, Any]
        The input coordinates are separated into the one along the axis provided
        and the two on the planar coordinates,
        like ``axis_coord, (planar_coord1, planar_coord2)``.
    """
    plane_vals = list(coord)
    axis_val = plane_vals.pop(axis)
    if transpose:
        plane_vals = [plane_vals[1], plane_vals[0]]
    return axis_val, tuple(plane_vals)


def unpop_axis_and_swap(
    ax_coord: Any,
    plane_coords: tuple[Any, Any],
    axis: int,
    transpose: bool = False,
) -> tuple[Any, Any, Any]:
    """
    ``unpop_axis_and_swap()`` is identical to ``Geompetry.unpop_axis()``, except that
    it accepts an additional ``transpose`` argument which reverses the order of
    ``plane_coords`` before sending them to ``unpop_axis()``.  For example:

    ``unpop_axis_and_swap("y", ("x", "z"), 1, transpose=False)``  -->  ``("x", "y", "z")``
    ``unpop_axis_and_swap("y", ("x", "z"), 1, transpose=True)``   -->  ``("z", "y", "x")``

    This function is the inverse of ``pop_axis_and_swap()``.  For example:
    ``unpop_axis_and_swap("y", ("z", "x"), 1, transpose=True)``   -->  ``("x", "y", "z")``

    Parameters
    ----------
    ax_coord : Any
        Value along axis direction.
    plane_coords : Tuple[Any, Any]
        Values along ordered planar directions.
    axis : int
        Integer index into 'xyz' (0,1,2).
    transpose : bool = False
        Optional: Swap the order of the entries in plane_coords[].
        (This overrides the default ascending axis order.)

    Returns
    -------
    Tuple[Any, Any, Any]
        The three values in the xyz coordinate system.
    """
    coords = list(plane_coords)
    if transpose:
        coords = [coords[1], coords[0]]
    coords.insert(axis, ax_coord)
    return tuple(coords)


def shape_swap_xy(shape: Shapely) -> Shapely:
    """Create a new version of a shapely object with the X and Y coordinates swapped.
    IMPORTANT: This does not work if any of the coordinates are infinite."""
    # Define the transformation matrix for swapping X and Y coords.  For details, see:
    # https://shapely.readthedocs.io/en/stable/manual.html#affine-transformations
    transform_matrix = (0, 1, 1, 0, 0, 0)
    shape_new = shapely.affinity.affine_transform(shape, transform_matrix)
    return shape_new


def warn_untested_argument(cls_name: Optional[str], func_name: str, arg: str, val: str):
    """Generic warning message if a function has never been manually tested with ``arg=val``. (This"
    is typically used for plot functions where manual tests and visual confirmation is needed.)"""
    prefix = ""
    if cls_name:
        prefix = cls_name + "."
    log.warning(
        f"UNTESTED!  The `{prefix}{func_name}()` function has not yet been tested with `{arg}={val}`.",
        log_once=True,
    )
