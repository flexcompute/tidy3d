"""Utilities for base geometry classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shapely

from tidy3d.compat import _package_is_older_than
from tidy3d.log import log

from .constants import POLY_DISTANCE_TOLERANCE, POLY_TOLERANCE_RATIO

if TYPE_CHECKING:
    from tidy3d.components.types import (
        Shapely,
    )


def cleanup_shapely_object(obj: Shapely, tolerance_ratio: float = POLY_TOLERANCE_RATIO) -> Shapely:
    """Remove small geometric features from the boundaries of a shapely object including
    inward and outward spikes, thin holes, and thin connections between larger regions.

    Parameters
    ----------
    obj : shapely
        a shapely object (typically a ``Polygon`` or a ``MultiPolygon``)
    tolerance_ratio : float = ``POLY_TOLERANCE_RATIO``
        Features on the boundaries of polygons will be discarded if they are smaller
        or narrower than ``tolerance_ratio`` multiplied by the size of the object.

    Returns
    -------
    Shapely
        A new shapely object whose small features (eg. thin spikes or holes) are removed.

    Notes
    -----
    This function does not attempt to delete overlapping, nearby, or collinear vertices.
    To solve that problem, use ``shapely.simplify()`` afterwards.
    """
    if _package_is_older_than("shapely", "2.1"):
        log.warning("Versions of shapely prior to v2.1 may cause plot errors.", log_once=True)
        return obj
    if obj.is_empty:
        return obj
    centroid = obj.centroid
    object_size = min(obj.bounds[2] - obj.bounds[0], obj.bounds[3] - obj.bounds[1])
    if object_size == 0.0:
        return shapely.Polygon([])

    # To prevent numerical over- or underflow errors, subtract the centroid and rescale
    normalized_obj = shapely.affinity.affine_transform(
        obj,
        matrix=[
            1 / object_size,
            0.0,
            0.0,
            1 / object_size,
            -centroid.x / object_size,
            -centroid.y / object_size,
        ],
    )
    # Important: Remove any self intersections beforehand using `shapely.make_valid()`.
    valid_obj = shapely.make_valid(normalized_obj, method="structure", keep_collapsed=False)

    # To get rid of small thin features, erode(shrink), dilate(expand), and erode again.
    eroded_obj = shapely.buffer(
        valid_obj,
        distance=-tolerance_ratio,
        cap_style="square",
        quad_segs=3,
    )
    dilated_obj = shapely.buffer(
        eroded_obj,
        distance=2 * tolerance_ratio,
        cap_style="square",
        quad_segs=3,
    )
    cleaned_obj = dilated_obj

    # Optional: Now shrink the polygon back to the original size.
    cleaned_obj = shapely.buffer(
        cleaned_obj,
        distance=-tolerance_ratio,
        cap_style="square",
        quad_segs=3,
    )
    # Clean vertices of very close distances created during the erosion/dilation process.
    # The distance value is heuristic.
    cleaned_obj = cleaned_obj.simplify(POLY_DISTANCE_TOLERANCE, preserve_topology=True)
    # Revert to the original scale and position.
    rescaled_clean_obj = shapely.affinity.affine_transform(
        cleaned_obj,
        matrix=[
            object_size,
            0.0,
            0.0,
            object_size,
            centroid.x,
            centroid.y,
        ],
    )
    return rescaled_clean_obj
