"""Tests 2d corner finder."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.grid.corner_finder import CornerFinderSpec
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.grid.grid_spec import GAP_MESHING_TOL, GridRefinement, LayerRefinementSpec

CORNER_FINDER = CornerFinderSpec()
GRID_REFINEMENT = GridRefinement()
LAYER_REFINEMENT = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2))
LAYER2D_REFINEMENT = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 0))


def _contains_point(points, target, atol=1e-6):
    """Whether a 2D point list contains ``target`` within tolerance."""
    return any(np.allclose(point, target, atol=atol) for point in points)


def _contains_optional_point(points, target, atol=1e-6):
    """Whether a 3D optional coordinate list contains ``target`` within tolerance."""
    for point in points:
        match = True
        for value, expected in zip(point, target):
            if value is None or expected is None:
                match &= value is None and expected is None
            else:
                match &= np.isclose(value, expected, atol=atol)
        if match:
            return True
    return False


def _assert_same_point_set(actual, expected, atol=1e-6):
    """Assert that two 2D point sets match up to reordering."""
    assert len(actual) == len(expected)
    for point in actual:
        assert _contains_point(expected, point, atol=atol)
    for point in expected:
        assert _contains_point(actual, point, atol=atol)


def _corners_for_vertices(vertices, collapse_extent=None):
    """Detect corners for one 2D polygon vertex list."""
    polyslab = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]
    finder = CORNER_FINDER
    if collapse_extent is not None:
        finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=collapse_extent)
    return finder.corners(normal_axis=2, coord=0, structure_list=structures)


def _assert_rounded_case_matches_sharp_baseline(
    sharp_vertices, rounded_vertices, expected_corner, collapse_extent, should_recover
):
    """Compare rounded-corner recovery against the corresponding sharp-corner baseline."""
    sharp_corners = _corners_for_vertices(sharp_vertices)
    assert _contains_point(sharp_corners, expected_corner) == should_recover

    rounded_corners = _corners_for_vertices(rounded_vertices)
    assert not _contains_point(rounded_corners, expected_corner)

    recovered_corners = _corners_for_vertices(rounded_vertices, collapse_extent=collapse_extent)
    assert _contains_point(recovered_corners, expected_corner) == should_recover
    _assert_same_point_set(recovered_corners, sharp_corners)


def _sharp_rectangle_vertices(xmax=4.0, ymax=4.0):
    """Axis-aligned rectangle with sharp corners."""
    return [(0.0, 0.0), (xmax, 0.0), (xmax, ymax), (0.0, ymax)]


def _rounded_convex_corner_vertices(radius=1.0, xmax=4.0, ymax=4.0, num_arc_pts=16):
    """Rectangle with the lower-left corner replaced by a quarter-circle chain."""
    arc = [
        (radius + radius * np.cos(t), radius + radius * np.sin(t))
        for t in np.linspace(np.pi, 1.5 * np.pi, num_arc_pts)
    ]
    return [*arc, (xmax, 0.0), (xmax, ymax), (0.0, ymax)]


def _rounded_top_right_corner_vertices(xmax=3.0, ymax=3.0, radius=1.0, num_arc_pts=16):
    """Rectangle with the upper-right corner replaced by a quarter-circle chain."""
    cx, cy = xmax - radius, ymax - radius
    arc = [
        (cx + radius * np.cos(t), cy + radius * np.sin(t))
        for t in np.linspace(0, 0.5 * np.pi, num_arc_pts)
    ]
    return [
        (0.0, 0.0),
        (xmax, 0.0),
        (xmax, ymax - radius),
        *arc[1:-1],
        (xmax - radius, ymax),
        (0.0, ymax),
    ]


def _rounded_convex_corner_split_support_vertices(radius=1.0, xmax=4.0, ymax=4.0, num_arc_pts=16):
    """Rounded lower-left corner with extra collinear vertices on the adjacent straight supports."""
    arc = [
        (radius + radius * np.cos(t), radius + radius * np.sin(t))
        for t in np.linspace(np.pi, 1.5 * np.pi, num_arc_pts)
    ]
    return [
        *arc,
        (1.5, 0.0),
        (2.5, 0.0),
        (xmax, 0.0),
        (xmax, ymax),
        (0.0, ymax),
        (0.0, 2.5),
        (0.0, 1.5),
    ]


def _chamfered_convex_corner_vertices(offset=1.0, xmax=4.0, ymax=4.0):
    """Rectangle with the lower-left corner replaced by a single chamfer segment."""
    return [(offset, 0.0), (xmax, 0.0), (xmax, ymax), (0.0, ymax), (0.0, offset)]


def _rounded_concave_notch_vertices(
    xmax=4.0, ymax=4.0, notch_x=2.0, notch_y=2.0, radius=1.0, num_arc_pts=16
):
    """L-shaped polygon with a rounded concave notch corner at ``(notch_x, notch_y)``."""
    cx, cy = notch_x + radius, notch_y + radius
    arc = [
        (cx + radius * np.cos(t), cy + radius * np.sin(t))
        for t in np.linspace(1.5 * np.pi, np.pi, num_arc_pts)
    ]
    return [
        (0.0, 0.0),
        (xmax, 0.0),
        (xmax, notch_y),
        (notch_x + radius, notch_y),
        *arc[1:-1],
        (notch_x, notch_y + radius),
        (notch_x, ymax),
        (0.0, ymax),
    ]


def _sharp_concave_notch_vertices(xmax=4.0, ymax=4.0, notch_x=2.0, notch_y=2.0):
    """L-shaped polygon with a sharp concave notch corner at ``(notch_x, notch_y)``."""
    return [
        (0.0, 0.0),
        (xmax, 0.0),
        (xmax, notch_y),
        (notch_x, notch_y),
        (notch_x, ymax),
        (0.0, ymax),
    ]


def _rotate_vertices(vertices, angle, center=(0.0, 0.0)):
    """Rotate a 2D vertex list about ``center``."""
    center = np.asarray(center, dtype=float)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    return [
        tuple(rotation @ (np.asarray(vertex, dtype=float) - center) + center) for vertex in vertices
    ]


def _shear_vertices(vertices, shear):
    """Apply the linear map ``(x, y) -> (x + shear * y, y)``."""
    return [(x + shear * y, y) for x, y in vertices]


def test_2dcorner_finder_filter_collinear_vertex():
    """In corner finder, test that collinear vertices are filtered"""
    # 2nd and 3rd vertices are on a collinear line
    vertices = ((0, 0), (0.1, 0), (0.5, 0), (1, 0), (1, 1))
    polyslab = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 3

    # if angle threshold is 0, collinear vertex will not be filtered
    corner_finder = CORNER_FINDER.updated_copy(angle_threshold=0)
    corners = corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 5


def test_2dcorner_finder_filter_nearby_vertex():
    """In corner finder, test that vertices that are very close are filtered"""
    # filter duplicate vertices
    vertices = ((0, 0), (0, 0), (1e-4, -1e-4), (1, 0), (1, 1))
    polyslab = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 4

    # filter very close vertices
    corner_finder = CORNER_FINDER.updated_copy(distance_threshold=2e-4)
    corners = corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 3


def test_2dcorner_finder_medium():
    """No corner found if the medium is dielectric while asking to search for corner of metal."""
    structures = [td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 0


def test_2dcorner_finder_polygon_with_hole():
    """Find corners related to interior holes of a polygon."""
    structures = [
        td.Structure(geometry=td.Box(size=(2, 2, 2)), medium=td.PEC),
        td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium()),
    ]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    # 4 interior, 4 exterior
    assert len(corners) == 8


def test_2dcorner_finder_detect_rounded_convex_corner():
    """Collapse a small rounded convex chain into one synthetic corner."""
    polyslab = td.PolySlab(
        vertices=_rounded_convex_corner_vertices(radius=0.5),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, (0.0, 0.0))
    assert len(corners) == 3

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=0.55)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, (0.0, 0.0))
    assert not _contains_point(corners, (0.0, 0.5))
    assert not _contains_point(corners, (0.5, 0.0))
    assert len(corners) == 4


def test_2dcorner_finder_detect_flipped_rounded_corner():
    """Detect the diagonally flipped rounded corner and avoid the inner false positive."""
    polyslab = td.PolySlab(
        vertices=_rounded_top_right_corner_vertices(),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, (3.0, 3.0))
    assert len(corners) == 3

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.05)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, (3.0, 3.0))
    assert not _contains_point(corners, (2.0, 2.0))
    assert len(corners) == 4


def test_2dcorner_finder_detect_chamfered_corner():
    """Collapse a small chamfer into the same synthetic sharp corner."""
    polyslab = td.PolySlab(
        vertices=_chamfered_convex_corner_vertices(),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, (0.0, 0.0))
    assert len(corners) == 5

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.05)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, (0.0, 0.0))
    assert not _contains_point(corners, (0.0, 1.0))
    assert not _contains_point(corners, (1.0, 0.0))
    assert len(corners) == 4


def test_2dcorner_finder_detect_rotated_rounded_convex_corner():
    """Recover a rounded convex corner whose support edges are rotated off the global axes."""
    angle = np.pi / 6
    expected_corner = _rotate_vertices([(0.0, 0.0)], angle)[0]
    polyslab = td.PolySlab(
        vertices=_rotate_vertices(_rounded_convex_corner_vertices(radius=0.5), angle),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, expected_corner)
    assert len(corners) == 3

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=0.55)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, expected_corner)
    assert len(corners) == 4


def test_2dcorner_finder_rounded_collapse_is_ring_order_invariant():
    """Recover the same rounded corner even when the ring starts inside the rounded chain."""
    angle = np.pi / 4
    vertices = _rotate_vertices(_rounded_convex_corner_vertices(radius=0.5), angle)
    shifted_vertices = vertices[3:] + vertices[:3]
    ring = np.array([*shifted_vertices, shifted_vertices[0]], dtype=float)

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=0.55)
    corners, _ = rounded_corner_finder._filter_collinear_vertices(
        rounded_corner_finder._collapse_rounded_corners(ring)
    )

    expected_corners = _rotate_vertices([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)], angle)
    assert len(corners) == len(expected_corners)
    for expected_corner in expected_corners:
        assert _contains_point(corners, expected_corner)


def test_2dcorner_finder_detect_rotated_chamfered_corner():
    """Recover a chamfered corner whose support edges are rotated off the global axes."""
    angle = -np.pi / 5
    expected_corner = _rotate_vertices([(0.0, 0.0)], angle)[0]
    polyslab = td.PolySlab(
        vertices=_rotate_vertices(_chamfered_convex_corner_vertices(), angle),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, expected_corner)
    assert len(corners) == 5

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.05)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, expected_corner)
    assert len(corners) == 4


def test_2dcorner_finder_detect_non_right_rounded_convex_corner_consistent_with_legacy():
    """Recover a rounded 135-degree corner exactly when the sharp legacy corner is detected."""
    shear = -1.0
    expected_corner = _shear_vertices([(0.0, 0.0)], shear)[0]
    _assert_rounded_case_matches_sharp_baseline(
        sharp_vertices=_shear_vertices(_sharp_rectangle_vertices(), shear),
        rounded_vertices=_shear_vertices(_rounded_convex_corner_vertices(radius=0.5), shear),
        expected_corner=expected_corner,
        collapse_extent=0.8,
        should_recover=True,
    )


def test_2dcorner_finder_detect_rounded_corner_with_split_support_runs():
    """Recover the rounded corner when adjacent straight supports are split into collinear segments."""
    angle = np.pi / 6
    expected_corner = _rotate_vertices([(0.0, 0.0)], angle)[0]
    vertices = _rotate_vertices(_rounded_convex_corner_split_support_vertices(radius=0.5), angle)
    shifted_vertices = vertices[6:] + vertices[:6]
    polyslab = td.PolySlab(vertices=shifted_vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, expected_corner)
    assert len(corners) == 3

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=0.55)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, expected_corner)
    assert len(corners) == 4


def test_2dcorner_finder_does_not_invent_corners_on_circle():
    """Do not synthesize corners on a smooth curve with no straight support runs."""
    circle = td.Cylinder(radius=1.0, axis=2, length=2.0)
    structures = [td.Structure(geometry=circle, medium=td.PEC)]

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.1)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 0


def test_2dcorner_finder_detect_rounded_concave_corner():
    """Collapse a rounded concave notch back to the missing sharp corner."""
    polyslab = td.PolySlab(
        vertices=_rounded_concave_notch_vertices(),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, (2.0, 2.0))
    assert len(corners) == 5

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.05)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, (2.0, 2.0))
    assert len(corners) == 6


def test_2dcorner_finder_detect_rotated_rounded_concave_corner():
    """Recover a rounded concave notch whose support edges are rotated off the global axes."""
    angle = np.pi / 7
    expected_corner = _rotate_vertices([(2.0, 2.0)], angle)[0]
    polyslab = td.PolySlab(
        vertices=_rotate_vertices(_rounded_concave_notch_vertices(), angle),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, expected_corner)
    assert len(corners) == 5

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=1.05)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert _contains_point(corners, expected_corner)
    assert len(corners) == 6


def test_2dcorner_finder_detect_non_right_rounded_concave_corner_consistent_with_legacy():
    """Recover a rounded 225-degree notch exactly when the sharp legacy corner is detected."""
    shear = -1.0
    expected_corner = _shear_vertices([(2.0, 2.0)], shear)[0]
    _assert_rounded_case_matches_sharp_baseline(
        sharp_vertices=_shear_vertices(_sharp_concave_notch_vertices(), shear),
        rounded_vertices=_shear_vertices(_rounded_concave_notch_vertices(), shear),
        expected_corner=expected_corner,
        collapse_extent=1.5,
        should_recover=True,
    )


def test_2dcorner_finder_respects_rounded_corner_extent():
    """Do not synthesize a corner when the rounded chain exceeds the allowed extent."""
    polyslab = td.PolySlab(
        vertices=_rounded_convex_corner_vertices(radius=1.0),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]

    rounded_corner_finder = CORNER_FINDER.updated_copy(corner_rounding_collapse_extent=0.5)
    corners = rounded_corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert not _contains_point(corners, (0.0, 0.0))
    assert len(corners) == 3


def test_2dcorner_finder_does_not_detect_shallow_rounded_bend_below_angle_threshold():
    """Do not recover a rounded bend whose sharp turn would fail the legacy angle threshold."""
    shear = -np.sqrt(3.0)
    expected_corner = _shear_vertices([(0.0, 0.0)], shear)[0]
    _assert_rounded_case_matches_sharp_baseline(
        sharp_vertices=_shear_vertices(_sharp_rectangle_vertices(), shear),
        rounded_vertices=_shear_vertices(_rounded_convex_corner_vertices(radius=0.5), shear),
        expected_corner=expected_corner,
        collapse_extent=1.4,
        should_recover=False,
    )


def test_polygons_from_merged_geos_filters_by_medium():
    """``_polygons_from_merged_geos`` keeps disjoint polygons matching ``CornerFinderSpec.medium``.

    Lossy metal counts as metal alongside PEC; ``medium="metal"`` drops dielectric while
    ``medium="all"`` keeps it.
    """
    pec_left = td.Structure(geometry=td.Box(center=(-3, 0, 0), size=(1, 1, 2)), medium=td.PEC)
    pec_right = td.Structure(geometry=td.Box(center=(3, 0, 0), size=(1, 1, 2)), medium=td.PEC)
    lossy = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 2)),
        medium=td.LossyMetalMedium(conductivity=1.0, frequency_range=(1e14, 2e14)),
    )
    dielectric = td.Structure(
        geometry=td.Box(center=(0, 4, 0), size=(1, 1, 2)), medium=td.Medium(permittivity=4)
    )

    merged = CornerFinderSpec._merged_pec_on_plane(
        normal_axis=2,
        coord=0,
        structure_list=[pec_left, pec_right, lossy, dielectric],
        interior_disjoint_geometries=True,
    )

    # medium="metal": three disjoint metal regions (two PEC + one lossy metal); dielectric excluded
    metal_polygons = CornerFinderSpec(medium="metal")._polygons_from_merged_geos(merged)
    assert len(metal_polygons) == 3
    centroids_x = sorted(poly.centroid.x for poly in metal_polygons)
    assert np.allclose(centroids_x, [-3, 0, 3], atol=1e-6)

    # medium="all": the dielectric region is kept too
    assert len(CornerFinderSpec(medium="all")._polygons_from_merged_geos(merged)) == 4


# --- in-plane edge refinement -------------------------------------------------


def test_corner_finder_axis_aligned_angle_threshold_field():
    """``axis_aligned_angle_threshold`` rejects values outside ``[0, pi/4)``."""
    with pytest.raises(ValidationError):
        CornerFinderSpec(axis_aligned_angle_threshold=np.pi / 4)
    with pytest.raises(ValidationError):
        CornerFinderSpec(axis_aligned_angle_threshold=-1e-3)


def test_in_plane_edge_refinement_effective_spec():
    """``_edge_refinement`` resolves ``mirror_corner``/``None``/explicit spec correctly."""
    base = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2))
    # mirror_corner resolves to corner_refinement
    assert base._edge_refinement == base.corner_refinement
    # None disables edge refinement
    assert base.updated_copy(in_plane_edge_refinement=None)._edge_refinement is None
    # explicit GridRefinement is used as-is
    explicit = GridRefinement(dl=0.01, num_cells=4)
    assert base.updated_copy(in_plane_edge_refinement=explicit)._edge_refinement == explicit
    # mirror_corner with no corner_refinement to mirror -> off
    assert base.updated_copy(corner_refinement=None)._edge_refinement is None
    # corner_finder off disables edge refinement even with an explicit edge spec
    assert (
        base.updated_copy(corner_finder=None, in_plane_edge_refinement=explicit)._edge_refinement
        is None
    )


def test_axis_unaligned_run_grouping():
    """Maximal runs of axis-unaligned segments are grouped; axis-aligned segments break runs."""
    layer = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2))

    # axis-aligned square: every segment is aligned -> no unaligned runs
    assert layer.corner_finder._axis_unaligned_runs([(0, 0), (4, 0), (4, 4), (0, 4)]) == []

    # 45-degree diamond: every segment is slanted -> a single run spanning the ring
    runs = layer.corner_finder._axis_unaligned_runs([(0, 1), (1, 0), (0, -1), (-1, 0)])
    assert len(runs) == 1
    assert np.isclose(runs[0][:, 0].min(), -1) and np.isclose(runs[0][:, 0].max(), 1)

    # one slanted edge between axis-aligned edges -> a single run over that edge's vertices
    runs = layer.corner_finder._axis_unaligned_runs([(0, 0), (4, 0), (4, 4), (2, 5), (0, 4)])
    assert len(runs) == 1
    assert np.isclose(runs[0][:, 1].min(), 4) and np.isclose(runs[0][:, 1].max(), 5)

    # arrowhead: a run straddling the ring-closure index (the dip at (2,-1)) is one run, not two
    runs = layer.corner_finder._axis_unaligned_runs([(0, 0), (2, -1), (4, 0), (4, 4), (0, 4)])
    assert len(runs) == 1
    assert np.isclose(runs[0][:, 1].min(), -1)


def test_axis_aligned_angle_threshold_tunes_edge_classification():
    """The tunable threshold decides how close to an axis a segment counts as aligned."""
    h = 4 * np.tan(np.deg2rad(0.5))  # bottom edge tilted 0.5deg off the x-axis
    shape = [(0, 0), (4, h), (4, 4), (0, 4)]

    coarse = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=CornerFinderSpec(axis_aligned_angle_threshold=np.deg2rad(1.0)),
    )
    assert len(coarse.corner_finder._axis_unaligned_runs(shape)) == 0  # 0.5deg < 1deg -> aligned

    fine = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=CornerFinderSpec(axis_aligned_angle_threshold=np.deg2rad(0.1)),
    )
    assert len(fine.corner_finder._axis_unaligned_runs(shape)) == 1  # 0.5deg > 0.1deg -> slanted


_EDGE_SIM_BOUNDS = [[-td.inf] * 3, [td.inf] * 3]
_EDGE_BOUNDARY_TYPES = [[None] * 2] * 3


def _pec_polyslab_structure(vertices):
    """A z-normal PEC PolySlab structure from in-plane vertices."""
    poly = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    return td.Structure(geometry=poly, medium=td.PEC)


def _diamond_structure():
    """45-degree rotated square: every in-plane edge is slanted."""
    return _pec_polyslab_structure([(0, 1), (1, 0), (0, -1), (-1, 0)])


def _inplane_overrides(layer, structure, grid_size_in_vacuum=1.0):
    """In-plane (normal-axis dl=None) override structures emitted by a layer."""
    overrides = layer.generate_override_structures(
        grid_size_in_vacuum=grid_size_in_vacuum,
        structure_list=[structure],
        sim_bounds=_EDGE_SIM_BOUNDS,
        boundary_type=_EDGE_BOUNDARY_TYPES,
    )
    return [o for o in overrides if o.dl[2] is None]


def test_edge_refinement_requires_corner_finder():
    """Edge refinement is off when ``corner_finder`` is ``None``, regardless of the edge spec."""
    layer = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=None,
        in_plane_edge_refinement=GridRefinement(dl=0.05, num_cells=3),
    )
    assert layer._edge_refinement is None
    assert _inplane_overrides(layer, _diamond_structure()) == []


def test_edge_refinement_override_on_slanted_polygon():
    """A slanted polygon's edges drive one padded override at the edge grid size."""
    edge_refinement = GridRefinement(dl=0.05, num_cells=3)
    layer = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_refinement=None,  # corner overrides resolve to the edge grid size and union in
        in_plane_edge_refinement=edge_refinement,
    )
    inplane = _inplane_overrides(layer, _diamond_structure())
    assert len(inplane) == 1
    override = inplane[0]
    assert not override.shadow
    assert np.isclose(override.dl[0], edge_refinement.dl)
    assert np.isclose(override.dl[1], edge_refinement.dl)
    margin = edge_refinement.num_cells * edge_refinement.dl
    # diamond bbox is [-1, 1] x [-1, 1], grown by num_cells * dl per axis
    assert np.isclose(override.geometry.size[0], 2 + margin)
    assert np.isclose(override.geometry.size[1], 2 + margin)


def test_suggested_dl_min_includes_edge_refinement():
    """``suggested_dl_min`` accounts for the effective edge grid size (finer of corner/edge)."""
    layer = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_refinement=GridRefinement(dl=0.2),
        in_plane_edge_refinement=GridRefinement(dl=0.05),
    )
    dl_min = layer.suggested_dl_min(
        grid_size_in_vacuum=1.0,
        structures=[],
        sim_bounds=_EDGE_SIM_BOUNDS,
        boundary_type=_EDGE_BOUNDARY_TYPES,
    )
    assert np.isclose(dl_min, 0.05)


# --- union of overlapping override structures ---------------------------------


def test_same_dl_overrides_union_by_overlap():
    """Same-grid-size overrides merge per connected component: overlapping -> one box, disjoint -> separate."""
    # overlapping: a diamond's 4 corner + 1 edge overrides are all at the same dl and touch -> one bbox
    corner_refinement = GridRefinement(dl=0.1)
    overlapping = LayerRefinementSpec(
        axis=2, size=(td.inf, td.inf, 2), corner_refinement=corner_refinement
    )
    inplane = _inplane_overrides(overlapping, _diamond_structure())
    assert len(inplane) == 1
    assert np.isclose(inplane[0].dl[0], corner_refinement.dl)
    assert np.isclose(inplane[0].dl[1], corner_refinement.dl)
    # union bbox is the grown diamond edge run (mirrored from corner_refinement)
    margin = corner_refinement.num_cells * corner_refinement.dl
    assert np.isclose(inplane[0].geometry.size[0], 2 + margin)

    # disjoint: two far-apart squares (edge off) keep all 8 tiny corner boxes as separate components
    disjoint = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2), in_plane_edge_refinement=None)
    left = _pec_polyslab_structure([(-22, -2), (-18, -2), (-18, 2), (-22, 2)])
    right = _pec_polyslab_structure([(18, -2), (22, -2), (22, 2), (18, 2)])
    overrides = disjoint.generate_override_structures(
        grid_size_in_vacuum=1.0,
        structure_list=[left, right],
        sim_bounds=_EDGE_SIM_BOUNDS,
        boundary_type=_EDGE_BOUNDARY_TYPES,
    )
    assert len([o for o in overrides if o.dl[2] is None]) == 8


def test_same_dl_overrides_union_iterates_over_bbox_collapse():
    """Collapsing a component to its bbox can overlap boxes it never touched directly, so the
    union must iterate to a fixpoint, not stop after one connected-components pass.

    A truncated-corner hexagon has two long diagonal edges whose runs span opposite halves and
    overlap near the center: their bounding boxes merge into one whole-shape box that then overlaps
    the two right-angle corner boxes (which touch neither diagonal run). A single pass leaves three
    overlapping boxes; the iterated union merges all of them into one.
    """
    corner_refinement = GridRefinement(dl=0.1)
    layer = LayerRefinementSpec(
        axis=2, size=(td.inf, td.inf, 2), corner_refinement=corner_refinement
    )
    hexagon = _pec_polyslab_structure(
        [(-10, -10), (-1, -10), (10, 1), (10, 10), (1, 10), (-10, -1)]
    )
    inplane = _inplane_overrides(layer, hexagon)
    # one box, not the three a single connected-components pass would leave
    assert len(inplane) == 1
    # it spans the whole hexagon footprint (20) grown by the corner margin on each merged side
    margin = corner_refinement.num_cells * corner_refinement.dl
    assert np.isclose(inplane[0].geometry.size[0], 20 + margin)
    assert np.isclose(inplane[0].geometry.size[1], 20 + margin)


def test_different_dl_overrides_kept_separate():
    """Corner and edge overrides at different grid sizes are not merged across grid sizes."""
    layer = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_refinement=GridRefinement(dl=0.05, num_cells=3),  # corner finer than edge
        in_plane_edge_refinement=GridRefinement(dl=0.2, num_cells=3),
    )
    inplane = _inplane_overrides(layer, _diamond_structure())
    corner_boxes = [o for o in inplane if np.isclose(o.dl[0], 0.05)]
    edge_boxes = [o for o in inplane if np.isclose(o.dl[0], 0.2)]
    # both grid sizes survive; nothing merged across grid sizes
    assert len(corner_boxes) == 4  # four disjoint corner boxes (do not overlap each other)
    assert len(edge_boxes) == 1  # one slanted-edge run spanning the diamond
    assert len(inplane) == 5


# --- small-geometry resolution (min_steps_per_geometry) -----------------------
#
# Small-geometry resolution is a post-mesh *measurement* pass: it counts
# cells of the constructed grid across each disjoint metal geometry and refines only the axes that
# fall below ``min_steps_per_geometry``. It is not a pre-mesh override source, so it does not enter
# ``generate_override_structures`` / the override union, nor the static ``suggested_dl_min`` bound.


def _small_geometry_layer(**kwargs):
    """A layer that isolates small-geometry resolution (no corner snapping/refinement, no edges).

    Small-geometry resolution requires ``corner_finder``, so keep it but turn off every other
    consumer (snapping, corner refinement, edge refinement).
    """
    defaults = {
        "axis": 2,
        "size": (td.inf, td.inf, 2),
        "corner_finder": CornerFinderSpec(),
        "corner_snapping": False,
        "corner_refinement": None,
        "in_plane_edge_refinement": None,
        "min_steps_per_geometry": 2,
    }
    defaults.update(kwargs)
    return LayerRefinementSpec(**defaults)


def _uniform_inplane_grid(dl=1.0, span=5.0):
    """A z-normal grid with uniform in-plane spacing ``dl`` over ``[-span, span]``.

    Grid boundaries sit on integer multiples of ``dl``, so a measurement pass counts them
    deterministically against a geometry's bounding box.
    """
    coords = np.arange(-span, span + dl / 2, dl)
    return Grid(boundaries=Coords(x=coords, y=coords, z=np.array([-1.0, 1.0])))


def _pec_merged_geos(bbox):
    """A single-PEC-box merged-geometry list from ``(umin, vmin, umax, vmax)``."""
    return [(td.PEC, shapely.box(*bbox))]


def test_min_steps_per_geometry_field():
    """``min_steps_per_geometry`` must be positive, and ``None`` disables it."""
    LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2), min_steps_per_geometry=None)
    with pytest.raises(ValidationError):
        LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2), min_steps_per_geometry=0)
    with pytest.raises(ValidationError):
        LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2), min_steps_per_geometry=-1)


@pytest.mark.parametrize(
    "bbox, expected_dl",
    [
        # uniform dl=1 grid with boundaries on the integers; an axis is refined to extent/2 iff it
        # fully contains fewer than min_steps_per_geometry (=2) cells, i.e. fewer than 3 grid
        # boundaries fall in the bbox along it (cells contained = boundaries in bbox - 1).
        ((0, 0, 0.1, 0.1), (0.05, 0.05)),  # tiny: 1 boundary -> 0 cells per axis -> both refined
        ((0, 0, 0.1, 4.0), (0.05, None)),  # thin trace: x 0 cells, y 4 cells -> narrow x only
        ((0, 0, 1.0, 1.0), (0.5, 0.5)),  # 2 boundaries -> 1 cell per axis (< 2) -> refined to 1.0/2
        ((0, 0, 2.0, 2.0), (None, None)),  # 3 boundaries -> 2 cells per axis (>= 2) -> resolved
        ((0, 0, 4.0, 4.0), (None, None)),  # large: 5 boundaries -> 4 cells per axis -> resolved
    ],
)
def test_small_geometry_measurement_per_axis(bbox, expected_dl):
    """A geometry's in-plane axis is refined to ``extent/min_steps`` iff the constructed grid fully
    contains fewer than ``min_steps`` cells across it; resolved axes/geometries are left untouched."""
    grid = _uniform_inplane_grid(dl=1.0)
    overrides, dl_min = _small_geometry_layer()._small_geometry_measurement_overrides(
        grid, _pec_merged_geos(bbox)
    )
    if all(dl is None for dl in expected_dl):
        assert overrides == [] and dl_min == td.inf
        return
    assert len(overrides) == 1
    override = overrides[0]
    assert not override.shadow and override.priority == -1
    assert override.dl[2] is None  # normal axis untouched
    umin, vmin, umax, vmax = bbox
    assert np.isclose(override.geometry.size[0], umax - umin)  # override spans the geometry bbox
    assert np.isclose(override.geometry.size[1], vmax - vmin)
    for axis2d, expected in enumerate(expected_dl):
        if expected is None:
            assert override.dl[axis2d] is None
        else:
            assert np.isclose(override.dl[axis2d], expected)
    assert np.isclose(dl_min, min(dl for dl in expected_dl if dl is not None))


def test_small_geometry_skips_near_zero_extent():
    """A near-zero in-plane extent (e.g. a Shapely sliver) is skipped rather than refined to a
    vanishing ``dl``; a genuinely under-resolved axis on the same geometry is still refined."""
    grid = _uniform_inplane_grid(dl=1.0)
    # x extent below GAP_MESHING_TOL -> skipped; y spans 1 cell (< min_steps=2) -> refined to 0.5
    bbox = (0.0, 0.0, GAP_MESHING_TOL / 10, 1.0)
    overrides, dl_min = _small_geometry_layer()._small_geometry_measurement_overrides(
        grid, _pec_merged_geos(bbox)
    )
    assert len(overrides) == 1
    override = overrides[0]
    assert override.dl[0] is None  # sliver x axis skipped, not refined to extent/min_steps
    assert np.isclose(override.dl[1], 0.5)  # under-resolved y axis still refined
    assert np.isclose(dl_min, 0.5)


def test_small_geometry_requires_corner_finder():
    """Small-geometry resolution is off when ``corner_finder`` is ``None``."""
    grid = _uniform_inplane_grid(dl=1.0)
    layer = _small_geometry_layer(corner_finder=None)
    overrides, dl_min = layer._small_geometry_measurement_overrides(
        grid, _pec_merged_geos((0, 0, 0.1, 0.1))
    )
    assert overrides == [] and dl_min == td.inf


def test_min_steps_per_geometry_resolves_small_via_in_grid():
    """End-to-end: the measurement pass refines a small under-resolved via to >= min_steps cells."""
    via = _pec_polyslab_structure([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
    layer = _small_geometry_layer()

    def build(min_steps):
        return td.Simulation(
            size=(4, 4, 2),
            grid_spec=td.GridSpec.auto(
                wavelength=2.0,
                layer_refinement_specs=[layer.updated_copy(min_steps_per_geometry=min_steps)],
            ),
            run_time=1e-13,
            structures=[via],
        )

    def cells_across(sim, axis2d):
        centers = np.asarray(sim.grid.centers.to_list[axis2d])
        return int(np.sum((centers >= 0) & (centers <= 0.1)))

    sim_on = build(2)
    sim_off = build(None)
    assert cells_across(sim_on, 0) >= 2 and cells_across(sim_on, 1) >= 2
    # without the feature the via is under-resolved, so turning it on strictly adds cells
    assert cells_across(sim_on, 0) > cells_across(sim_off, 0)


def test_small_geometry_rebuild_keeps_internal_overrides_on_none_path():
    """The post-mesh small-geometry rebuild must keep the layer corner/edge overrides even when the
    caller passes ``internal_override_structures=None`` (the ``make_grid`` default), so the grid
    matches the one built from explicitly supplied overrides instead of dropping in-plane
    refinement and applying only the small-geometry boxes."""
    sim_structure = td.Structure(geometry=td.Box(size=(8, 8, 2)), medium=td.Medium())
    # a small under-resolved via triggers the rebuild; a separate metal block contributes corner
    # overrides that visibly change the grid, so dropping them would change the result
    via = _pec_polyslab_structure([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
    block = _pec_polyslab_structure([(-3, -3), (-1, -3), (-1, -1), (-3, -1)])
    structures = [sim_structure, via, block]
    layer = LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=CornerFinderSpec(),
        corner_snapping=False,
        corner_refinement=GridRefinement(dl=0.05, num_cells=2),
        min_steps_per_geometry=2,
    )
    grid_spec = td.GridSpec.auto(wavelength=2.0, layer_refinement_specs=[layer])
    build_kwargs = {
        "structures": structures,
        "symmetry": (0, 0, 0),
        "periodic": (False, False, False),
        "sources": [],
        "num_pml_layers": ((0, 0), (0, 0), (0, 0)),
    }
    boundary_types = [[None, None]] * 3
    internal_overrides = grid_spec.internal_override_structures(
        structures, grid_spec.get_wavelength([]), sim_structure.geometry.bounds, (), boundary_types
    )
    grid_explicit, _ = grid_spec._make_grid_and_snapping_lines(
        internal_override_structures=internal_overrides, **build_kwargs
    )
    grid_none, _ = grid_spec._make_grid_and_snapping_lines(
        internal_override_structures=None, **build_kwargs
    )
    assert grid_none == grid_explicit


def test_from_layer_bounds_new_inplane_fields():
    """``from_layer_bounds`` plumbs both new in-plane fields."""
    spec = GridRefinement(dl=0.05)
    layer = LayerRefinementSpec.from_layer_bounds(
        axis=2, bounds=(0, 1), in_plane_edge_refinement=spec, min_steps_per_geometry=4
    )
    assert layer.in_plane_edge_refinement == spec
    assert layer.min_steps_per_geometry == 4


def test_from_bounds_and_structures_new_inplane_fields():
    """``from_bounds`` and ``from_structures`` plumb both new in-plane fields."""
    spec = GridRefinement(dl=0.05)
    via = _pec_polyslab_structure([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
    explicit = (
        LayerRefinementSpec.from_bounds(
            rmin=(0, 0, 0),
            rmax=(1, 1, 1),
            in_plane_edge_refinement=spec,
            min_steps_per_geometry=4,
        ),
        LayerRefinementSpec.from_structures(
            structures=[via], in_plane_edge_refinement=spec, min_steps_per_geometry=4
        ),
    )
    for layer in explicit:
        assert layer.in_plane_edge_refinement == spec
        assert layer.min_steps_per_geometry == 4


def test_gridrefinement():
    """Test GradRefinement is working as expected."""

    # generate override structures for z-axis
    center = [None, None, 0]
    size = (0, 0, 0)  # refine around a point
    grid_size_in_vaccum = 1
    structure = GRID_REFINEMENT.override_structure(center, size, grid_size_in_vaccum, True)
    assert not structure.shadow
    for axis in range(2):
        assert structure.dl[axis] is None
        assert structure.geometry.size[axis] == td.inf
    dl = grid_size_in_vaccum / GRID_REFINEMENT._refinement_factor
    assert np.isclose(structure.dl[2], dl)
    assert np.isclose(structure.geometry.size[2], dl * GRID_REFINEMENT.num_cells)

    # explicitly define step size in refinement region that is smaller than that of refinement_factor
    dl = 1
    grid_refinement = GRID_REFINEMENT.updated_copy(dl=dl)
    structure = grid_refinement.override_structure(center, size, grid_size_in_vaccum, True)
    for axis in range(2):
        assert structure.dl[axis] is None
        assert structure.geometry.size[axis] == td.inf
    assert np.isclose(structure.dl[2], dl)
    assert np.isclose(structure.geometry.size[2], dl * GRID_REFINEMENT.num_cells)


def test_layerrefinement():
    """Test LayerRefinementSpec is working as expected."""

    # size along axis must be inf
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec(axis=0, size=(td.inf, 0, 0))

    # classmethod
    for axis in range(3):
        layer = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, 1))
        assert layer.center[axis] == 0.5
        assert layer.size[axis] == 1
        assert layer.size[(axis + 1) % 3] == td.inf
        assert layer.size[(axis + 2) % 3] == td.inf
        assert not layer._is_inplane_bounded(layer)

    layer = LayerRefinementSpec.from_bounds(axis=axis, rmin=(0, 0, 0), rmax=(1, 2, 3))
    layer = LayerRefinementSpec.from_bounds(rmin=(0, 0, 0), rmax=(1, 2, 3))
    assert layer.axis == 0
    assert np.isclose(layer.length_axis, 1)
    assert np.isclose(layer.center_axis, 0.5)
    assert layer._is_inplane_bounded(layer)

    # from structures
    structures = [td.Structure(geometry=td.Box(size=(td.inf, 2, 3)), medium=td.Medium())]
    layer = LayerRefinementSpec.from_structures(structures)
    assert layer._is_inplane_bounded(layer)
    assert layer.axis == 1

    with pytest.raises(ValidationError):
        structures = [
            td.Structure(geometry=td.Box(size=(td.inf, td.inf, td.inf)), medium=td.Medium())
        ]
        layer = LayerRefinementSpec.from_structures(structures)

    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, td.inf))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(td.inf, 0))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(-td.inf, 0))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(1, -1))


def test_layerrefinement_inplane_inside():
    # inplane inside
    layer = LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(0, 1))
    assert not layer._is_inplane_bounded(layer)
    assert layer._inplane_inside(layer, [3e3, 4e4])
    layer = LayerRefinementSpec(axis=1, size=(1, 0, 1))
    assert layer._inplane_inside(layer, [0, 0])
    assert not layer._inplane_inside(layer, [2, 0])


def test_layerrefinement_snapping_points():
    """Test snapping points for LayerRefinementSpec is working as expected."""

    # snapping points for layer bounds
    points = LAYER2D_REFINEMENT._snapping_points_along_axis
    assert len(points) == 1
    assert points[0] == (None, None, 0)

    points = LAYER_REFINEMENT._snapping_points_along_axis
    assert len(points) == 1
    assert points[0] == (None, None, -1)


def test_layerrefinement_detect_rounded_corner():
    """LayerRefinementSpec should expose the synthetic rounded corner as a snapping point."""
    polyslab = td.PolySlab(
        vertices=_rounded_convex_corner_vertices(),
        axis=2,
        slab_bounds=[-1, 1],
    )
    structure = td.Structure(geometry=polyslab, medium=td.PEC)
    sim_bounds = [
        [-td.inf] * 3,
        [td.inf] * 3,
    ]
    boundary_types = [[None] * 2] * 3

    layer_without_rounding = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
    )
    points = layer_without_rounding.generate_snapping_points(
        [structure], sim_bounds, boundary_types
    )
    assert not _contains_optional_point(points, (0.0, 0.0, None))

    layer_with_rounding = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(corner_rounding_collapse_extent=1.05),
    )
    points = layer_with_rounding.generate_snapping_points([structure], sim_bounds, boundary_types)
    assert _contains_optional_point(points, (0.0, 0.0, None))

    override_structures = layer_with_rounding.generate_override_structures(
        grid_size_in_vacuum=1.0,
        structure_list=[structure],
        sim_bounds=sim_bounds,
        boundary_type=boundary_types,
    )

    def _covers_origin(override):
        """Whether the override's in-plane footprint covers the collapsed corner at (0, 0)."""
        center, size = override.geometry.center, override.geometry.size
        return all(
            abs(center[axis]) <= size[axis] / 2 or np.isclose(abs(center[axis]), size[axis] / 2)
            for axis in range(2)
        )

    # the collapsed corner region is refined; with edge refinement + union the covering override
    # is a bounding box that need not be centered exactly on the corner
    assert any(_covers_origin(override) for override in override_structures)


def test_grid_spec_with_layers():
    """Test the application of layer_specs to GridSpec."""

    thickness = 1e-3
    lumped_elements = []
    # a PEC thin layer structure
    box1 = td.Box(size=(thickness, 2, 2))
    box2 = td.Box(center=(0, -1, 0), size=(thickness, 1, 1))
    pec_str = td.Structure(geometry=box1 - box2, medium=td.PEC)
    # a pin
    pin_str = td.Structure(
        geometry=td.Cylinder(axis=0, radius=0.1, length=1.1, center=(-0.5, 0, 0)), medium=td.PEC
    )
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ]
    )
    assert layer.axis == 0

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer]
        ),
        boundary_spec=td.BoundarySpec.pml(),
        structures=[pec_str, pin_str],
        run_time=1e-12,
    )
    # lower bound is snapped
    assert any(np.isclose(sim.grid.boundaries.x, -thickness / 2))
    # corner snapped
    assert any(np.isclose(sim.grid.boundaries.y, -0.5))
    assert any(np.isclose(sim.grid.boundaries.z, -0.5))
    assert any(np.isclose(sim.grid.boundaries.z, 0.5))

    # differnt laye parameters
    def update_sim_with_newlayer(layer):
        return sim.updated_copy(
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer]
            )
        )

    # bounds snapping
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_snapping="bounds",
    )
    sim2 = update_sim_with_newlayer(layer)
    assert any(np.isclose(sim2.grid.boundaries.x, -thickness / 2))
    assert any(np.isclose(sim2.grid.boundaries.x, thickness / 2))

    # layer thickness refinement
    def count_grids_within_layer(sim_t):
        float_relax = 1.001
        x = sim_t.grid.boundaries.x
        x = x[x >= -thickness / 2 * float_relax]
        x = x[x <= thickness / 2 * float_relax]
        return len(x)

    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) == 4

    # layer thickness refinement + bounds refinement, but the latter is too coarse so that it's abandoned
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
        bounds_refinement=td.GridRefinement(),
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) == 4
    # much finer refinement to be included
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
        bounds_refinement=td.GridRefinement(dl=thickness / 10),
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) > 10

    # layer bounds refinement: combined into one structure when they overlap
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_refinement=td.GridRefinement(dl=thickness * 1.1, num_cells=1),
        corner_finder=None,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert (
        len(
            sim2.grid_spec.all_override_structures(
                list(sim2.structures),
                1.0,
                lumped_elements,
                sim2._internal_layerrefinement_boundary_types,
                sim2.bounds,
            )
        )
        == 1
    )

    # separate when they don't overlap
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_refinement=td.GridRefinement(dl=thickness * 0.9, num_cells=1),
        corner_finder=None,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert (
        len(
            sim2.grid_spec.all_override_structures(
                list(sim2.structures),
                1.0,
                lumped_elements,
                sim2._internal_layerrefinement_boundary_types,
                sim2.bounds,
            )
        )
        == 2
    )


def test_grid_spec_with_layers_interior_disjoint():
    """Test the application of layer_specs with interior_disjoint assumption to GridSpec."""

    thickness = 1e-3
    lumped_elements = []
    # a PEC thin layer structure
    box1 = td.Box(size=(thickness, 2, 2))
    box2 = td.Box(center=(0, -1, 0), size=(thickness, 1, 1))
    pec_str = td.Structure(geometry=box1 - box2, medium=td.PEC)
    # a dielectric that overlaps with PEC structure, hence breaking interior-disjoint assumption
    die_str = td.Structure(
        geometry=td.Box(size=(thickness, 0.5, 0.5)),
        medium=td.Medium(),
    )
    # layer spec assuming interior disjoint geometries
    layer_disjoint = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        interior_disjoint_geometries=True,
    )
    # layer spec for general geometries
    layer_general = layer_disjoint.updated_copy(interior_disjoint_geometries=False)

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer_disjoint]
        ),
        boundary_spec=td.BoundarySpec.pml(),
        structures=[pec_str],
        run_time=1e-12,
    )
    sim_general = sim.updated_copy(
        grid_spec=sim.grid_spec.updated_copy(layer_refinement_specs=[layer_general])
    )

    # for simulations with interior-disjoint geometries, same corner finder results for both layer settings
    def is_equal_snapping_points(plist1, plist2):
        if len(plist1) != len(plist2):
            return False
        for p1, p2 in zip(plist1, plist2):
            if p1 != p2:
                return False
        return True

    assert is_equal_snapping_points(
        sim.internal_snapping_points, sim_general.internal_snapping_points
    )

    # for simulations with overlapping geometries, assuming interior-disjoint misses some corners in this simulation
    sim = sim.updated_copy(structures=[pec_str, die_str])
    sim_general = sim_general.updated_copy(structures=[pec_str, die_str])
    assert len(sim_general.internal_snapping_points) > len(sim.internal_snapping_points)


@pytest.mark.parametrize("gap_meshing_iters", [0, 1])
def test_corner_refinement_outside_domain(gap_meshing_iters):
    """Test the behavior of corner refinement if corners are outside the simulation domain."""

    # CPW waveguides that goes through the simulation domain along x-axis, so that the corners
    # are outside the simulation domain.
    wg_length = 10
    wg1 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1)),
        medium=td.PEC,
    )

    wg2 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1), center=(0, 0.25, 0)),
        medium=td.PEC,
    )

    wg3 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1), center=(0, -0.25, 0)),
        medium=td.PEC,
    )
    structures = [wg1, wg2, wg3]

    # 1) not refined in the gap along y-axis because corners are outside the simulation domain
    layer = td.LayerRefinementSpec.from_structures(
        structures,
        axis=2,
        corner_refinement=td.GridRefinement(refinement_factor=5),
        refinement_inside_sim_only=True,
        gap_meshing_iters=gap_meshing_iters,
    )

    sim = td.Simulation(
        size=(2, 2, 2),
        structures=structures,
        grid_spec=td.GridSpec.auto(
            wavelength=1,
            layer_refinement_specs=[layer],
        ),
        run_time=1e-20,
    )

    def count_grids_within_gap(sim_t):
        float_relax = 1.001
        y = sim_t.grid.boundaries.y
        y = y[y * float_relax >= 0.1]
        y = y[y <= 0.15 * float_relax]
        return len(y)

    # just 2 grids sampling the gap
    assert count_grids_within_gap(sim) == gap_meshing_iters + 2

    # 2) refined if corners outside simulation domain is accounted for.
    layer = layer.updated_copy(refinement_inside_sim_only=False)
    sim = sim.updated_copy(grid_spec=td.GridSpec.auto(wavelength=1, layer_refinement_specs=[layer]))

    assert count_grids_within_gap(sim) > 2


def test_dl_min_from_smallest_feature():
    structure = td.Structure(
        geometry=td.PolySlab(
            vertices=[
                [0, 0],
                [2, 0],
                [2, 1],
                [1, 1],
                [1, 1.1],
                [2, 1.1],
                [2, 2],
                [1, 2],
                [1, 2.2],
                [0.7, 2.2],
                [0.7, 2],
                [0, 2],
            ],
            slab_bounds=[-1, 1],
            axis=2,
        ),
        medium=td.PECMedium(),
    )
    boundary_types = [[None] * 2] * 3
    sim_bounds = [
        [-td.inf] * 3,
        [td.inf] * 3,
    ]
    # check expected dl_min
    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(
            convex_resolution=10,
        ),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.3 / 10, dl_min)

    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(mixed_resolution=10),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.2 / 10, dl_min)

    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(
            concave_resolution=10,
        ),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.1 / 10, dl_min)

    # check grid is generated succesfully
    sim = td.Simulation(
        size=(5, 5, 5),
        structures=[structure],
        grid_spec=td.GridSpec.auto(layer_refinement_specs=[layer_spec], wavelength=100 * td.C_0),
        run_time=1e-20,
    )

    _ = sim.grid


def test_gap_meshing():
    w = 1
    length = 10

    l_shape_1 = td.Structure(
        medium=td.PECMedium(),
        geometry=td.PolySlab(
            axis=2,
            slab_bounds=[0, 2],
            vertices=[
                [0, 0],
                [length, 0],
                [length, w],
                [w, w],
                [w, length],
                [0, length],
            ],
        ),
    )

    gap = 0.1
    l_shape_2 = td.Structure(
        medium=td.PECMedium(),
        geometry=td.PolySlab(
            axis=2,
            slab_bounds=[0, 2],
            vertices=[
                [w + gap, w + gap],
                [w + gap + length, w + gap],
                [w + gap + length, w + gap + w],
                [w + gap + w, w + gap + w],
                [w + gap + w, w + gap + length],
                [w + gap, w + gap + length],
                [0, w + gap + length],
                [0, gap + length],
                [w + gap, gap + length],
            ],
        ),
    )

    # ax = l_shape_1.plot(z=1)
    # l_shape_2.plot(z=1, ax=ax)

    for num_iters in range(2):
        grid_spec = td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=10),
            grid_y=td.AutoGrid(min_steps_per_wvl=10),
            grid_z=td.AutoGrid(min_steps_per_wvl=10),
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=2,
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=num_iters,
                    size=[td.inf, td.inf, 2],
                    center=[0, 0, 1],
                )
            ],
            wavelength=7,
        )

        sim = td.Simulation(
            structures=[l_shape_1, l_shape_2],
            grid_spec=grid_spec,
            size=(1.2 * length, 1.2 * length, 2),
            center=(0.5 * length, 0.5 * length, 0),
            run_time=1e-15,
        )

        # ax = sim.plot(z=1)
        # sim.plot_grid(z=1, ax=ax)
        # ax.set_xlim([0, 2])
        # ax.set_ylim([0, 2])

        resolved_x = np.any(
            np.logical_and(sim.grid.boundaries.x > w, sim.grid.boundaries.x < w + gap)
        )
        resolved_y = np.any(
            np.logical_and(sim.grid.boundaries.y > w, sim.grid.boundaries.y < w + gap)
        )

        if num_iters == 0:
            assert (not resolved_x) and (not resolved_y)
        else:
            assert resolved_x and resolved_y

    # test ingored small feature
    sim_size = (1, 1, 1)
    box = td.Structure(
        geometry=td.Box(size=(0.95, 0.2, 0.95)),
        medium=td.PECMedium(),
    )

    reentry_gap = td.Structure(
        geometry=td.PolySlab(
            slab_bounds=(-0.2, 0.2),
            axis=1,
            vertices=[(-0.3, 0.52), (-0.05, 0.3), (0.2, 0.52)],
        ),
        medium=td.Medium(),
    )

    aux = td.Structure(
        geometry=td.Box(center=(0.0, 0, 0.5), size=(2, 0.4, 0.23)),
        medium=td.Medium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[box, reentry_gap, aux],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    assert not any(
        np.isclose(sim.grid.boundaries.x, reentry_gap.geometry.bounding_box.center[0], rtol=1e-2)
    )
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test internal polygon

    gap_z = td.Structure(
        geometry=td.Box(center=(0.3, 0, 0.1), size=(0.05, 0.4, 0.5)),
        medium=td.Medium(),
    )

    gap_x = td.Structure(
        geometry=td.Box(center=(0.03, 0, 0.07), size=(0.3, 0.4, 0.05)),
        medium=td.Medium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[box, gap_x, gap_z],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=2,
                    dl_min_from_gap_width=True,
                    interior_disjoint_geometries=False,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    assert any(np.isclose(sim.grid.boundaries.x, gap_z.geometry.center[0], atol=1e-4))
    assert any(np.isclose(sim.grid.boundaries.z, gap_x.geometry.center[2], atol=1e-4))
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test gaps near pec/pmc
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert any(np.isclose(sim.grid.boundaries.x, -expected_grid_line))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert any(np.isclose(sim.grid.boundaries.z, expected_grid_line))
    assert any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test limited size of layer spec
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.5, 0.2, 0.5),
                    center=(0.5, 0, -0.5),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # pretty much zero size layer spec
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.05, 0.2, 0.05),
                    center=(0.05, 0, 0.05),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=2,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert not any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert not any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # layer spec is outside
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.01, 0.2, 0.01),
                    center=(10.0, 0, 10.0),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert not any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert not any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test gap near periodic
    sim = td.Simulation(
        size=sim_size,
        center=(0.025, -0.25, 0),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.pec(),
            z=td.Boundary.periodic(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    assert any(np.isclose(sim.grid.boundaries.x, 0.5, atol=1e-4))
    assert any(np.isclose(sim.grid.boundaries.z, -0.5, atol=1e-4))

    # test a thin strip near periodic
    strip = td.Structure(
        geometry=td.Box(center=(0, 0.5, 0), size=(0.2, 0.05, 0.6)),
        medium=td.PECMedium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pmc(),
        ),
        structures=[strip],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=0,
                    size=(0.2, td.inf, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    assert any(np.isclose(sim.grid.boundaries.y, strip.geometry.center[1], atol=1e-4))
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(x=0, ax=ax)
    # sim.plot_grid(x=0, ax=ax)
    # plt.show()


def test_gap_meshing_skip_small_gap():
    """When the gap is very small, make sure it's skipped."""

    f0 = 7e9

    mm = 1000  # Conversion mm to micron
    H = 0.8 * mm  # Substrate thickness
    T = 0.035 * mm  # Metal thickness

    # Resonator dimensions
    MA, MB, MC, MD = (3.9 * mm, 7.1 * mm, 3.1 * mm, 2.3 * mm)
    ME, MF, MG, MH = (0.6 * mm, 0.2 * mm, 1.2 * mm, 0.5 * mm)
    MJ, MK, MM, MN = (4.8 * mm, 0.3 * mm, 0.1 * mm, 0.7 * mm)
    MP, MQ, MR, MS = (0.1 * mm, 0.7 * mm, 0.4 * mm, 0.3 * mm)
    Lsub, Wsub = (2 * MC + MH, 2 * (MH + MK + MB))

    geom_patch = td.Box.from_bounds(
        rmin=(-MA / 2, MH / 2 + MK, 0), rmax=(MA / 2, MH / 2 + MK + MB, T)
    )
    geom_hole1 = td.Box.from_bounds(
        rmin=(-MH / 2 - MN - MF - ME, MH / 2 + MK + MS, 0),
        rmax=(-MH / 2 - MN - MF, MH / 2 + MK + MS + MG, T),
    )
    geom_hole5 = td.Box.from_bounds(
        rmin=(-MA / 2 + 1.5 * MF, MH / 2 + MK + MS + MG + MQ, 0),
        rmax=(-MA / 2 + 1.5 * MF + MM, MH / 2 + MK + MB - MP, T),
    )
    geom_hole6 = geom_hole5.translated(-2 * geom_hole5.center[0], 0, 0)
    geom_hole7 = td.Box.from_bounds(
        rmin=(-MH / 2 - MN, MH / 2 + MK, 0), rmax=(MH / 2 + MN, MH / 2 + MD, T)
    )
    for hole in [geom_hole1, geom_hole5, geom_hole6, geom_hole7]:
        geom_patch -= hole

    x0, y0, z0 = geom_patch.bounding_box.center
    struct_patch = td.Structure(geometry=geom_patch, medium=td.PEC)

    # Add padding
    padding = td.C_0 / f0 / 2
    sim_LX = Lsub + padding
    sim_LY = Wsub + padding
    sim_LZ = H + padding

    # Layer refinement on resonator
    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[struct_patch],
        min_steps_along_axis=1,
        corner_refinement=td.GridRefinement(dl=T, num_cells=2),
        dl_min_from_gap_width=True,
    )

    # Define overall grid spec
    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f0,
        min_steps_per_wvl=12,
        layer_refinement_specs=[lr_spec],
    )

    # Define simulation object
    sim = td.Simulation(
        center=(x0, y0, z0),
        size=(sim_LX, sim_LY, sim_LZ),
        structures=[struct_patch],
        grid_spec=grid_spec,
        run_time=1e-9,
    )
    assert sim.grid_info["min_grid_size"] > 20


def test_gap_meshing_tiny_nearly_parallel():
    """Test that tiny and nearly parallel features are handled properly. That is,
    even though they are ignored, they are still taken into account when removing
    reentry features."""

    sim_size = (1, 1, 1)

    diff = td.Structure(
        geometry=td.PolySlab(
            slab_bounds=[-0.2, 0.2],
            axis=1,
            vertices=[
                (-0.43, -0.43),
                (0.4, -0.35),
                (0 - 1e-14, 0.05 - 1e-4),
                (0 + 1e-14, 0.05 + 1e-4),
                (-0.35, 0.4),
            ],
        ),
        medium=td.PECMedium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[diff],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_finder=None,
                    # isolate gap meshing: disable the default-on in-plane edge refinement, which
                    # would otherwise refine the slanted PolySlab edges of this structure
                    in_plane_edge_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=7,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()
    assert sim.grid_info["min_grid_size"] > 0.05


def test_gap_meshing_dl_min_warning():
    """Test that warning is displayed when dl_min_from_gaps is very small relative to lateral grid size."""
    from ..utils import AssertLogStr

    wavelength = 1.0
    # Create a very small gap that will trigger the warning
    # For a grid with min_steps_per_wvl=10 and wavelength=1, the grid size will be around 0.1
    # We need gap_width such that DL_MIN_FROM_GAPS_FRACTION * gap_width < GAP_REFINEMENT_WARNING_THRESH * min_lateral_grid_size
    # gap_width < (0.1 * 0.1) / 0.45 ≈ 0.022
    small_gap_width = 0.01  # Very small gap that will trigger warning

    # Create two parallel PEC strips with a small gap between them
    # Make structures smaller to avoid edge warnings (size 0.3 instead of 0.5 in y-direction)
    strip1 = td.Structure(
        geometry=td.Box(center=(-small_gap_width / 2 - 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    strip2 = td.Structure(
        geometry=td.Box(center=(small_gap_width / 2 + 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    # Create layer refinement spec with gap meshing enabled
    layer_spec = td.LayerRefinementSpec(
        axis=2,  # z-axis
        size=(td.inf, td.inf, 0.2),
        center=(0, 0, 0),
        gap_meshing_iters=1,
        dl_min_from_gap_width=True,
        corner_snapping=False,
        corner_refinement=None,
        # isolate the gap-width warning: disable the default-on small-geometry resolution, which
        # would otherwise refine these deliberately small strips and change the lateral grid size
        min_steps_per_geometry=None,
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength,
        min_steps_per_wvl=10,  # This will create grid cells of ~0.1 size
        layer_refinement_specs=[layer_spec],
    )

    # Test that warning IS displayed for very small gap
    # Use AssertLogStr to filter out edge warnings and only check for our specific warning
    with AssertLogStr("WARNING", contains_str="detected a very small gap width"):
        sim = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1, strip2],
            grid_spec=grid_spec,
            run_time=1e-15,
        )

    # Now test with a larger gap that should NOT trigger warning
    # gap_width should be large enough: gap_width > (GAP_REFINEMENT_WARNING_THRESH * min_lateral_grid_size) / DL_MIN_FROM_GAPS_FRACTION
    # For min_lateral_grid_size ~ 0.1: gap_width > (0.1 * 0.1) / 0.45 ≈ 0.022
    large_gap_width = 0.05  # Large enough to not trigger warning

    strip1_large = td.Structure(
        geometry=td.Box(center=(-large_gap_width / 2 - 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    strip2_large = td.Structure(
        geometry=td.Box(center=(large_gap_width / 2 + 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    # Test that warning is NOT displayed for larger gap
    # Use AssertLogStr to exclude edge warnings and check that our specific warning is not present
    with AssertLogStr("WARNING", excludes_str="detected a very small gap width"):
        sim_large = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1_large, strip2_large],
            grid_spec=grid_spec,
            run_time=1e-15,
        )

    # Test with dl_min_from_gap_width=False - should not warn even with small gap
    layer_spec_no_gap = layer_spec.updated_copy(dl_min_from_gap_width=False)
    grid_spec_no_gap = grid_spec.updated_copy(layer_refinement_specs=[layer_spec_no_gap])

    with AssertLogStr("WARNING", excludes_str="detected a very small gap width"):
        sim_no_gap = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1, strip2],
            grid_spec=grid_spec_no_gap,
            run_time=1e-15,
        )
