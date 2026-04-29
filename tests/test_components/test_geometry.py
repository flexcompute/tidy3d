"""Tests Geometry objects."""

from __future__ import annotations

import math
import warnings

import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import pytest
import shapely
import trimesh
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import tidy3d as td
from tidy3d.compat import _package_is_older_than
from tidy3d.components.geometry.base import cleanup_shapely_object
from tidy3d.components.geometry.float_utils import increment_float
from tidy3d.components.geometry.mesh import AREA_SIZE_THRESHOLD
from tidy3d.components.geometry.polyslab import _PolyBulgeUtil
from tidy3d.components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    filter_intersecting_geometries,
    flatten_groups,
    flatten_shapely_geometries,
    merging_geometries_on_plane,
    snap_box_to_grid,
    traverse_geometries,
)
from tidy3d.components.geometry.utils_2d import _is_sliver_polygon, subdivide
from tidy3d.constants import LARGE_NUMBER, fp_eps
from tidy3d.exceptions import SetupError, Tidy3dKeyError, ValidationError

from ..utils import AssertLogLevel

GEO = td.Box(size=(1, 1, 1))
GEO_INF = td.Box(size=(1, 1, td.inf))
BOX = td.Box(size=(1, 1, 1))
BOX_2D = td.Box(size=(1, 0, 1))
POLYSLAB = td.PolySlab(vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(-0.5, 0.5), axis=2)
POLYSLAB_ARC = td.PolySlab(
    vertices=((0, 0), (1, 0), (1, 1), (0, 1)),
    bulges=[0.3, 0, -0.2, 0],
    slab_bounds=(-0.5, 0.5),
    axis=2,
)
SPHERE = td.Sphere(radius=1)
CYLINDER = td.Cylinder(axis=2, length=1, radius=1)

GROUP = td.GeometryGroup(
    geometries=(
        td.Box(center=(-0.25, 0, 0), size=(0.5, 1, 1)),
        td.Box(center=(0.25, 0, 0), size=(0.5, 1, 1)),
    )
)
UNION = td.ClipOperation(
    operation="union",
    geometry_a=td.Box(center=(-0.25, 0, 0), size=(0.5, 1, 1)),
    geometry_b=td.Box(center=(0.25, 0, 0), size=(0.5, 1, 1)),
)
INTERSECTION = td.ClipOperation(operation="intersection", geometry_a=UNION, geometry_b=SPHERE)
DIFFERENCE = td.ClipOperation(operation="difference", geometry_a=CYLINDER, geometry_b=BOX)
SYM_DIFFERENCE = td.ClipOperation(
    operation="symmetric_difference",
    geometry_a=td.ClipOperation(
        operation="difference",
        geometry_a=td.Box(size=(td.inf, td.inf, td.inf)),
        geometry_b=td.Box(center=(-0.25, 0, 0), size=(0.5, 1, 1)),
    ),
    geometry_b=td.ClipOperation(
        operation="difference",
        geometry_a=td.Box(size=(td.inf, td.inf, td.inf)),
        geometry_b=td.Box(center=(0.25, 0, 0), size=(0.5, 1, 1)),
    ),
)
TRANSFORMED = td.Transformed(
    geometry=BOX,
    transform=td.Transformed.rotation(np.pi / 6, 0),
)

GEOMETRY_ARRAY = td.GeometryArray(
    geometry=BOX,
    offsets=[[0, 0, 0], [2, 0, 0], [0, 2, 0]],
)


GEO_TYPES = [
    BOX,
    CYLINDER,
    SPHERE,
    POLYSLAB,
    POLYSLAB_ARC,
    UNION,
    INTERSECTION,
    DIFFERENCE,
    SYM_DIFFERENCE,
    GROUP,
    TRANSFORMED,
    GEOMETRY_ARRAY,
]

_, AX = plt.subplots()


@pytest.mark.parametrize("component", GEO_TYPES)
def test_plot(component):
    _ = component.plot(z=0, ax=AX)
    plt.close()


def test_plot_with_units():
    _ = BOX.plot(z=0, ax=AX, plot_length_units="nm")
    plt.close()
    _ = BOX.plot(z=0, ax=AX, plot_length_units="mil")
    plt.close()


def test_base_inside():
    assert td.Geometry.inside(GEO, x=0, y=0, z=0)
    assert np.all(td.Geometry.inside(GEO, np.array([0, 0]), np.array([0, 0]), np.array([0, 0])))
    assert np.all(
        td.Geometry.inside(GEO, np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]))
    )


def test_base_inside_meshgrid():
    assert np.all(td.Geometry.inside_meshgrid(GEO, x=[0], y=[0], z=[0]))
    assert np.all(td.Geometry.inside_meshgrid(GEO, [0, 0], [0, 0], [0, 0]))
    # Input dimensions different than 1 error for ``inside_meshgrid``.
    with pytest.raises(ValueError):
        _ = td.Geometry.inside_meshgrid(GEO, x=0, y=0, z=0)
    with pytest.raises(ValueError):
        _ = td.Geometry.inside_meshgrid(GEO, [[0, 0]], [[0, 0]], [[0, 0]])


def test_bounding_box():
    assert GEO.bounding_box == GEO
    assert GEO_INF.bounding_box == GEO_INF


@pytest.mark.parametrize("points_shape", [(3,), (3, 10)])
def test_rotate_points(points_shape):
    points = np.random.random(points_shape)
    points_rotated = td.Geometry.rotate_points(points=points, axis=(0, 0, 1), angle=2 * np.pi)
    assert np.allclose(points, points_rotated)
    points_rotated = td.Geometry.rotate_points(points=points, axis=(0, 0, 1), angle=np.pi)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflect_points(axis):
    points = np.random.random((3, 10))
    pr = GEO.reflect_points(points=points, polar_axis=2, angle_theta=2 * np.pi, angle_phi=0)
    assert np.allclose(pr, points)
    pr = GEO.reflect_points(points=points, polar_axis=2, angle_theta=0, angle_phi=2 * np.pi)
    assert np.allclose(pr, points)


@pytest.mark.parametrize("component", GEO_TYPES)
def test_volume(component):
    _ = component.volume()
    _ = component.volume(bounds=GEO.bounds)
    _ = component.volume(bounds=((-100, -100, -100), (100, 100, 100)))
    _ = component.volume(bounds=((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    _ = component.volume(bounds=((-100, -100, -100), (-10, -10, -10)))
    _ = component.volume(bounds=((10, 10, 10), (100, 100, 100)))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_surface_area(component):
    _ = component.surface_area()
    _ = component.surface_area(bounds=GEO.bounds)
    _ = component.surface_area(bounds=((-100, -100, -100), (100, 100, 100)))
    _ = component.surface_area(bounds=((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    _ = component.surface_area(bounds=((-100, -100, -100), (-10, -10, -10)))
    _ = component.surface_area(bounds=((10, 10, 10), (100, 100, 100)))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_bounds(component):
    _ = component.bounds


@pytest.mark.parametrize(
    "component,expected_bounds",
    [
        (CYLINDER, ((-1.0, -1.0, -0.5), (1.0, 1.0, 0.5))),
        (POLYSLAB, ((0.0, 0.0, -0.5), (1.0, 1.0, 0.5))),
    ],
)
def test_planar_bounds(component, expected_bounds):
    assert all(a == b for a, b in zip(component.bounds, expected_bounds))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_inside(component):
    _ = component.inside(0, 0, 0)
    _ = component.inside(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
    _ = component.inside(np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]))


def test_zero_dims():
    assert BOX.zero_dims == []
    assert BOX_2D.zero_dims == [1]


def test_inside_polyslab_sidewall():
    ps = POLYSLAB.copy(update={"sidewall_angle": 0.1})
    ps.inside(x=0, y=0, z=0)


# TODO: Weiliang fix this test? does not work when sidewall non-zero
def test_inside_polyslab_sidewall_arrays():
    inside_kwargs = {coord: np.array([-1, 0, 1]) for coord in "xyz"}
    POLYSLAB.inside(**inside_kwargs)
    # ps = POLYSLAB.copy(update=dict(sidewall_angle=0.1))
    # ps.inside(**inside_kwargs)


def test_array_to_vertices():
    vertices = ((0, 0), (1, 0), (1, 1))
    array = POLYSLAB.vertices_to_array(vertices)
    vertices2 = POLYSLAB.array_to_vertices(array)
    assert np.all(np.array(vertices) == np.array(vertices2))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_intersections_plane(component):
    assert len(component.intersections_plane(z=0.2)) > 0
    assert len(component.intersections_plane(x=0.2)) > 0
    assert len(component.intersections_plane(x=10000)) == 0


def test_intersections_plane_inf():
    a = (
        td.Cylinder(radius=3.2, center=(0.45, 9, 0), length=td.inf)
        + td.Box(center=(0, 0, 0), size=(0.9, 24, td.inf))
        + td.Box(center=(0, 0, 0), size=(7.3, 18, td.inf))
    )
    b = td.Cylinder(radius=2.9, center=(-0.45, 9, 0), length=td.inf)
    c = a - b
    assert len(c.intersections_plane(y=0)) == 1


@pytest.mark.parametrize("component", [BOX, CYLINDER, SPHERE, POLYSLAB, UNION, GROUP])
@pytest.mark.parametrize("cleanup", [True, False])
def test_intersections_plane_cleanup_param(component, cleanup):
    """Test that cleanup parameter is accepted by all geometry types."""
    shapes = component.intersections_plane(z=0, cleanup=cleanup)
    assert isinstance(shapes, list)
    shapes_default = component.intersections_plane(z=0)
    # Both should return valid shapely objects
    for shape in shapes:
        assert hasattr(shape, "is_valid")
    for shape in shapes_default:
        assert hasattr(shape, "is_valid")


@pytest.mark.parametrize("component", [SPHERE, CYLINDER])
@pytest.mark.parametrize("quad_segs", [8, 50, 200])
def test_intersections_plane_quad_segs(component, quad_segs):
    """Test that quad_segs parameter controls discretization of circular shapes."""
    shapes = component.intersections_plane(z=0, quad_segs=quad_segs)
    assert len(shapes) > 0
    # For circular shapes, quad_segs should affect the number of vertices
    # Higher quad_segs should generally give more vertices
    shape = shapes[0]
    num_coords = len(shape.exterior.coords)
    assert num_coords > 4 * quad_segs, (
        f"Expected more than {4 * quad_segs} coords, got {num_coords}"
    )


def test_center_not_inf_validate():
    with pytest.raises(pd.ValidationError):
        _ = td.Box(center=(td.inf, 0, 0))
    with pytest.raises(pd.ValidationError):
        _ = td.Box(center=(-td.inf, 0, 0))


def test_center_default_and_schema_regression():
    box = td.Box(size=(1, 1, 1))
    assert box.center == (0.0, 0.0, 0.0)

    with pytest.raises(pd.ValidationError):
        td.Box(size=(1, 1, 1), center=None)

    box_center_schema = td.Box.model_json_schema()["properties"]["center"]
    assert box_center_schema["default"] == [0.0, 0.0, 0.0]
    assert "anyOf" not in box_center_schema

    monitor_center_schema = td.FieldMonitor.model_json_schema()["properties"]["center"]
    assert monitor_center_schema["default"] == [0.0, 0.0, 0.0]
    assert "anyOf" not in monitor_center_schema

    simulation_center_schema = td.Simulation.model_json_schema()["properties"]["center"]
    assert simulation_center_schema["default"] == [0.0, 0.0, 0.0]
    assert "anyOf" not in simulation_center_schema


def test_radius_not_inf_validate():
    with pytest.raises(pd.ValidationError):
        _ = td.Sphere(radius=td.inf)
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(radius=td.inf, center=(0, 0, 0), axis=1, length=1)


def test_slanted_cylinder_infinite_length_validate():
    _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf)
    _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="top")
    _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="bottom")
    _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="middle")
    _ = td.Cylinder(
        radius=1,
        center=(0, 0, 0),
        axis=1,
        length=td.inf,
        sidewall_angle=0.1,
        reference_plane="middle",
    )
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(
            radius=1,
            center=(0, 0, 0),
            axis=1,
            length=td.inf,
            sidewall_angle=0.1,
            reference_plane="top",
        )
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(
            radius=1,
            center=(0, 0, 0),
            axis=1,
            length=td.inf,
            sidewall_angle=0.1,
            reference_plane="bottom",
        )


def test_cylinder_to_polyslab():
    ps = CYLINDER.to_polyslab(num_pts_circumference=10, dilation=0.02)


def test_box_from_bounds():
    b = td.Box.from_bounds(rmin=(-td.inf, 0, 0), rmax=(td.inf, 0, 0))
    assert b.center[0] == 0.0

    with pytest.raises(SetupError):
        _ = td.Box.from_bounds(rmin=(0, 0, 0), rmax=(td.inf, 0, 0))

    b = td.Box.from_bounds(rmin=(-1, -1, -1), rmax=(1, 1, 1))
    assert b.center == (0, 0, 0)


def test_box_padded_copy():
    """Test that padding layers are added along box boundaries."""
    box = td.Box(size=(3, 2, 4))
    padded_box = box.padded_copy(x=(4, 10), y=(1, 2))
    assert np.allclose(np.array(padded_box.size), np.array([17, 5, 4]))
    assert np.allclose(np.array(padded_box.center), np.array([3, 0.5, 0]))

    # ensure errors are raised if padding  format is invalid.
    with pytest.raises(ValueError):
        padded_box = box.padded_copy(x=(1, -2), z=(-2, 0))
    with pytest.raises(ValueError):
        padded_box = box.padded_copy(x=(1))


def test_polyslab_center_axis():
    """Test the handling of center_axis in a polyslab having (-td.inf, td.inf) bounds."""
    ps = POLYSLAB.copy(update={"slab_bounds": (-td.inf, td.inf)})
    assert ps.center_axis == 0


@pytest.mark.parametrize(
    "lower_bound, upper_bound", ((-td.inf, td.inf), (-1, td.inf), (-td.inf, 1))
)
def test_polyslab_inf_bounds(lower_bound, upper_bound):
    """Test the handling of various operations in a polyslab having inf bounds."""
    ps = POLYSLAB.copy(update={"slab_bounds": (lower_bound, upper_bound)})
    # catch any runtime warning related to inf operations
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = ps.bounds
        ps.intersections_plane(x=0.5)
        ps.intersections_plane(z=0)


def test_polyslab_bounds():
    with pytest.raises(pd.ValidationError):
        td.PolySlab(vertices=((0, 0), (1, 0), (1, 1)), slab_bounds=(0.5, -0.5), axis=2)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_polyslab_inf_to_finite_bounds(axis):
    """Test that finite_length_axis for PolySlab first clips at LARGE_NUMBER and then computes the length."""
    axis_bound = 20
    ps_low_inf = td.PolySlab(
        axis=axis,
        slab_bounds=[-td.inf, axis_bound],
        vertices=[[0, 0], [2.5, 1], [2, 3], [0.5, 4], [-1.5, 2.5]],
    )
    ps_high_inf = td.PolySlab(
        axis=axis,
        slab_bounds=[-axis_bound, td.inf],
        vertices=[[0, 0], [2.5, 1], [2, 3], [0.5, 4], [-1.5, 2.5]],
    )
    ps_inf = td.PolySlab(
        axis=axis,
        slab_bounds=[-td.inf, td.inf],
        vertices=[[0, 0], [2.5, 1], [2, 3], [0.5, 4], [-1.5, 2.5]],
    )

    assert ps_low_inf.finite_length_axis == (LARGE_NUMBER + axis_bound), (
        "Unexpected finite length for polyslab axis with -inf bound"
    )
    assert ps_high_inf.finite_length_axis == (LARGE_NUMBER + axis_bound), (
        "Unexpected finite length for polyslab axis with inf bound"
    )
    assert ps_inf.finite_length_axis == 2 * LARGE_NUMBER, (
        "Unexpected finite length for polyslab axis with two inf bounds"
    )


def test_validate_polyslab_vertices_valid():
    with pytest.raises(pd.ValidationError):
        POLYSLAB.copy(update={"vertices": (1, 2, 3)})
    with pytest.raises(pd.ValidationError):
        crossing_verts = ((0, 0), (1, 1), (0, 1), (1, 0))
        POLYSLAB.copy(update={"vertices": crossing_verts})


def test_sidewall_failed_validation():
    with pytest.raises(pd.ValidationError):
        POLYSLAB.copy(update={"sidewall_angle": 1000})


def test_bulge_size_validation():
    size = POLYSLAB.vertices.shape[0]
    with pytest.raises(pd.ValidationError):
        POLYSLAB.updated_copy(bulges=[1, 2])
    POLYSLAB.updated_copy(bulges=np.random.random(size))

    assert len(POLYSLAB._bulges) == POLYSLAB.vertices.shape[0]
    assert np.allclose(POLYSLAB._bulges, 0)


def test_bulge_compatibility_validation():
    polyslab = POLYSLAB.updated_copy(bulges=np.random.random(POLYSLAB.vertices.shape[0]))
    POLYSLAB.updated_copy(dilation=0.01)
    with pytest.raises(pd.ValidationError):
        polyslab.updated_copy(dilation=0.01)
    POLYSLAB.updated_copy(sidewall_angle=0.01)
    with pytest.raises(pd.ValidationError):
        polyslab.updated_copy(sidewall_angle=0.01)


def test_arc_geometry_helpers():
    """Test arc geometry helper functions."""
    # All tests use chord from (0,0) to (2,0), chord_length = 2.
    edge_start = np.array([[0.0, 0.0]])
    edge_end = np.array([[2.0, 0.0]])

    # --- Semicircle: bulge = tan(45°) = 1.0, included angle = 180° ---
    # Positive bulge → CCW arc bulging downward (perpendicular to chord).
    # radius = chord / (2 sin(90°)) = 1
    # center = midpoint + 0 * perp = (1, 0)  (midpoint_to_center_dist = 0)
    arc = _PolyBulgeUtil._arcs_from_bulges(edge_start, edge_end, np.array([1.0]))
    assert np.isclose(arc["included_angles"][0], np.pi)
    assert np.isclose(arc["radii"][0], 1.0)
    assert np.allclose(arc["centers"][0], [1.0, 0.0])
    assert np.isclose(arc["start_angles"][0], np.pi)  # atan2(0, -1)
    assert np.isclose(arc["end_angles"][0], 0.0)  # atan2(0, 1)

    # --- Quarter-circle: bulge = tan(22.5°), included angle = 90° ---
    # radius = chord / (2 sin(45°)) = 2 / √2 = √2
    # midpoint_to_center_dist = √2 - tan(22.5°) = 1.0
    # center = (1, 0) + 1 * (0, 1) = (1, 1)
    quarter_bulge = np.tan(np.pi / 8)  # tan(22.5°)
    arc = _PolyBulgeUtil._arcs_from_bulges(edge_start, edge_end, np.array([quarter_bulge]))
    assert np.isclose(arc["included_angles"][0], np.pi / 2)
    assert np.isclose(arc["radii"][0], np.sqrt(2))
    assert np.allclose(arc["centers"][0], [1.0, 1.0])
    assert np.isclose(arc["start_angles"][0], -3 * np.pi / 4)  # atan2(-1, -1)
    assert np.isclose(arc["end_angles"][0], -np.pi / 4)  # atan2(-1, 1)

    # --- Negative quarter-circle: bulge = -tan(22.5°), included angle = -90° ---
    # radius = √2 (same magnitude)
    # center = (1, 0) + (-1) * 1 * (0, 1) = (1, -1)
    arc = _PolyBulgeUtil._arcs_from_bulges(edge_start, edge_end, np.array([-quarter_bulge]))
    assert np.isclose(arc["included_angles"][0], -np.pi / 2)
    assert np.isclose(arc["radii"][0], np.sqrt(2))
    assert np.allclose(arc["centers"][0], [1.0, -1.0])
    assert np.isclose(arc["start_angles"][0], 3 * np.pi / 4)  # atan2(1, -1)
    assert np.isclose(arc["end_angles"][0], np.pi / 4)  # atan2(1, 1)

    # --- Batch: semicircle + negative quarter-circle in one call ---
    edge_starts = np.array([[0.0, 0.0], [0.0, 0.0]])
    edge_ends = np.array([[2.0, 0.0], [2.0, 0.0]])
    bulges = np.array([1.0, -quarter_bulge])
    arcs = _PolyBulgeUtil._arcs_from_bulges(edge_starts, edge_ends, bulges)
    assert np.isclose(arcs["radii"][0], 1.0)
    assert np.isclose(arcs["radii"][1], np.sqrt(2))
    assert np.allclose(arcs["centers"][0], [1.0, 0.0])
    assert np.allclose(arcs["centers"][1], [1.0, -1.0])
    assert arcs["included_angles"][0] > 0  # positive bulge → CCW sweep
    assert arcs["included_angles"][1] < 0  # negative bulge → CW sweep

    # --- Verify endpoints lie on the arc (distance to center == radius) ---
    for bulge_val in [1.0, quarter_bulge, -quarter_bulge]:
        arc = _PolyBulgeUtil._arcs_from_bulges(edge_start, edge_end, np.array([bulge_val]))
        center = arc["centers"][0]
        radius = arc["radii"][0]
        dist_start = np.linalg.norm(edge_start[0] - center)
        dist_end = np.linalg.norm(edge_end[0] - center)
        assert np.isclose(dist_start, radius)
        assert np.isclose(dist_end, radius)

    # --- _polygon_arcs_bounds ---
    # Arc that stays below the chord (positive bulge)
    bounds_verts = np.array([[0.0, 0.0], [2.0, 0.0]])
    bulge_data = _PolyBulgeUtil._compute_bulge_data(bounds_verts, np.array([0.5, 0.0]))
    min_c, max_c = _PolyBulgeUtil._polygon_arcs_bounds(bounds_verts, bulge_data)
    assert min_c[0, 0] <= 0 and max_c[0, 0] >= 2.0
    assert min_c[0, 1] <= 0  # arc bulges downward for positive bulge

    # Arc that stays above the chord (negative bulge)
    bulge_data = _PolyBulgeUtil._compute_bulge_data(bounds_verts, np.array([-0.5, 0.0]))
    min_c, max_c = _PolyBulgeUtil._polygon_arcs_bounds(bounds_verts, bulge_data)
    assert max_c[0, 1] >= 0  # arc bulges upward for negative bulge

    # --- _arcs_from_bulges: zero bulge must raise ValueError ---
    with pytest.raises(ValueError, match=r"non-zero"):
        _PolyBulgeUtil._arcs_from_bulges(edge_start, edge_end, np.array([0.0]))

    # --- _polygon_discretize: no-arc early return ---
    tri_verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    bd_zero = _PolyBulgeUtil._compute_bulge_data(tri_verts, np.array([0.0, 0.0, 0.0]))
    disc_zero = _PolyBulgeUtil._polygon_discretize(tri_verts, bd_zero)
    np.testing.assert_array_equal(disc_zero, tri_verts)

    # --- _polygon_discretize: explicit num_points_per_arc ---
    sq_verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    bd_one_arc = _PolyBulgeUtil._compute_bulge_data(sq_verts, np.array([0.3, 0.0, 0.0, 0.0]))
    disc_explicit = _PolyBulgeUtil._polygon_discretize(sq_verts, bd_one_arc, num_points_per_arc=10)
    assert len(disc_explicit) == 4 + 10  # 4 original vertices + 10 arc points


def test_polyslab_with_arcs_basic():
    """Test basic PolySlab creation with arc segments."""
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # Create with small bulge on first edge
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0.2, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )

    assert polyslab._has_arc_segments is True
    assert len(polyslab._bulges) == 4
    assert np.isclose(polyslab._bulges[0], 0.2)

    # Discretized polygon should have more points
    disc_verts = polyslab._discretized_reference_polygon
    assert len(disc_verts) > len(vertices)

    # JSON serialization should preserve geometry
    json_str = polyslab.json()
    polyslab_loaded = td.PolySlab.parse_raw(json_str)
    assert np.allclose(polyslab_loaded.bulges, polyslab.bulges)
    assert polyslab_loaded._has_arc_segments
    np.testing.assert_allclose(
        polyslab_loaded._discretized_reference_polygon,
        polyslab._discretized_reference_polygon,
    )
    assert np.allclose(polyslab_loaded.bounds, polyslab.bounds)

    # Trimesh via tilted-plane intersection should produce valid geometry
    # (exercises _do_intersections_tilted_plane with arc-discretized polygon)
    tilted_shapes = polyslab.intersections_plane(x=0.5)
    assert len(tilted_shapes) >= 1
    for shape in tilted_shapes:
        assert shape.is_valid

    # Non-axis-aligned (diagonal) edges with arcs
    tri_verts = [(0, 0), (2, 0), (1, 2)]
    tri_ps = td.PolySlab(vertices=tri_verts, bulges=[0.3, 0.2, 0], axis=2, slab_bounds=(-0.5, 0.5))
    assert tri_ps._has_arc_segments
    assert len(tri_ps._discretized_reference_polygon) > 3


def test_polyslab_arc_self_intersection_validation():
    """Test that self-intersecting arc polygons raise error."""
    # Narrow rectangle where inward bulges cross in the middle
    vertices = [(0, 0), (2, 0), (2, 0.3), (0, 0.3)]

    # Large inward bulges that cross each other
    with pytest.raises(pd.ValidationError):
        td.PolySlab(
            vertices=vertices,
            bulges=[-1.5, 0, 1.5, 0],  # both bulge toward center, causing intersection
            axis=2,
            slab_bounds=(-0.5, 0.5),
        )


def test_polyslab_arc_area_perimeter():
    """Test area and perimeter calculations with arc segments."""
    # Unit square
    vertices = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Without arcs: area = 1, perimeter = 4
    area_no_arc = td.PolySlab._area(vertices)
    perim_no_arc = td.PolySlab._perimeter(vertices)
    assert np.isclose(abs(area_no_arc), 1.0)
    assert np.isclose(perim_no_arc, 4.0)

    # With semicircular bulge on one edge (bulge=1 on bottom edge)
    # The semicircle adds area = pi*r^2/2 = pi*0.5^2/2 = pi/8
    bulges = np.array([1.0, 0, 0, 0])
    bulge_data = _PolyBulgeUtil._compute_bulge_data(vertices, bulges)
    area_with_arc = td.PolySlab._area(vertices) + _PolyBulgeUtil._arc_segment_area(bulge_data)
    # For a unit chord with bulge=1: radius=0.5, segment area = r^2*(pi - sin(pi))/2 = 0.5^2*pi/2
    expected_segment_area = 0.25 * np.pi / 2
    assert np.isclose(abs(area_with_arc), 1.0 + expected_segment_area, rtol=0.01)

    # Perimeter with arc: replaces chord=1 with arc=pi*0.5=pi/2
    perim_with_arc = _PolyBulgeUtil._polygon_perimeter(vertices, bulge_data)
    assert np.isclose(perim_with_arc, 3 + np.pi / 2, rtol=0.01)


def test_polyslab_arc_bounds():
    """Test that bounds properly account for arc extents."""
    # Square from (0,0) to (1,1)
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # No arcs: bounds should be exactly the vertices
    polyslab_no_arc = td.PolySlab(
        vertices=vertices,
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    bounds = polyslab_no_arc.bounds
    assert np.isclose(bounds[0][0], 0)  # xmin
    assert np.isclose(bounds[1][0], 1)  # xmax
    assert np.isclose(bounds[0][1], 0)  # ymin
    assert np.isclose(bounds[1][1], 1)  # ymax

    # With outward bulge on bottom edge: y should extend below 0
    polyslab_arc = td.PolySlab(
        vertices=vertices,
        bulges=[0.5, 0, 0, 0],  # bulge outward on bottom edge
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    bounds_arc = polyslab_arc.bounds
    assert bounds_arc[0][1] < 0  # ymin should be negative (arc extends below)

    # Multiple arcs (outward bottom + inward right + outward top)
    polyslab_multi = td.PolySlab(
        vertices=vertices,
        bulges=[0.5, -0.3, 0.5, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    bounds_multi = polyslab_multi.bounds
    assert bounds_multi[0][1] < 0  # bottom outward arc extends below
    assert bounds_multi[1][1] > 1  # top outward arc extends above
    # z bounds unchanged by arcs
    assert np.isclose(bounds_multi[0][2], -0.5)
    assert np.isclose(bounds_multi[1][2], 0.5)


def test_polyslab_arc_inside():
    """Test point containment with arc segments."""
    # Square with outward bulge on bottom
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0.5, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )

    # Point inside the square part
    x = np.array([0.5])
    y = np.array([0.5])
    z = np.array([0.0])
    assert polyslab.inside(x, y, z)[0]

    # Point in the arc bulge region (below y=0 but within arc)
    x_arc = np.array([0.5])
    y_arc = np.array([-0.05])  # slightly below y=0
    z_arc = np.array([0.0])
    assert polyslab.inside(x_arc, y_arc, z_arc)[0]

    # Point outside the arc bulge region
    x_out = np.array([0.5])
    y_out = np.array([-0.5])  # far below y=0
    z_out = np.array([0.0])
    assert not polyslab.inside(x_out, y_out, z_out)[0]

    # Negative bulge (inward arc) on right edge
    polyslab_neg = td.PolySlab(
        vertices=vertices,
        bulges=[0, -0.5, 0, 0],  # inward arc on right edge (1,0)->(1,1)
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    # Point at the edge midpoint should be outside (arc bows inward)
    assert not polyslab_neg.inside(np.array([1.0]), np.array([0.5]), np.array([0.0]))[0]
    # Point well inside should still be inside
    assert polyslab_neg.inside(np.array([0.3]), np.array([0.5]), np.array([0.0]))[0]


def test_polyslab_arc_intersections():
    """Test cross-section intersections with arc segments."""
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0.3, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )

    # Get intersection at z=0 (normal to axis)
    shapes = polyslab._intersections_normal(z=0.0)
    assert len(shapes) == 1

    # The intersection polygon should have more than 4 vertices (due to arc discretization)
    poly = shapes[0]
    assert len(poly.exterior.coords) > 5

    # Side intersection (plane orthogonal to slab, cutting through arc region).
    # The arc on edge (0,0)→(1,0) with bulge=0.3 extends below y=0,
    # so y=-0.01 only intersects the polyslab if arc discretization is used.
    shapes_x = polyslab.intersections_plane(x=0.5)
    assert len(shapes_x) >= 1
    shapes_y = polyslab.intersections_plane(y=-0.01)
    assert len(shapes_y) >= 1

    # interior_angle is not implemented for arc segments
    with pytest.raises(NotImplementedError):
        polyslab.interior_angle


def test_polyslab_arc_inside_scalar():
    """Scalar and ndarray inside() must agree for a point in the arc bulge region."""
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0.5, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )

    # Point in the arc bulge region (below y=0 but within arc)
    x_val, y_val, z_val = 0.5, -0.05, 0.0

    # ndarray path
    inside_array = polyslab.inside(np.array([x_val]), np.array([y_val]), np.array([z_val]))[0]

    # scalar path
    inside_scalar = polyslab.inside(x_val, y_val, z_val)

    assert inside_array == inside_scalar, (
        f"Scalar ({inside_scalar}) and ndarray ({inside_array}) inside() disagree "
        f"for point in arc bulge region"
    )
    # Both should be True — the point is inside the arc
    assert inside_scalar


def test_polyslab_arc_tilted_plane_intersection():
    """Tilted-plane intersection trimesh must include arc bulge geometry."""
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0.5, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )

    # Intersect with the y=-0.01 plane (cuts through the arc bulge region only)
    shapes = polyslab.intersections_plane(y=-0.01)
    assert len(shapes) >= 1, "Tilted-plane intersection missed the arc bulge region"


def test_bulge_finite_validation():
    """Inf and NaN bulge values must raise ValidationError."""
    size = POLYSLAB.vertices.shape[0]
    with pytest.raises(pd.ValidationError):
        POLYSLAB.updated_copy(bulges=[np.inf, 0, 0, 0])
    with pytest.raises(pd.ValidationError):
        POLYSLAB.updated_copy(bulges=[0, -np.inf, 0, 0])
    with pytest.raises(pd.ValidationError):
        POLYSLAB.updated_copy(bulges=[0, 0, np.nan, 0])


def test_zero_length_edge_nonzero_bulge():
    """Duplicate vertex with non-zero bulge must raise an error."""
    # Two identical vertices at (0,0) with a non-zero bulge on that zero-length edge
    vertices = [(0, 0), (0, 0), (1, 0), (1, 1)]
    with pytest.raises(pd.ValidationError):
        td.PolySlab(
            vertices=vertices,
            bulges=[0.5, 0, 0, 0],
            axis=2,
            slab_bounds=(-0.5, 0.5),
        )


def test_zero_length_edge_zero_bulge_dropped():
    """Duplicate vertex with zero bulge should be silently dropped."""
    # (0,0) appears twice; the zero-length edge has bulge=0, should be dropped
    vertices = [(0, 0), (0, 0), (1, 0), (1, 1)]
    polyslab = td.PolySlab(
        vertices=vertices,
        bulges=[0, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    canon_verts, canon_bulges = polyslab._canonical_vertices_and_bulges
    # After dropping the duplicate, should have 3 vertices
    assert len(canon_verts) == 3
    assert len(canon_bulges) == 3


def test_winding_reversal_adjusts_bulges():
    """CW input should get reversed to CCW with bulges permuted and sign-flipped."""
    # CW-ordered square: (0,0) -> (0,1) -> (1,1) -> (1,0)
    vertices_cw = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=float)
    bulges_cw = np.array([0.2, 0.0, -0.3, 0.0])

    canon_verts, canon_bulges = td.PolySlab._canonicalize_vertices_and_bulges(
        vertices_cw, bulges_cw
    )
    # After canonicalization, should be CCW
    assert td.PolySlab._area(canon_verts) > 0
    # Bulge signs should be flipped and permuted
    # For reversal: bulges_new = -np.roll(bulges[::-1], -1)
    expected_bulges = -np.roll(bulges_cw[::-1], -1)
    np.testing.assert_allclose(canon_bulges, expected_bulges)


def test_adjoint_error_with_bulges():
    """_compute_derivatives must raise NotImplementedError for non-zero bulges."""
    polyslab = td.PolySlab(
        vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
        bulges=[0.2, 0, 0, 0],
        axis=2,
        slab_bounds=(-0.5, 0.5),
    )
    with pytest.raises(NotImplementedError, match=r"Adjoint derivatives are not supported"):
        polyslab._compute_derivatives(derivative_info=None)


def test_surfaces():
    with pytest.raises(SetupError):
        td.Box.surfaces(size=(1, 0, 1), center=(0, 0, 0))

    td.FluxMonitor.surfaces(
        size=(1, 1, 1), center=(0, 0, 0), normal_dir="+", name="test", freqs=[1e12]
    )
    td.Box.surfaces(size=(1, 1, 1), center=(0, 0, 0), normal_dir="+")


def test_surfaces_with_exclusion():
    surfaces = td.Box.surfaces_with_exclusion(
        size=(1, 2, 3), center=(0, 0, 0), exclude_surfaces=["x-", "z+"]
    )

    assert [surface.center for surface in surfaces] == [
        (0.5, 0, 0),
        (0, -1.0, 0),
        (0, 1.0, 0),
        (0, 0, -1.5),
    ]
    assert [surface.size for surface in surfaces] == [
        (0.0, 2, 3),
        (1, 0.0, 3),
        (1, 0.0, 3),
        (1, 2, 0.0),
    ]


def test_arrow_both_dirs():
    _, ax = plt.subplots()
    GEO._plot_arrow(direction=(1, 2, 3), x=0, both_dirs=True, ax=ax)


def test_gdstk_cell():
    gds_cell = gdstk.Cell("name")
    gds_cell.add(gdstk.rectangle((0, 0), (1, 1)))
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0)
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0, gds_dtype=0)
    with pytest.raises(Tidy3dKeyError):
        td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1)
    with pytest.raises(Tidy3dKeyError):
        td.PolySlab.from_gds(
            gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1, gds_dtype=0
        )


def make_geo_group():
    """Make a generic Geometry Group."""
    boxes = tuple(td.Box(size=(1, 1, 1), center=(i, 0, 0)) for i in range(-5, 5))
    return td.GeometryGroup(geometries=boxes)


def test_geo_group_initialize():
    """make sure you can construct one."""
    _ = make_geo_group()


def test_geo_group_structure():
    """make sure you can construct a structure using GeometryGroup."""
    geo_group = make_geo_group()
    _ = td.Structure(geometry=geo_group, medium=td.Medium())


def test_geo_group_methods():
    """Tests the geometry methods of geo group."""
    geo_group = make_geo_group()
    geo_group.inside(0, 1, 2)
    geo_group.inside(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    geo_group.inside_meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    geo_group.intersections_plane(y=0)
    geo_group.intersects(td.Box(size=(1, 1, 1)))
    _ = geo_group.bounds


def test_geo_group_empty():
    """dont allow empty geometry list."""
    with pytest.raises(pd.ValidationError):
        _ = td.GeometryGroup(geometries=())


def test_geo_group_volume():
    geo_group = make_geo_group()
    geo_group.volume(bounds=GEO.bounds)


def test_geo_group_surface_area():
    geo_group = make_geo_group()
    geo_group.surface_area(bounds=GEO.bounds)


def test_geometryoperations():
    assert BOX + CYLINDER == td.GeometryGroup(geometries=(BOX, CYLINDER))
    assert BOX + UNION == td.GeometryGroup(geometries=(BOX, UNION.geometry_a, UNION.geometry_b))
    assert UNION + CYLINDER == td.GeometryGroup(
        geometries=(UNION.geometry_a, UNION.geometry_b, CYLINDER)
    )
    assert BOX + GROUP == td.GeometryGroup(geometries=(BOX, *GROUP.geometries))
    assert GROUP + CYLINDER == td.GeometryGroup(geometries=(*GROUP.geometries, CYLINDER))

    assert BOX | CYLINDER == td.GeometryGroup(geometries=(BOX, CYLINDER))
    assert BOX | UNION == td.GeometryGroup(geometries=(BOX, UNION.geometry_a, UNION.geometry_b))
    assert UNION | CYLINDER == td.GeometryGroup(
        geometries=(UNION.geometry_a, UNION.geometry_b, CYLINDER)
    )
    assert BOX | GROUP == td.GeometryGroup(geometries=(BOX, *GROUP.geometries))
    assert GROUP | CYLINDER == td.GeometryGroup(geometries=(*GROUP.geometries, CYLINDER))

    assert BOX * SPHERE == td.ClipOperation(
        operation="intersection", geometry_a=BOX, geometry_b=SPHERE
    )

    assert BOX & SPHERE == td.ClipOperation(
        operation="intersection", geometry_a=BOX, geometry_b=SPHERE
    )

    assert BOX - SPHERE == td.ClipOperation(
        operation="difference", geometry_a=BOX, geometry_b=SPHERE
    )

    assert BOX ^ SPHERE == td.ClipOperation(
        operation="symmetric_difference", geometry_a=BOX, geometry_b=SPHERE
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_planar_transform(axis):
    geo = (
        td.Box(size=(3 * axis, 2 * abs(axis - 1), 4 * (2 - axis)))
        .rotated(2.0, axis)
        .reflected((axis, 2 * (axis - 1), 3 * (axis - 2)))
        .translated(-1, 2, 3)
        .scaled(1.4, -1.2, 1.3)
    )
    assert geo.bounds[0][axis] == geo.bounds[1][axis]


def test_transforms():
    xyz = (np.array([1.4, 0]), np.array([0, 0.5]), np.array([0, 1.4]))
    geo = td.Box(size=(2, 2, 2))
    assert not geo.inside(*xyz).any()
    geo = geo.rotated(np.pi / 4, 2).rotated(np.pi / 5, 0)
    geo.plot(x=0)
    assert geo.inside(*xyz).all()

    xyz = (np.array([0, 0, -1.5 + 1e-6]), np.array([0, 0, 0]), np.array([-1e-6, 4 - 1e-6, 2]))
    geo = td.Sphere(radius=1)
    assert (geo.inside(*xyz) == (True, False, False)).all()
    geo = geo.translated(0, 0, 1).scaled(1.5, 1, 2)
    geo.plot(y=0)
    assert (geo.inside(*xyz) == (False, True, True)).all()

    xyz = (np.array([0.8, -0.8, -0.7]), np.array([0, 0, 0]), np.array([1.2, -1.2, 0]))
    geo = td.Cylinder(length=2, radius=1)
    assert (geo.inside(*xyz) == (False, False, True)).all()
    geo = geo.scaled(0.5, 2, 1).rotated(-np.pi / 6, 2).rotated(np.pi / 2, 0)
    assert (geo.inside(*xyz) == (True, True, False)).all()

    xyz = (np.array([0, 2, 1, 3, -0.5]), np.array([0, 0, 0, 0, 0.5]), np.array([0, 0, 1.5, 0, 0]))
    geo = geo = td.PolySlab(
        vertices=[(2, -1), (-1, 1), (4, 1), (-1, 2), (4, 2), (1, 3), (5, 3), (5, -1)],
        slab_bounds=(-1, 1),
    )
    assert (geo.inside(*xyz) == (False, True, False, True, False)).all()
    assert len(geo.intersections_plane(x=0)) == 2
    assert len(geo.intersections_plane(z=0)) == 1
    geo = geo.translated(-2, 0, 0).rotated(-np.pi * 0.4, 1).scaled(0.99)
    assert (geo.inside(*xyz) == (True, False, True, False, True)).all()
    assert len(geo.intersections_plane(x=0)) == 1
    assert len(geo.intersections_plane(z=0)) == 3

    # Test reflection of a Box across the XY plane and verify point inclusion.
    xyz = (np.array([1, 1, 1, 3]), np.array([1, 1, 1, 3]), np.array([1, -1, -1.5, 3]))
    geo = td.Box(center=(1, 1, 1), size=(2, 2, 2))
    assert (geo.inside(*xyz) == (True, False, False, False)).all()
    geo = geo.reflected((0, 0, 1))
    assert (geo.inside(*xyz) == (False, True, True, False)).all()

    # Test Sphere multiple reflections not influencing point inclusion.
    xyz = (np.array([1, 2, 2, 1]), np.array([2, 1, 2, 3]), np.array([2, -2, -0.5, -2]))
    geo = td.Sphere(radius=3.5)
    assert (geo.inside(*xyz) == (True, True, True, False)).all()
    geo = geo.reflected((2, 3, 1)).reflected((1, 2, 3))
    assert (geo.inside(*xyz) == (True, True, True, False)).all()

    # Test PolySlab reflection across non-axis plane and verify point inclusion.
    xyz = (np.array([0, 1.5, -1.5, -1.5]), np.array([0, 1.5, -1.5, -2.5]), np.array([0, 0, 0, 0]))
    geo = td.PolySlab(
        vertices=[(1, 0), (3, 2), (2, 2), (0, 0), (2, -2), (3, -2)],
        slab_bounds=(-1, 1),
    )
    assert (geo.inside(*xyz) == (True, True, False, False)).all()
    geo = geo.reflected((1, 1, 0))
    assert (geo.inside(*xyz) == (True, False, True, True)).all()


def test_polyslab_transforms():
    # More tests on PolySlab tranforms matching direct Transformed
    xyz = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    xyz = [c.flatten() for c in xyz]
    geo = geo = td.PolySlab(
        vertices=[(2, -1), (-1, 1), (4, 1), (-1, 2), (4, 2), (1, 3), (5, 3), (5, -1)],
        slab_bounds=(-1, 1),
        axis=1,
    )
    geo_trans = td.Transformed(geometry=geo, transform=td.Transformed.translation(-0.4, 0.5, 0.1))
    geo = geo.translated(-0.4, 0.5, 0.1)
    assert geo.type != geo_trans.type
    assert np.allclose(geo.inside(*xyz), geo_trans.inside(*xyz))
    geo_trans = td.Transformed(geometry=geo, transform=td.Transformed.scaling(0.7, 0.6, 1.5))
    geo = geo.scaled(0.7, 0.6, 1.5)
    assert geo.type != geo_trans.type
    assert np.allclose(geo.inside(*xyz), geo_trans.inside(*xyz))
    geo_trans = td.Transformed(geometry=geo, transform=td.Transformed.rotation(0.3, (0, -0.2, 0)))
    geo = geo.rotated(0.3, (0, -0.2, 0))
    assert geo.type != geo_trans.type
    assert np.allclose(geo.inside(*xyz), geo_trans.inside(*xyz))
    geo_trans = td.Transformed(geometry=geo, transform=td.Transformed.reflection((1, 0, 2)))
    geo = geo.reflected((1, 0, 2))
    assert geo.type != geo_trans.type
    assert np.allclose(geo.inside(*xyz), geo_trans.inside(*xyz))


def test_general_rotation():
    # Magnitude of axis direction does not matter
    assert np.allclose(td.Transformed.rotation(0.1, 0), td.Transformed.rotation(0.1, [2, 0, 0]))
    assert np.allclose(td.Transformed.rotation(0.2, 1), td.Transformed.rotation(0.2, [0, 3, 0]))
    assert np.allclose(td.Transformed.rotation(0.3, 2), td.Transformed.rotation(0.3, [0, 0, 4]))
    # Negative axis direction means negative angle
    assert np.allclose(td.Transformed.rotation(0.1, 0), td.Transformed.rotation(-0.1, [-2, 0, 0]))
    assert np.allclose(td.Transformed.rotation(0.2, 1), td.Transformed.rotation(-0.2, [0, -3, 0]))
    assert np.allclose(td.Transformed.rotation(0.3, 2), td.Transformed.rotation(-0.3, [0, 0, -4]))


def test_general_reflection():
    # Magnitude of normal direction does not affect the transformation.
    assert np.allclose(td.Transformed.reflection((1, 1, 0)), td.Transformed.reflection((5, 5, 0)))
    assert np.allclose(td.Transformed.reflection((1, 0, 1)), td.Transformed.reflection((5, 0, 5)))
    assert np.allclose(td.Transformed.reflection((0, 1, 1)), td.Transformed.reflection((0, 5, 5)))
    # Negative normal direction means the same transformation.
    assert np.allclose(td.Transformed.reflection((1, 1, 0)), td.Transformed.reflection((-1, -1, 0)))
    assert np.allclose(td.Transformed.reflection((1, 0, 1)), td.Transformed.reflection((-1, 0, -1)))
    assert np.allclose(td.Transformed.reflection((0, 1, 1)), td.Transformed.reflection((0, -1, -1)))
    # Magnitude and sign of normal direction does not affect the transformation.
    assert np.allclose(td.Transformed.reflection((1, 1, 0)), td.Transformed.reflection((-5, -5, 0)))
    assert np.allclose(td.Transformed.reflection((1, 0, 1)), td.Transformed.reflection((-5, 0, -5)))
    assert np.allclose(td.Transformed.reflection((0, 1, 1)), td.Transformed.reflection((0, -5, -5)))


def test_flattening():
    flat = list(
        flatten_groups(
            td.GeometryGroup(
                geometries=(
                    td.Box(size=(1, 1, 1)),
                    td.Box(size=(0, 1, 0)),
                    td.ClipOperation(
                        operation="union",
                        geometry_a=td.Box(size=(0, 0, 1)),
                        geometry_b=td.GeometryGroup(
                            geometries=(
                                td.Box(size=(2, 2, 2)),
                                td.GeometryGroup(
                                    geometries=(td.Box(size=(3, 3, 3)), td.Box(size=(3, 0, 3)))
                                ),
                            )
                        ),
                    ),
                )
            )
        )
    )
    assert len(flat) == 6
    assert all(isinstance(g, td.Box) for g in flat)

    flat = list(
        flatten_groups(
            td.GeometryGroup(
                geometries=(
                    td.Box(size=(1, 1, 1)),
                    td.Box(size=(0, 1, 0)),
                    td.ClipOperation(
                        operation="intersection",
                        geometry_a=td.Box(size=(0, 0, 1)),
                        geometry_b=td.GeometryGroup(
                            geometries=(
                                td.Box(size=(2, 2, 2)),
                                td.GeometryGroup(
                                    geometries=(td.Box(size=(3, 3, 3)), td.Box(size=(3, 0, 3)))
                                ),
                            )
                        ),
                    ),
                )
            )
        )
    )
    assert len(flat) == 3
    assert all(
        isinstance(g, td.Box) or (isinstance(g, td.ClipOperation) and g.operation == "intersection")
        for g in flat
    )

    t0 = np.array([[2, 0, 0, 0], [3, 2, 0, 0], [1, 0, 2, 0], [0, 0, 0, 1.0]])
    g0 = td.Sphere(radius=1)
    t1 = np.array([[2, 0, 5, 0], [0, 1, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 1.0]])
    g1 = td.Box(size=(1, 2, 3))
    flat = list(
        flatten_groups(
            td.Transformed(
                transform=t0,
                geometry=td.ClipOperation(
                    operation="union",
                    geometry_a=g0,
                    geometry_b=td.Transformed(transform=t1, geometry=g1),
                ),
            ),
            flatten_transformed=True,
        )
    )
    assert len(flat) == 2

    assert isinstance(flat[0], td.Transformed)
    assert flat[0].geometry == g0
    assert np.allclose(flat[0].transform, t0)

    assert isinstance(flat[1], td.Transformed)
    assert flat[1].geometry == g1
    assert np.allclose(flat[1].transform, t0 @ t1)


def test_geometry_traversal():
    geometries = list(traverse_geometries(td.Box(size=(1, 1, 1))))
    assert len(geometries) == 1

    geo_tree = td.GeometryGroup(
        geometries=(
            td.Box(size=(1, 0, 0)),
            td.ClipOperation(
                operation="intersection",
                geometry_a=td.GeometryGroup(
                    geometries=(
                        td.Box(size=(5, 0, 0)),
                        td.Box(size=(6, 0, 0)),
                    )
                ),
                geometry_b=td.ClipOperation(
                    operation="difference",
                    geometry_a=td.Box(size=(7, 0, 0)),
                    geometry_b=td.Box(size=(8, 0, 0)),
                ),
            ),
            td.GeometryGroup(
                geometries=(
                    td.Box(size=(3, 0, 0)),
                    td.Box(size=(4, 0, 0)),
                )
            ),
            td.Box(size=(2, 0, 0)),
        )
    )
    geometries = list(traverse_geometries(geo_tree))
    assert len(geometries) == 13


""" geometry """


def test_geometry():
    _ = td.Box(size=(1, 1, 1), center=(0, 0, 0))
    _ = td.Sphere(radius=1, center=(0, 0, 0))
    _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=1)
    _ = td.PolySlab(vertices=((1, 2), (3, 4), (5, 4)), slab_bounds=(-1, 1), axis=2)
    # vertices_np = np.array(s.vertices)
    # _ = PolySlab(vertices=vertices_np, slab_bounds=(-1, 1), axis=1)

    # make sure wrong axis arguments error
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=-1, length=1)
    with pytest.raises(pd.ValidationError):
        _ = td.PolySlab(radius=1, center=(0, 0, 0), axis=-1, slab_bounds=(-0.5, 0.5))
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pd.ValidationError):
        _ = td.PolySlab(radius=1, center=(0, 0, 0), axis=3, slab_bounds=(-0.5, 0.5))

    # make sure negative values error
    with pytest.raises(pd.ValidationError):
        _ = td.Sphere(radius=-1, center=(0, 0, 0))
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(radius=-1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pd.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=-1)


def test_geometry_sizes():
    # negative in size kwargs errors
    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pd.ValidationError):
            _ = td.Box(size=size, center=(0, 0, 0))
        with pytest.raises(pd.ValidationError):
            _ = td.Simulation(size=size, run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

    # negative grid sizes error?
    with pytest.raises(pd.ValidationError):
        _ = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.uniform(dl=-1.0), run_time=1e-12)


@pytest.mark.parametrize("x0", [5])
def test_geometry_touching_intersections_plane(x0):
    """Two touching boxes should show at least one intersection at plane where they touch."""

    # size of each box
    # L = 1 # works
    # L = 0.1 # works
    # L = 0.12 # assertion errors
    L = 0.24  # assertion errors
    # L = 0.25 # works

    # one box to the left of x0 and one box to the right of x0, touching at x0
    b1 = td.Box(center=(x0 - L / 2, 0, 0), size=(L, L, L))
    b2 = td.Box(center=(x0 + L / 2, 0, 0), size=(L, L, L))

    ints1 = b1.intersections_plane(x=x0)
    ints2 = b2.intersections_plane(x=x0)

    ints_total = ints1 + ints2

    assert len(ints_total) > 0, "no intersections found at plane where two boxes touch"


def test_pop_axis():
    b = td.Box(size=(1, 1, 1))
    for axis in range(3):
        coords = (1, 2, 3)
        Lz, (Lx, Ly) = b.pop_axis(coords, axis=axis)
        _coords = b.unpop_axis(Lz, (Lx, Ly), axis=axis)
        assert all(c == _c for (c, _c) in zip(coords, _coords))
        _Lz, (_Lx, _Ly) = b.pop_axis(_coords, axis=axis)
        assert Lz == _Lz
        assert Lx == _Lx
        assert Ly == _Ly


def test_2b_box_intersections():
    plane = td.Box(size=(1, 4, 0))
    box1 = td.Box(size=(1, 1, 1))
    box2 = td.Box(size=(1, 1, 1), center=(3, 0, 0))

    result = plane.intersections_with(box1)
    assert len(result) == 1
    assert result[0].geom_type == "Polygon"
    assert len(plane.intersections_with(box2)) == 0

    with pytest.raises(ValidationError):
        _ = box1.intersections_with(box2)

    assert len(box1.intersections_2dbox(plane)) == 1
    assert len(box2.intersections_2dbox(plane)) == 0

    with pytest.raises(ValidationError):
        _ = box2.intersections_2dbox(box1)


def test_2d_box_intersections_relaxed_for_small_transformed_2d_offset():
    z_expected = 0.3
    z_offset = float(increment_float(z_expected, 1.0))
    plane = td.Box(center=(0, 0, z_expected), size=(4, 4, 0))
    geometry = td.Transformed(
        geometry=td.Transformed(
            geometry=td.Box(center=(0, 0, z_expected), size=(2, 1, 0)),
            transform=td.Transformed.rotation(0.37, 2),
        ),
        transform=td.Transformed.translation(0.0, 0.0, z_offset - z_expected),
    )

    assert len(plane.intersections_with(geometry)) == 0
    assert len(plane.intersections_with(geometry, section_tolerance_2d=True)) == 1


def test_geometry_group_intersections_plane_relaxed_for_small_transformed_2d_offset():
    z_expected = 0.3
    z_offset = float(increment_float(z_expected, 1.0))
    geometry = td.Transformed(
        geometry=td.Transformed(
            geometry=td.Box(center=(0, 0, z_expected), size=(2, 1, 0)),
            transform=td.Transformed.rotation(0.37, 2),
        ),
        transform=td.Transformed.translation(0.0, 0.0, z_offset - z_expected),
    )
    group = td.GeometryGroup(
        geometries=(
            geometry,
            td.Box(center=(0, 0, 20), size=(1, 1, 1)),
        )
    )

    assert len(group.intersections_plane(z=z_expected)) == 0
    assert len(group.intersections_plane(z=z_expected, section_tolerance_2d=True)) == 1


def test_polyslab_merge():
    """make sure polyslabs from gds get merged when they should."""

    def make_polyslabs(gap_size):
        """Construct two rectangular polyslabs separated by a gap."""
        cell = gdstk.Cell(f"polygons_{gap_size:.2f}")
        rect1 = gdstk.rectangle((gap_size / 2, 0), (1, 1))
        rect2 = gdstk.rectangle((-1, 0), (-gap_size / 2, 1))
        cell.add(rect1, rect2)
        return td.PolySlab.from_gds(gds_cell=cell, gds_layer=0, axis=2, slab_bounds=(-1, 1))

    polyslabs_gap = make_polyslabs(gap_size=0.3)
    assert len(polyslabs_gap) == 2, "untouching polyslabs were merged incorrectly."

    polyslabs_touching = make_polyslabs(gap_size=0)
    assert len(polyslabs_touching) == 1, "polyslabs didn't merge correctly."


def test_polyslab_side_plot_merge():
    """In side plot, make sure splitted polygons merge."""
    x0 = 2
    y0 = 4
    z0 = 1
    R = 5
    wg_width = 0.5
    wg_thickness = 0.22
    sidewall_angle = 15 * np.pi / 180

    cell = gdstk.Cell("bottom")
    path_bottom = gdstk.RobustPath(
        (x0 + R, y0), wg_width - wg_thickness * np.tan(np.abs(sidewall_angle)), layer=1, datatype=0
    )

    path_bottom.arc(R, 0, -np.pi)
    cell.add(path_bottom)
    ring_bottom_geo = td.PolySlab.from_gds(
        cell,
        gds_layer=1,
        axis=2,
        slab_bounds=(z0 - wg_thickness / 2, z0 + wg_thickness / 2),
        sidewall_angle=sidewall_angle,
        reference_plane="top",
    )
    assert len(ring_bottom_geo[0].intersections_plane(x=2)) == 1


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_polyslab_axis(axis):
    ps = td.PolySlab(slab_bounds=(-1, 1), vertices=((-5, -5), (-5, 5), (5, 5), (5, -5)), axis=axis)

    # bound test
    bounds_ideal = [-5, -5]
    bounds_ideal.insert(axis, -1)
    bounds_ideal = np.array(bounds_ideal)
    np.allclose(ps.bounds[0], bounds_ideal)
    np.allclose(ps.bounds[1], -bounds_ideal)

    # inside
    point = [0, 0]
    point.insert(axis, 3)
    assert not ps.inside(point[0], point[1], point[2])

    # intersections
    plane_coord = [None] * 3
    plane_coord[axis] = 3
    assert not ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2])
    plane_coord[axis] = -3
    assert not ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2])


def test_polyslab_intersection_inf_bounds():
    """Test if intersection returns correct shapes when one of the slab_bounds is Inf."""
    # 1) [0, inf]
    poly = td.PolySlab(
        vertices=[[2, -1], [-2, -1], [-2, 1], [2, 1]],
        slab_bounds=[0, td.inf],
    )
    assert len(poly.intersections_plane(x=0)) == 1
    assert poly.intersections_plane(x=0)[0] == shapely.box(-1, 0.0, 1, LARGE_NUMBER)

    # 2) [-inf, 0]
    poly = poly.updated_copy(slab_bounds=(-td.inf, 0))
    assert len(poly.intersections_plane(x=0)) == 1
    assert poly.intersections_plane(x=0)[0] == shapely.box(-1, -LARGE_NUMBER, 1, 0)


def test_polyslab_intersection_with_coincident_plane():
    """Test if intersection returns the correct shape when the plane is coincident with the side face."""
    poly = td.PolySlab(
        vertices=[[500.0, -7500.0], [500.0, 7500.0], [-500.0, 7500.0], [-500.0, -7500.0]],
        slab_bounds=[0, 50],
        axis=2,
    )
    # Each case should give one side face of the polyslab
    expected_x_face = shapely.box(-7500, 0, 7500, 50)  # y-extent × z-extent
    expected_y_face = shapely.box(-500, 0, 500, 50)  # x-extent × z-extent

    assert poly.intersections_plane(x=-500) == [expected_x_face]
    assert poly.intersections_plane(x=500) == [expected_x_face]
    assert poly.intersections_plane(y=-7500) == [expected_y_face]
    assert poly.intersections_plane(y=7500) == [expected_y_face]


def test_polyslab_intersection_rotated_square():
    """Test PolySlab plane intersection with a rotated square (diamond shape)."""
    # Create a diamond by rotating a square 45 degrees
    size = 2.0
    angle = np.pi / 4
    base_vertices = np.array(
        [[-size / 2, -size / 2], [size / 2, -size / 2], [size / 2, size / 2], [-size / 2, size / 2]]
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = base_vertices @ rotation.T
    rotated = rotated - rotated.min(axis=0) + 0.5  # shift to positive quadrant
    vertices = [tuple(v) for v in rotated]

    polyslab = td.PolySlab(vertices=vertices, slab_bounds=(0, 3), axis=2)

    all_verts = np.array(vertices)
    left_tip_x = all_verts[:, 0].min()
    bottom_tip_y = all_verts[:, 1].min()
    x_center = (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2

    # Test 1: Cut at z=1.5 (middle of slab) - should give full diamond
    cross_section = polyslab.intersections_plane(z=1.5)
    assert len(cross_section) == 1
    assert np.isclose(cross_section[0].area, 4.0)

    # Test 2: Cut through center at x=x_center - should give rectangle
    cross_section = polyslab.intersections_plane(x=x_center)
    assert len(cross_section) == 1
    assert cross_section[0].area > 0

    # Test 3: Cut at left corner tip (tangent touch) - should give degenerate shape
    cross_section = polyslab.intersections_plane(x=left_tip_x)
    assert len(cross_section) == 1
    assert np.isclose(cross_section[0].area, 0.0)

    # Test 4: Cut near left corner (slightly inside) - should give small shape
    cross_section = polyslab.intersections_plane(x=left_tip_x + 0.3)
    assert len(cross_section) == 1
    assert cross_section[0].area > 0

    # Test 5: Cut at bottom corner (tangent touch) - should give degenerate shape
    cross_section = polyslab.intersections_plane(y=bottom_tip_y)
    assert len(cross_section) == 1
    assert np.isclose(cross_section[0].area, 0.0)

    # Test 6: Cut at z=0 (bottom boundary) - should give full diamond
    cross_section = polyslab.intersections_plane(z=0)
    assert len(cross_section) == 1
    assert np.isclose(cross_section[0].area, 4.0)


def test_from_shapely():
    ring = shapely.LinearRing([(-16, 9), (-8, 9), (-12, 2)])
    poly = shapely.Polygon([(-2, 0), (-10, 0), (-6, 7)])
    hole = shapely.Polygon(
        [(0, 0), (9, 0), (9, 9), (0, 9), (0, 2), (2, 2), (2, 7), (7, 7), (7, 2), (0, 2)]
    ).buffer(0)
    collection = shapely.GeometryCollection((shapely.MultiPolygon((poly,)), hole, ring))

    geo = td.Geometry.from_shapely(collection, 2, (0, 1))
    assert len(geo.intersections_plane(z=0.5)) == 3

    geo = td.Geometry.from_shapely(
        collection, 2, (0, 1), sidewall_angle=1.0, reference_plane="bottom"
    )
    assert len(geo.intersections_plane(z=0)) == 3
    assert len(geo.intersections_plane(z=1)) == 2

    geo = td.Geometry.from_shapely(
        collection, 2, (0, 1), sidewall_angle=-1.0, reference_plane="top"
    )
    assert len(geo.intersections_plane(z=0)) == 2
    assert len(geo.intersections_plane(z=1)) == 3


def test_from_gds():
    ring = gdstk.Polygon([(-16, 9), (-8, 9), (-12, 2)], layer=1)
    poly = gdstk.Polygon([(-2, 0), (-10, 0), (-6, 7)])
    hole = gdstk.Polygon(
        [(0, 0), (9, 0), (9, 9), (0, 9), (0, 2), (2, 2), (2, 7), (7, 7), (7, 2), (0, 2)]
    )
    cell = gdstk.Cell("CELL").add(ring, poly, hole)
    geo = td.Geometry.from_gds(
        cell, 2, (0, 1), gds_layer=0, dilation=-0.5, sidewall_angle=0.5, reference_plane="bottom"
    )
    assert len(geo.intersections_plane(z=0)) == 2
    assert len(geo.intersections_plane(z=1)) == 1


@pytest.mark.parametrize("geometry", GEO_TYPES)
def test_to_gds(geometry, tmp_path):
    fname = str(tmp_path / f"{geometry.__class__.__name__}.gds")
    geometry.to_gds_file(fname, z=0, gds_cell_name=geometry.__class__.__name__)
    cell = gdstk.read_gds(fname).cells[0]
    assert cell.name == geometry.__class__.__name__
    assert len(cell.polygons) > 0

    fname = str(tmp_path / f"{geometry.__class__.__name__}-empty.gds")
    geometry.to_gds_file(fname, y=1e30, gds_cell_name=geometry.__class__.__name__)
    cell = gdstk.read_gds(fname).cells[0]
    assert cell.name == geometry.__class__.__name__
    assert len(cell.polygons) == 0


def test_custom_surface_geometry(tmp_path):
    # create tetrahedron STL
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    geom = td.TriangleMesh.from_trimesh(tetrahedron)

    # test import
    import_geom = td.TriangleMesh.from_stl("tests/data/tetrahedron.stl")
    assert np.allclose(import_geom.triangles, geom.triangles)

    # test export and then import
    geom.trimesh.export(str(tmp_path / "export.stl"))
    import_geom = td.TriangleMesh.from_stl(str(tmp_path / "export.stl"))
    assert np.allclose(import_geom.triangles, geom.triangles)

    # assert np.array_equal(tetrahedron.vectors, export_vectors)

    areas = [0.5 * np.sqrt(2) * np.sqrt(1 + 2 * 0.5**2), 0.5, 0.5, 0.5]
    unit_normals_unnormalized = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    unit_normals = [n / np.linalg.norm(n) for n in unit_normals_unnormalized]
    _ = [n * a for (n, a) in zip(unit_normals, areas)]

    # test bounds
    assert np.allclose(np.array(geom.bounds), [[0, 0, 0], [1, 1, 1]])

    # test surface area
    assert np.isclose(geom.surface_area(), np.sum(areas))

    # test volume
    assert np.isclose(geom.volume(), 1 / 6)

    # test intersections
    assert shapely.equals(geom.intersections_plane(x=0), shapely.Polygon([[0, 0], [0, 1], [1, 0]]))
    assert shapely.equals(
        geom.intersections_plane(z=0.5), shapely.Polygon([[0, 0], [0, 0.5], [0.5, 0]])
    )

    # test inside
    assert geom.inside([0.2], [0.2], [0.2])[0]
    assert not geom.inside([0.8], [0.2], [0.2])[0]

    # test plot
    _, ax = plt.subplots()
    _ = geom.plot(z=0.1, ax=ax)
    plt.close()

    # test inconsistent winding
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[2, 3, 1], [0, 2, 3], [0, 3, 1], [0, 1, 2]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel("WARNING", contains_str="face orientations"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    with AssertLogLevel(None):
        geom = geom.fix_winding()

    # test non-watertight mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel("WARNING", contains_str="watertight"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    with AssertLogLevel(None):
        geom = geom.fill_holes()

    # test inward normals
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[2, 1, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel("WARNING", contains_str="outward"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    with AssertLogLevel(None):
        geom = geom.fix_normals()

    # test zero area triangles
    vertices = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel("WARNING"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    assert all(np.array(geom.trimesh.area_faces) > AREA_SIZE_THRESHOLD)

    # test trimesh.Scene
    import_geom = td.TriangleMesh.from_stl("tests/data/two_boxes_separate.stl")
    sim = sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[],
        structures=[td.Structure(geometry=import_geom, medium=td.Medium(permittivity=2))],
        monitors=[],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
    )
    _, ax = plt.subplots()
    _ = sim.plot(y=0, ax=ax)
    plt.close()

    # allow small triangles
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vertices_small = vertices * 1e-6
    td.TriangleMesh.from_vertices_faces(vertices_small, faces)


@pytest.mark.parametrize("binary", [True, False])
def test_triangle_mesh_to_stl_roundtrip(tmp_path, binary):
    """Exporting with 'to_stl' should be readable via 'from_stl'."""

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    mesh = td.TriangleMesh.from_vertices_faces(vertices, faces)

    export_path = tmp_path / ("mesh_binary.stl" if binary else "mesh_ascii.stl")
    mesh.to_stl(str(export_path), binary=binary)

    roundtrip = td.TriangleMesh.from_stl(str(export_path))

    assert np.allclose(roundtrip.triangles, mesh.triangles)


def test_geo_group_sim():
    geo_grp = td.TriangleMesh.from_stl("tests/data/two_boxes_separate.stl")
    geos_orig = list(geo_grp.geometries)
    geo_grp_full = geo_grp.updated_copy(geometries=(*geos_orig, td.Box(size=(1, 1, 1))))

    sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[],
        structures=[td.Structure(geometry=geo_grp_full, medium=td.Medium(permittivity=2))],
        monitors=[],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
    )

    # why is this failing?  assert 4==2
    assert len(sim.custom_datasets) == len(geos_orig)


def test_finite_geometry_transformation():
    with pytest.raises(pd.ValidationError):
        _ = td.Box(size=(td.inf, 0, 1)).scaled(1, 1, 1)


def test_update_from_bounds():
    # Test the functionality for updating bounds of geometries that support 2d materials
    box2d = td.Box(size=(1, 1, 0))
    polyslab2d = td.PolySlab(
        vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(0.5, 0.5), axis=2
    )
    cylinder2d = td.Cylinder(axis=2, length=0, radius=1, center=(0, 0, 0.5))
    geo_group2d = td.GeometryGroup(geometries=(cylinder2d, polyslab2d))
    clip2d = td.ClipOperation(operation="union", geometry_a=cylinder2d, geometry_b=polyslab2d)

    # Some test transformations that preserve the normal
    translate = td.Transformed.translation(x=0, y=0, z=1)
    rotate = td.Transformed.rotation(angle=np.pi * (1 / 8), axis=2)
    scale = td.Transformed.scaling(x=2, y=2, z=1)
    shift = td.Transformed(geometry=cylinder2d, transform=translate)
    shift_rotate = td.Transformed(geometry=shift, transform=rotate)
    transformed_2d = td.Transformed(geometry=shift_rotate, transform=scale)

    new_bounds = (3.2, 6.4)
    axis = 2

    geometries = [
        box2d,
        polyslab2d,
        cylinder2d,
        geo_group2d,
        clip2d,
        shift,
        shift_rotate,
        transformed_2d,
    ]
    for geom2d in geometries:
        geom_update = geom2d._update_from_bounds(bounds=new_bounds, axis=axis)
        test_bounds = (geom_update.bounds[0][axis], geom_update.bounds[1][axis])
        assert np.isclose(test_bounds, new_bounds).all()

    # By default geometries should raise a NotImplementedError if they are not supported
    sphere = td.Sphere(radius=1, center=(0, 0, 0.5))
    geometries = [sphere]
    for geom2d in geometries:
        with pytest.raises(NotImplementedError):
            geom_update = geom2d._update_from_bounds(bounds=new_bounds, axis=axis)


def test_subdivide():
    # Test the functionality that subdivides structures with Medium2D into partitions,
    # where each partition is paired with homogeneous medium above and below
    box = td.Box(size=(1, 1, 0))
    overlap_box = td.Box(size=(2, 0.5, 0), center=(0.5, 0, 0))
    # These overlapping boxes have their left edge coincident, and the second box has a smaller left edge.
    # This results in an invalid geometry for MultiPolygon which must be fixed by applying a union
    # operation in ``subdivide``. This should fix other types of invalid geometries as well.
    overlapping_boxes = td.GeometryGroup(geometries=(box, overlap_box))

    background_structure = td.Structure(medium=td.Medium(), geometry=td.Box(size=(10, 10, 10)))
    subdivisions = subdivide(geom=overlapping_boxes, structures=(background_structure,))
    assert len(subdivisions) == 1

    # Test that when a small sliver is created during subdivide
    # it gets correctly removed before creating the Polyslab
    box_sliver = td.Structure(
        medium=td.Medium(), geometry=td.Box(size=(1, 1, 1), center=(1 - fp_eps, 0, 0))
    )
    subdivisions = subdivide(geom=overlapping_boxes, structures=(background_structure, box_sliver))


def test_subdivide_large_geometry_sliver_filter():
    """Test that sliver polygons are filtered based on grid cell size for large geometries.

    When geometries have large coordinates, floating-point precision can create thin
    "sliver" polygons during boolean operations. These should be filtered out when
    their dimensions are much smaller than the grid cell size.
    """
    # Create a 2D geometry at large coordinates (similar to real-world case)
    large_offset = 7000.0
    geom = td.Box(size=(100, 35, 0), center=(0, large_offset, 0))

    # Create background structure that covers the geometry
    background = td.Structure(
        medium=td.Medium(), geometry=td.Box(size=(200, 200, 200), center=(0, large_offset, 0))
    )

    # Create a structure that partially overlaps the 2D geometry.
    # Position it so that boolean operations produce a tiny sliver at the edge
    # due to floating-point precision at large coordinates.
    # The sliver width will be on the order of fp_eps * coordinate_value ~ 1e-7 * 7000 ~ 0.001
    sliver_offset = fp_eps * large_offset * 10  # ~0.001
    overlapping_box = td.Structure(
        medium=td.Medium(permittivity=2.0),
        geometry=td.Box(
            size=(50, 35 + sliver_offset, 100),
            center=(25, large_offset + sliver_offset / 2, 50),
        ),
    )

    # Create a grid with cell sizes much larger than the sliver
    # Grid cell size of 1.0 means sliver with dim < 0.0001 should be filtered
    # (threshold = 1.0 * _SLIVER_DIM_RTOL where _SLIVER_DIM_RTOL = 1e-4)
    x_coords = np.linspace(-100, 100, 201)  # dx = 1.0
    y_coords = np.linspace(large_offset - 50, large_offset + 50, 101)  # dy = 1.0
    z_coords = np.linspace(-50, 50, 101)  # dz = 1.0
    grid = td.Grid(boundaries=td.Coords(x=x_coords, y=y_coords, z=z_coords))

    # Without grid-based filtering, this might create multiple subdivisions including slivers
    # With grid-based filtering, slivers should be removed
    subdivisions = subdivide(geom=geom, structures=[background, overlapping_box], grid=grid)

    # Should have at most 2 subdivisions (the main regions, not slivers)
    # In practice, for this setup we expect the subdivisions to be clean
    # and not contain tiny sliver artifacts
    assert len(subdivisions) == 2

    # Verify that no subdivision has a dimension smaller than the sliver threshold
    for subdivision in subdivisions:
        subdiv_geom = subdivision[0]
        bounds = subdiv_geom.bounds
        for dim in range(3):
            dim_size = bounds[1][dim] - bounds[0][dim]
            if dim_size > 0:  # Skip zero-thickness dimensions (the 2D plane normal)
                # The dimension should be at least comparable to grid cell size
                # (not a tiny sliver)
                assert dim_size >= 0.01, f"Found sliver with dimension {dim_size} in axis {dim}"


def test_is_sliver_polygon_grid_based_filtering():
    """Test _is_sliver_polygon function with grid-based relative thresholds.

    This tests both the area-based and dimension-based sliver detection when a grid
    is provided.
    """
    # Create a grid with cell size 1.0 in x and y directions
    # For axis=2 (z-normal plane), tangential axes are x and y
    x_coords = np.linspace(0, 10, 11)  # dx = 1.0
    y_coords = np.linspace(0, 10, 11)  # dy = 1.0
    z_coords = np.linspace(0, 10, 11)  # dz = 1.0
    grid = td.Grid(boundaries=td.Coords(x=x_coords, y=y_coords, z=z_coords))

    # With cell size 1.0:
    # - Area threshold = 1.0 * 1.0 * 1e-4 = 1e-4
    # - Dimension threshold = 1.0 * 1e-4 = 1e-4

    # Test 1: Polygon with area below relative threshold (hits line 130)
    # Small square polygon with sides 0.005 -> area = 2.5e-5 < 1e-4
    # Both dimensions (0.005) are > 1e-4, so it won't be caught by dimension check
    small_area_polygon = shapely.Polygon([(0, 0), (0.005, 0), (0.005, 0.005), (0, 0.005)])
    assert small_area_polygon.area < 1e-4  # Verify area is below threshold
    assert _is_sliver_polygon(small_area_polygon, axis=2, grid=grid) is True

    # Test 2: Thin polygon with one dimension below relative threshold (hits line 138)
    # Very thin but long polygon: 0.00005 x 10 -> area = 0.0005 >= 1e-4
    # But one dimension (0.00005) is < 1e-4
    thin_polygon = shapely.Polygon([(0, 0), (10, 0), (10, 0.00005), (0, 0.00005)])
    assert thin_polygon.area >= 1e-4  # Area is above threshold
    assert _is_sliver_polygon(thin_polygon, axis=2, grid=grid) is True

    # Test 3: Normal polygon that should NOT be filtered
    # Square polygon with sides 1.0 -> area = 1.0, both dims = 1.0
    normal_polygon = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert _is_sliver_polygon(normal_polygon, axis=2, grid=grid) is False

    # Test 4: Without grid, only absolute threshold applies
    # The small_area_polygon has area 2.5e-5, which may be above _MIN_POLYGON_AREA
    # depending on its value, so let's test with a polygon that's definitely small
    # but above absolute minimum
    medium_polygon = shapely.Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])
    # Without grid, this should NOT be filtered (area 1e-4 is above absolute minimum)
    assert _is_sliver_polygon(medium_polygon, axis=2, grid=None) is False
    # With grid, this should be filtered (area 1e-4 = threshold, but < due to floating point)
    # Actually 1e-4 == 1e-4, so it won't be filtered. Let's use slightly smaller.
    smaller_polygon = shapely.Polygon([(0, 0), (0.009, 0), (0.009, 0.009), (0, 0.009)])
    # Area = 8.1e-5 < 1e-4, so should be filtered with grid
    assert _is_sliver_polygon(smaller_polygon, axis=2, grid=grid) is True


def test_subdivide_geometry_group_with_polygon_holes():
    """Test that unionized geometry containing a hole works correctly."""
    mm = 1000.0

    # Create four boxes arranged to form a cross pattern with a square hole in the middle
    box_a = td.Box.from_bounds((-10 * mm, -10 * mm, 0 * mm), (-5 * mm, 5 * mm, 0 * mm))
    box_b = td.Box.from_bounds((-10 * mm, 5 * mm, 0 * mm), (5 * mm, 10 * mm, 0 * mm))
    box_c = td.Box.from_bounds((5 * mm, -5 * mm, 0 * mm), (10 * mm, 10 * mm, 0 * mm))
    box_d = td.Box.from_bounds((-5 * mm, -10 * mm, 0 * mm), (10 * mm, -5 * mm, 0 * mm))

    geom_group = td.GeometryGroup(geometries=(box_a, box_b, box_c, box_d))
    geom_structures_group = [td.Structure(geometry=geom_group, medium=td.PEC2D)]

    feed_pin_bottom = -2 * mm
    feed_pin_top = 0 * mm
    feed_pin_center = 0.5 * (feed_pin_top + feed_pin_bottom)
    feed_pin_length = feed_pin_top - feed_pin_bottom
    feed_center_x = -7.5 * mm
    feed_center_y = -7.5 * mm
    rfeed = 1.0 * mm

    feed_pin = td.Structure(
        geometry=td.Cylinder(
            center=(feed_center_x, feed_center_y, feed_pin_center),
            radius=rfeed,
            length=feed_pin_length,
            axis=2,
        ),
        medium=td.PECMedium(),
    )

    structures_list = [feed_pin, *geom_structures_group]

    freq = 1500 * 1e6
    dl = (td.C_0 / freq) / 300.0

    mesh_overrides = [
        td.MeshOverrideStructure(
            geometry=td.Box(
                center=(0, 0, 0 * mm),
                size=(20 * mm, 20 * mm, 6 * mm),
            ),
            dl=[dl, dl, dl],
        )
    ]

    sim = td.Simulation(
        size=[100 * mm, 100 * mm, 30 * mm],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,
            wavelength=td.C_0 / freq,
            override_structures=mesh_overrides,
        ),
        structures=structures_list,
        run_time=1e-13,
    )

    contains_difference_operation = False
    for structure in sim._finalized.structures:
        geo = structure.geometry
        if isinstance(geo, td.ClipOperation) and geo.operation == "difference":
            contains_difference_operation = True
    assert contains_difference_operation


@pytest.mark.parametrize("snap_location", [SnapLocation.Boundary, SnapLocation.Center])
@pytest.mark.parametrize(
    "snap_behavior",
    [
        SnapBehavior.Off,
        SnapBehavior.Closest,
        SnapBehavior.Expand,
        SnapBehavior.Contract,
        SnapBehavior.StrictExpand,
        SnapBehavior.StrictContract,
    ],
)
def test_snap_box_to_grid(snap_location, snap_behavior):
    """ "Test that all combinations of SnappingSpec correctly modify a test box without error."""
    snap_spec = SnappingSpec(location=[snap_location] * 3, behavior=[snap_behavior] * 3)
    box = td.Box(center=(0, 0, 0), size=(0.2, 0.23, 0.1))

    xyz = np.linspace(0, 1, 11)
    coords = td.Coords(x=xyz, y=xyz, z=xyz)
    grid = td.Grid(boundaries=coords)
    new_box = snap_box_to_grid(grid, box, snap_spec)
    if snap_behavior != SnapBehavior.Off:
        # The box must have changed location
        assert not np.allclose(new_box.bounds, box.bounds)

    # Test box that is bigger than the grid.
    # Also test a corner case, where the box boundary and grid are approximately equal.
    box = td.Box(center=(0.5, 0.200000001, 0), size=(1.1, 0.2, 0.1))
    new_box = snap_box_to_grid(grid, box, snap_spec)

    if snap_behavior != SnapBehavior.Off and snap_location == SnapLocation.Boundary:
        # Strict behaviors have different snapping rules, so skip these specific assertions
        if snap_behavior not in (SnapBehavior.StrictExpand, SnapBehavior.StrictContract):
            # Check that the box boundary slightly off from 0.1 was correctly snapped to 0.1
            assert math.isclose(new_box.bounds[0][1], xyz[1])
            # Check that the box boundary slightly off from 0.3 was correctly snapped to 0.3
            assert math.isclose(new_box.bounds[1][1], xyz[3])
            # Check that the box boundary outside the grid was snapped to the smallest grid coordinate
            assert math.isclose(new_box.bounds[0][2], xyz[0])


def test_snap_box_to_grid_strict_behaviors():
    """Test StrictExpand and StrictContract behaviors specifically."""
    xyz = np.linspace(0, 1, 11)  # Grid points at 0.0, 0.1, 0.2, ..., 1.0
    coords = td.Coords(x=xyz, y=xyz, z=xyz)
    grid = td.Grid(boundaries=coords)

    # Test StrictExpand: should always move endpoints outwards, even if coincident
    box_coincident = td.Box(
        center=(0.1, 0.2, 0.3), size=(0, 0, 0)
    )  # Centered exactly on grid points
    snap_spec_strict_expand = SnappingSpec(
        location=[SnapLocation.Boundary] * 3, behavior=[SnapBehavior.StrictExpand] * 3
    )

    expanded_box = snap_box_to_grid(grid, box_coincident, snap_spec_strict_expand)

    # StrictExpand should move bounds outwards even when already on grid
    assert np.isclose(expanded_box.bounds[0][0], 0.0)  # Left bound moved left from 0.1
    assert np.isclose(expanded_box.bounds[1][0], 0.2)  # Right bound moved right from 0.1
    assert np.isclose(expanded_box.bounds[0][1], 0.1)  # Bottom bound moved down from 0.2
    assert np.isclose(expanded_box.bounds[1][1], 0.3)  # Top bound moved up from 0.2

    # Test StrictContract: should always move endpoints inwards, even if coincident
    box_large = td.Box(center=(0.5, 0.5, 0.5), size=(0.4, 0.4, 0.4))  # Spans multiple grid cells
    snap_spec_strict_contract = SnappingSpec(
        location=[SnapLocation.Boundary] * 3, behavior=[SnapBehavior.StrictContract] * 3
    )

    contracted_box = snap_box_to_grid(grid, box_large, snap_spec_strict_contract)

    # StrictContract should make the box smaller than the original
    assert contracted_box.size[0] < box_large.size[0]
    assert contracted_box.size[1] < box_large.size[1]
    assert contracted_box.size[2] < box_large.size[2]

    # Test edge case: box coincident with grid boundaries
    box_on_grid = td.Box(
        center=(0.15, 0.25, 0.35), size=(0.1, 0.1, 0.1)
    )  # Boundaries at 0.1,0.2 and 0.2,0.3

    # Regular Expand shouldn't change a box already coincident with grid
    snap_spec_regular_expand = SnappingSpec(
        location=[SnapLocation.Boundary] * 3, behavior=[SnapBehavior.Expand] * 3
    )
    regular_expanded = snap_box_to_grid(grid, box_on_grid, snap_spec_regular_expand)
    assert np.allclose(regular_expanded.bounds, box_on_grid.bounds)  # Should be unchanged

    # StrictExpand should still expand even when coincident
    strict_expanded = snap_box_to_grid(grid, box_on_grid, snap_spec_strict_expand)
    assert not np.allclose(strict_expanded.bounds, box_on_grid.bounds)  # Should be changed
    assert strict_expanded.size[0] > box_on_grid.size[0]  # Should be larger

    # Test with margin parameter for strict behaviors
    snap_spec_strict_expand_margin = SnappingSpec(
        location=[SnapLocation.Boundary] * 3,
        behavior=[SnapBehavior.StrictExpand] * 3,
        margin=(1, 1, 1),  # Consider 1 additional grid point when expanding
    )

    margin_expanded = snap_box_to_grid(grid, box_coincident, snap_spec_strict_expand_margin)
    # With margin=1, should expand even further than without margin
    assert margin_expanded.size[0] >= expanded_box.size[0]


def test_triangulation_with_collinear_vertices():
    xr = np.linspace(0, 1, 6)
    a = np.array([[x, -0.5] for x in xr] + [[x, 0.5] for x in xr[::-1]])
    assert len(td.components.geometry.triangulation.triangulate(a)) == 10


def test_triangle_mesh_from_height():
    """Test the TriangleMesh.from_height_function and from_height_grid constructors."""

    # Test successful creation with a valid height function
    def valid_height_func(x, y):
        return 0.5 + 0.2 * np.sin(4 * (x + 1)) * np.cos(3 * y)

    axis = 2
    direction = "+"
    base = 0.0
    center = [0, 0]
    size = [1.5, 2]
    grid_size = [20, 15]

    geometry_from_func = td.TriangleMesh.from_height_function(
        axis=axis,
        direction=direction,
        base=base,
        center=center,
        size=size,
        grid_size=grid_size,
        height_func=valid_height_func,
    )

    assert isinstance(geometry_from_func, td.TriangleMesh)

    # Test equivalence with from_height_grid method
    x = np.linspace(center[0] - 0.5 * size[0], center[0] + 0.5 * size[0], grid_size[0])
    y = np.linspace(center[1] - 0.5 * size[1], center[1] + 0.5 * size[1], grid_size[1])
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

    geometry_from_grid = td.TriangleMesh.from_height_grid(
        axis=axis,
        direction=direction,
        base=base,
        grid=(x, y),
        height=valid_height_func(x_mesh, y_mesh),
    )

    # Check if the two TriangleMesh objects are equivalent
    assert geometry_from_func == geometry_from_grid

    # Test ValueError for negative height values
    def negative_height_func(x, y):
        return 0.5 + 0.2 * np.sin(4 * (x + 1)) * np.cos(3 * y) - 2

    with pytest.raises(
        ValueError,
        match=r"All height values must be non-negative.",
    ):
        td.TriangleMesh.from_height_function(
            axis=axis,
            direction=direction,
            base=base,
            center=center,
            size=size,
            grid_size=grid_size,
            height_func=negative_height_func,
        )

    # Test ValueError for height_func returning ndarray with wrong shape
    def wrong_shape_height_func(x, y):
        return np.zeros((3, 3))  # Incorrect shape

    expected_shape = (grid_size[0], grid_size[1])

    # Test for the presence of key parts of the error message
    with pytest.raises(ValueError) as excinfo:
        td.TriangleMesh.from_height_function(
            axis=axis,
            direction=direction,
            base=base,
            center=center,
            size=size,
            grid_size=grid_size,
            height_func=wrong_shape_height_func,
        )
    # Check that the error message contains the expected information
    error_message = str(excinfo.value)
    assert f"shape {expected_shape}" in error_message
    assert "shape (3, 3)" in error_message


def test_cleanup_shapely_object():
    if _package_is_older_than("shapely", "2.1"):
        # (Old versions of shapely don't support `shapely.make_valid()` with the correct arguments.
        # However older alternatives like `.buffer(0)` are not as robust.  `.buffer(0)` is likely
        # to generate polygons which look correct, but have extra vertices, causing test to fail.
        # So if `shapely.make_valid()` is not supported, the safest thing to do is skip this test.)
        pytest.skip("This test requires `shapely` version 2.1 or later")

    # Test 1: A square containing a triangular hole, and an infinitley thin hole.
    square_with_spikes_5x5 = np.array(
        (
            (0, 0),
            (5, 0),
            (5, 0),
            (5, 10),  # this vertex creates an outward spike and should be removed
            (5, 5),
            (0, 5 - 1e-13),  # this vertex should be rounded to (0, 5)
            (3, 3),  # this vertex creates an inward spike and should be removed
            (0, 5 + 1e-13),  # this vertex should be removed because it duplicates (0, 5 - 1e-13)
        )
    )
    triangle_empty_tails = np.array(((1, 1), (3, 1), (2, 2), (2.5, 2.5), (0.5, 0.5)))
    triangle_collinear = np.array(((4, 2), (3, 3), (2, 4), (4, 2)))  # has zero area
    # NOTE: Test will fail for self intersecting polys like: ((0,0), (1,1), (2,-1), (3,1), (4,0))
    # Now build a shapely polygon with the 4 small polygons enclosed by big_square_5x5
    exterior_coords = square_with_spikes_5x5
    interior_coords_list = [
        triangle_empty_tails,
        triangle_collinear,  # this triangle should be eliminated
    ]
    # Test using a non-empty exterior polygon (big_square_5x5)
    orig_polygon = shapely.Polygon(exterior_coords, interior_coords_list)
    new_polygon = cleanup_shapely_object(orig_polygon, tolerance_ratio=1e-12)
    # Delete any nearby or overlapping vertices (cleanup_shapely_object() now does this).
    # Now `new_polygon` should only contain the coordinates of the square (with a duplicate at end).
    assert len(new_polygon.exterior.coords) == 5  # squares have 4 vertices but shapely adds 1
    assert len(new_polygon.interiors) == 1  # only the "triangle_empty_tails" interior hole survives
    assert len(new_polygon.interiors[0].coords) == 4  # triangles have 3 vertices but shapely adds 1
    # Test 2: An infinitely thin triangle exterior with some holes (which should be deleted)
    exterior_coords = triangle_collinear  # has zero area
    orig_polygon = shapely.Polygon(exterior_coords)
    new_polygon = cleanup_shapely_object(orig_polygon, tolerance_ratio=1e-12)
    assert len(new_polygon.exterior.coords) == 0  # empty / collinear polygons should get deleted


def test_flatten_shapely_geometries():
    """Test the flatten_shapely_geometries utility function comprehensively."""
    # Test 1: Single polygon (should be wrapped in list and returned)
    single_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = flatten_shapely_geometries(single_polygon)
    assert len(result) == 1
    assert result[0] == single_polygon

    # Test 2: List of polygons (should return as-is)
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    polygon_list = [poly1, poly2]
    result = flatten_shapely_geometries(polygon_list)
    assert len(result) == 2
    assert result == polygon_list

    # Test 3: MultiPolygon (should be flattened)
    multi_polygon = MultiPolygon([poly1, poly2])
    result = flatten_shapely_geometries(multi_polygon)
    assert len(result) == 2
    assert result[0] == poly1
    assert result[1] == poly2

    # Test 4: Empty geometries (should be filtered out)
    empty_polygon = Polygon()
    mixed_list = [poly1, empty_polygon, poly2]
    result = flatten_shapely_geometries(mixed_list)
    assert len(result) == 2
    assert empty_polygon not in result

    # Test 5: GeometryCollection (should be recursively flattened)
    line = LineString([(0, 0), (1, 1)])
    point = Point(0, 0)
    collection = GeometryCollection([poly1, line, point, poly2])
    result = flatten_shapely_geometries(collection)
    assert len(result) == 2  # Only polygons kept by default
    assert poly1 in result
    assert poly2 in result

    # Test 6: Custom keep_types parameter
    result_with_lines = flatten_shapely_geometries(collection, keep_types=(Polygon, LineString))
    assert len(result_with_lines) == 3  # 2 polygons + 1 line
    assert poly1 in result_with_lines
    assert poly2 in result_with_lines
    assert line in result_with_lines

    # Test 7: Nested collections and multi-geometries
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(2, 2), (3, 3)])
    multi_line = MultiLineString([line1, line2])
    nested_collection = GeometryCollection(
        [
            collection,  # Contains poly1, line, point, poly2
            multi_line,
            poly1,
        ]
    )
    result = flatten_shapely_geometries(nested_collection)
    assert len(result) == 3  # poly1 (from collection), poly2 (from collection), poly1 (direct)

    # Test 8: MultiPoint (should be handled)
    point1 = Point(0, 0)
    point2 = Point(1, 1)
    multi_point = MultiPoint([point1, point2])
    result = flatten_shapely_geometries(multi_point, keep_types=(Point,))
    assert len(result) == 2
    assert point1 in result
    assert point2 in result

    # Test 9: MultiLineString (should be handled)
    result = flatten_shapely_geometries(multi_line, keep_types=(LineString,))
    assert len(result) == 2
    assert line1 in result
    assert line2 in result

    # Test 10: Mixed empty and non-empty geometries
    empty_multi = MultiPolygon([])
    mixed_with_empty = [poly1, empty_multi, empty_polygon, poly2]
    result = flatten_shapely_geometries(mixed_with_empty)
    assert len(result) == 2
    assert poly1 in result
    assert poly2 in result

    # Test 11: Deeply nested structure
    inner_collection = GeometryCollection([poly1, line])
    outer_multi = MultiPolygon([poly2])
    deep_collection = GeometryCollection([inner_collection, outer_multi])
    result = flatten_shapely_geometries(deep_collection)
    assert len(result) == 2
    assert poly1 in result
    assert poly2 in result

    # Test 12: All geometry types filtered out
    points_and_lines = GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])])
    result = flatten_shapely_geometries(points_and_lines)  # Default keeps only Polygons
    assert len(result) == 0

    # Test 13: Edge case - single empty geometry
    result = flatten_shapely_geometries(empty_polygon)
    assert len(result) == 0


def test_merging_geometries_on_plane_overlapping_lines():
    """Test merging_geometries_on_plane with overlapping zero-area (LineString) geometries."""
    # Two thin (zero-thickness in y) boxes that overlap along x.
    # Box A: x in [-0.75, 0.25], Box B: x in [-0.25, 0.75]  =>  overlap in [-0.25, 0.25]
    geo_a = td.Box(center=(-0.25, 1, 0), size=(1, 0, 2))
    geo_b = td.Box(center=(0.25, 1, 0), size=(1, 0, 2))

    # XY cross-section plane at z=0
    plane = td.Box(center=(0, 1, 0), size=(4, 4, 0))

    # Same property => shapes should be merged into a single LineString
    prop = "PEC"
    results = merging_geometries_on_plane(
        geometries=[geo_a, geo_b],
        plane=plane,
        property_list=[prop, prop],
        interior_disjoint_geometries=True,
    )
    assert len(results) == 1
    result_prop, result_shape = results[0]
    assert result_prop == prop
    assert result_shape.geom_type == "LineString"
    minx, _, maxx, _ = result_shape.bounds
    assert minx == pytest.approx(-0.75)
    assert maxx == pytest.approx(0.75)

    # Different properties => two separate results, each unmodified
    results = merging_geometries_on_plane(
        geometries=[geo_a, geo_b],
        plane=plane,
        property_list=["PEC", "copper"],
        interior_disjoint_geometries=True,
    )
    assert len(results) == 2
    props_returned = {r[0] for r in results}
    assert props_returned == {"PEC", "copper"}


# ======================= GeometryArray Tests =======================


def test_geometry_array_basic():
    """Test GeometryArray creation, bounds, and inside methods."""
    box = td.Box(size=(1, 1, 1))
    offsets = [[0, 0, 0], [3, 0, 0]]
    array = td.GeometryArray(geometry=box, offsets=offsets)

    # Creation
    assert array.num_geometries == 2
    assert array.geometry == box

    # Bounds: boxes at (0,0,0) and (3,0,0)
    np.testing.assert_allclose(array.bounds[0], (-0.5, -0.5, -0.5))
    np.testing.assert_allclose(array.bounds[1], (3.5, 0.5, 0.5))

    # Inside with scalars
    assert array.inside(0, 0, 0)
    assert array.inside(3, 0, 0)
    assert not array.inside(1.5, 0, 0)

    # Inside with arrays
    result = array.inside(np.array([0, 1.5, 3]), np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(result, [True, False, True])

    # Convenience method
    array2 = box.array(offsets=offsets)
    assert isinstance(array2, td.GeometryArray)
    assert array2.num_geometries == 2


def test_geometry_array_with_transforms():
    """Test GeometryArray with transforms (with and without offsets)."""
    box = td.Box(size=(2, 1, 1))
    rotation = td.Transformed.rotation(np.pi / 2, 2)

    # Transforms only (no offsets)
    array = td.GeometryArray(geometry=box, transforms=[np.eye(4), rotation])
    assert array.num_geometries == 2
    assert array.offsets is None
    assert array.inside(0.5, 0, 0)  # First instance along x
    assert array.inside(0, 0.8, 0)  # Second instance rotated, long axis now along y

    # Transforms with offsets
    array = td.GeometryArray(
        geometry=box, offsets=[[0, 0, 0], [4, 0, 0]], transforms=[np.eye(4), rotation]
    )
    assert array.inside(0.5, 0, 0)  # First instance
    assert array.inside(4, 0.5, 0)  # Second instance rotated and translated


def test_geometry_array_both_none():
    """Test GeometryArray with both offsets and transforms as None."""
    box = td.Box(size=(1, 1, 1))
    array = td.GeometryArray(geometry=box)

    assert array.num_geometries == 1
    assert array.inside(0, 0, 0)
    assert not array.inside(1, 0, 0)
    np.testing.assert_allclose(array.bounds[0], (-0.5, -0.5, -0.5))


def _transform_with_translation():
    """Helper to create a transform matrix with translation (invalid for GeometryArray)."""
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]  # Add translation - not allowed in GeometryArray
    return t


def _transform_with_bad_bottom_row():
    """Helper to create a transform matrix with invalid bottom row."""
    t = np.eye(4)
    t[3, :] = [1, 0, 0, 1]  # Invalid bottom row
    return t


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        ({"offsets": []}, "at least one offset"),
        ({"transforms": []}, "at least one transform"),
        ({"offsets": [[0, 0], [2, 0]]}, None),  # Wrong shape
        ({"offsets": [0, 0, 0]}, None),  # 1D instead of 2D
        ({"offsets": [[0, 0, 0], [2, 0, 0]], "transforms": [np.eye(4)]}, "must match"),
        ({"offsets": [[0, 0, 0]], "transforms": [np.zeros((4, 4))]}, "singular"),
        # Linear-only transform validation
        ({"transforms": [_transform_with_translation()]}, "contains translation"),
        ({"transforms": [_transform_with_bad_bottom_row()]}, "invalid homogeneous form"),
    ],
)
def test_geometry_array_validation(kwargs, error_match):
    """Test GeometryArray validation errors."""
    box = td.Box(size=(1, 1, 1))
    with pytest.raises(pd.ValidationError):
        td.GeometryArray(geometry=box, **kwargs)


def test_geometry_array_validation_infinite_geometry():
    """Test validation rejects infinite geometry."""
    with pytest.raises(pd.ValidationError):
        td.GeometryArray(geometry=td.Box(size=(1, 1, td.inf)), offsets=[[0, 0, 0]])


def test_geometry_array_geometry_operations():
    """Test plot, intersections, volume, and surface_area."""
    box = td.Box(size=(1, 1, 1))
    offsets = [[0, 0, 0], [3, 0, 0]]
    array = td.GeometryArray(geometry=box, offsets=offsets)

    # Plot
    fig, ax = plt.subplots()
    array.plot(z=0, ax=ax)
    plt.close(fig)

    # Intersections
    assert len(array.intersections_plane(z=0)) == 2
    assert len(array.intersections_plane(z=10)) == 0

    # Volume and surface area
    assert np.isclose(array.volume(), 2.0)
    assert np.isclose(array.surface_area(), 12.0)


def test_geometry_array_equivalence_with_geometry_group():
    """Test that GeometryArray matches equivalent GeometryGroup."""
    box = td.Box(size=(1, 1, 1))
    offsets = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

    array = td.GeometryArray(geometry=box, offsets=offsets)
    group = td.GeometryGroup(
        geometries=[box.translated(0, 0, 0), box.translated(2, 0, 0), box.translated(0, 2, 0)]
    )

    # Bounds match
    np.testing.assert_allclose(array.bounds[0], group.bounds[0], atol=1e-10)
    np.testing.assert_allclose(array.bounds[1], group.bounds[1], atol=1e-10)

    # Inside matches
    for point in [(0, 0, 0), (2, 0, 0), (0, 2, 0), (1, 1, 0), (10, 10, 10)]:
        assert array.inside(*point) == group.inside(*point)


def test_filter_geometry_array_preserves_array_type():
    """Filtering a GeometryArray should preserve the array when multiple instances survive."""
    box = td.Box(size=(1, 1, 1))
    rotation = td.Transformed.rotation(np.pi / 2, 2)
    array = td.GeometryArray(
        geometry=box,
        offsets=[[0, 0, 0], [2, 0, 0], [5, 0, 0]],
        transforms=[np.eye(4), rotation, np.eye(4)],
    )
    region = td.Box(center=(1, 0, 0), size=(4, 3, 3))

    filtered = filter_intersecting_geometries([array], region)[0]

    assert isinstance(filtered, td.GeometryArray)
    assert filtered.geometry == box
    assert filtered.offsets == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert filtered.transforms is not None
    np.testing.assert_allclose(filtered.transforms[0], np.eye(4))
    np.testing.assert_allclose(filtered.transforms[1], rotation)


def test_filter_geometry_array_prunes_composite_base_and_rebuilds_array():
    """Filtering should recurse into composite bases before rebuilding GeometryArray."""
    keep_box = td.Box(center=(-1, 0, 0), size=(1, 1, 1))
    drop_box = td.Box(center=(5, 0, 0), size=(1, 1, 1))
    array = td.GeometryArray(
        geometry=td.GeometryGroup(geometries=(keep_box, drop_box)),
        offsets=[[0, 0, 0], [0, 3, 0]],
    )
    region = td.Box(center=(-1, 1.5, 0), size=(2, 5, 2))

    filtered = filter_intersecting_geometries([array], region)[0]

    assert isinstance(filtered, td.GeometryArray)
    assert filtered.geometry == keep_box
    assert filtered.offsets == ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0))
    assert filtered.transforms is None


def test_filter_geometry_array_matches_equivalent_geometry_group():
    """Filtering a GeometryArray should match filtering its equivalent GeometryGroup."""
    keep_box = td.Box(center=(-1, 0, 0), size=(1, 1, 1))
    drop_box = td.Box(center=(5, 0, 0), size=(1, 1, 1))
    array = td.GeometryArray(
        geometry=td.GeometryGroup(geometries=(keep_box, drop_box)),
        offsets=[[0, 0, 0], [0, 3, 0], [0, 6, 0]],
    )
    region = td.Box(center=(-1, 1.5, 0), size=(2, 5, 2))

    filtered_array = filter_intersecting_geometries([array], region)[0]
    filtered_group = filter_intersecting_geometries([array._geometry_group], region)[0]

    assert isinstance(filtered_array, td.GeometryArray)
    assert filtered_array._geometry_group == filtered_group


def test_filter_nested_geometry_array_matches_equivalent_geometry_group():
    """Filtering should agree for nested GeometryGroup and GeometryArray compositions."""
    nested_group = td.GeometryGroup(
        geometries=(
            td.Box(center=(-1, 0, 0), size=(1, 1, 1)),
            td.Box(center=(4, 0, 0), size=(1, 1, 1)),
        )
    )
    nested_array = td.GeometryArray(
        geometry=td.GeometryGroup(
            geometries=(
                td.Box(center=(-1, 2, 0), size=(1, 1, 1)),
                td.Box(center=(4, 2, 0), size=(1, 1, 1)),
            )
        ),
        offsets=[[0, 0, 0], [0, 3, 0]],
    )
    array = td.GeometryArray(
        geometry=td.GeometryGroup(geometries=(nested_group, nested_array)),
        offsets=[[0, 0, 0], [0, 0, 4]],
    )
    region = td.Box(center=(-1, 1.5, 2), size=(2, 4, 6))

    filtered_array = filter_intersecting_geometries([array], region)[0]
    filtered_group = filter_intersecting_geometries([array._geometry_group], region)[0]

    assert isinstance(filtered_array, td.GeometryArray)
    assert isinstance(filtered_array.geometry, td.GeometryGroup)
    assert filtered_array.geometry == td.GeometryGroup(
        geometries=(
            td.Box(center=(-1, 0, 0), size=(1, 1, 1)),
            td.Box(center=(-1, 2, 0), size=(1, 1, 1)),
        )
    )
    assert filtered_array._geometry_group == filtered_group


def test_filter_geometry_array_single_survivor_collapses_to_transformed_geometry():
    """A single surviving GeometryArray instance should collapse to one geometry."""
    box = td.Box(size=(1, 1, 1))
    array = td.GeometryArray(geometry=box, offsets=[[2, 0, 0], [6, 0, 0]])
    region = td.Box(center=(2, 0, 0), size=(2, 2, 2))

    filtered = filter_intersecting_geometries([array], region)[0]

    assert filtered == td.Transformed(geometry=box, transform=td.Transformed.translation(2, 0, 0))


def test_filter_geometry_array_groups_distinct_filtered_bases():
    """Filtering should bucket surviving instances by their filtered local base geometry."""
    left_box = td.Box(center=(-1, 0, 0), size=(0.5, 1, 1))
    right_box = td.Box(center=(1, 0, 0), size=(0.5, 1, 1))
    rotation = td.Transformed.rotation(np.pi, 2)
    array = td.GeometryArray(
        geometry=td.GeometryGroup(geometries=(left_box, right_box)),
        transforms=[np.eye(4), rotation],
    )
    region = td.Box(center=(-1, 0, 0), size=(0.75, 2, 2))

    filtered = filter_intersecting_geometries([array], region)[0]

    assert isinstance(filtered, td.GeometryGroup)
    assert filtered.geometries[0] == left_box
    assert isinstance(filtered.geometries[1], td.Transformed)
    assert filtered.geometries[1].geometry == right_box
    np.testing.assert_allclose(filtered.geometries[1].transform, rotation)


def test_filter_transformed_geometry_array_preserves_nested_array():
    """Filtering should preserve an outer Transformed wrapper around a GeometryArray."""
    array = td.GeometryArray(
        geometry=td.Box(size=(1, 1, 1)),
        offsets=[[0, 0, 0], [2, 0, 0], [6, 0, 0]],
    )
    translation = td.Transformed.translation(1, 0, 0)
    transformed = td.Transformed(geometry=array, transform=translation)
    region = td.Box(center=(2, 0, 0), size=(4, 2, 2))

    filtered = filter_intersecting_geometries([transformed], region)[0]

    assert isinstance(filtered, td.Transformed)
    np.testing.assert_allclose(filtered.transform, translation)
    assert isinstance(filtered.geometry, td.GeometryArray)
    assert filtered.geometry.offsets == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_filter_union_clip_operation_prunes_non_intersecting_children():
    """Union clip operations should prune children recursively when localizing geometry."""
    left_box = td.Box(center=(-2, 0, 0), size=(1, 1, 1))
    right_box = td.Box(center=(2, 0, 0), size=(1, 1, 1))
    union = td.ClipOperation(operation="union", geometry_a=left_box, geometry_b=right_box)

    filtered_left = filter_intersecting_geometries(
        [union], td.Box(center=(-2, 0, 0), size=(1.5, 2, 2))
    )[0]
    assert filtered_left == left_box

    filtered_right = filter_intersecting_geometries(
        [union], td.Box(center=(2, 0, 0), size=(1.5, 2, 2))
    )[0]
    assert filtered_right == right_box

    filtered_both = filter_intersecting_geometries(
        [union], td.Box(center=(0, 0, 0), size=(5, 2, 2))
    )[0]
    assert isinstance(filtered_both, td.ClipOperation)
    assert filtered_both.operation == "union"
    assert filtered_both.geometry_a == left_box
    assert filtered_both.geometry_b == right_box


def test_geometry_array_with_different_geometries():
    """Test GeometryArray with Sphere, Cylinder, and PolySlab."""
    offsets = [[0, 0, 0], [2, 0, 0]]

    for geom in [
        td.Sphere(radius=0.5),
        td.Cylinder(radius=0.5, length=1, axis=2),
    ]:
        array = td.GeometryArray(geometry=geom, offsets=offsets)
        assert array.inside(0, 0, 0)
        assert array.inside(2, 0, 0)
        assert not array.inside(1, 0, 0)

    # PolySlab
    polyslab = td.PolySlab(vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(-0.5, 0.5))
    array = td.GeometryArray(geometry=polyslab, offsets=offsets)
    assert array.inside(0.5, 0.5, 0)
    assert array.inside(2.5, 0.5, 0)


def test_geometry_array_adjoint_not_supported():
    """Test that GeometryArray raises NotImplementedError for adjoint/autodiff."""
    box = td.Box(size=(1, 1, 1))
    array = td.GeometryArray(geometry=box, offsets=[[0, 0, 0], [2, 0, 0]])

    with pytest.raises(NotImplementedError):
        array._compute_derivatives(derivative_info=None)


def test_geometry_array_normal_2dmaterial():
    """Test _normal_2dmaterial for GeometryArray with 2D geometries."""
    # Zero-thickness box normal to z
    box_2d = td.Box(center=(0, 0, 0), size=(1, 1, 0))
    offsets = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    array = td.GeometryArray(geometry=box_2d, offsets=offsets)
    assert array._normal_2dmaterial == 2

    # Zero-thickness box normal to y
    box_2d_y = td.Box(center=(0, 0, 0), size=(1, 0, 1))
    array_y = td.GeometryArray(geometry=box_2d_y, offsets=offsets)
    assert array_y._normal_2dmaterial == 1

    # 3D geometry should raise
    box_3d = td.Box(size=(1, 1, 1))
    array_3d = td.GeometryArray(geometry=box_3d, offsets=offsets)
    with pytest.raises(ValidationError):
        _ = array_3d._normal_2dmaterial


def test_geometry_array_update_from_bounds():
    """Test _update_from_bounds for GeometryArray with 2D geometries."""
    box_2d = td.Box(center=(0, 0, 0), size=(1, 1, 0))
    offsets = np.array([[0, 0, 0], [2, 0, 0]])
    array = td.GeometryArray(geometry=box_2d, offsets=offsets)

    new_bounds = (3.2, 6.4)
    axis = 2
    updated = array._update_from_bounds(bounds=new_bounds, axis=axis)

    # Result should be a GeometryGroup
    assert isinstance(updated, td.GeometryGroup)

    # All sub-geometries should have the new bounds along the axis
    for geom in updated.geometries:
        geom_bounds = (geom.bounds[0][axis], geom.bounds[1][axis])
        assert np.isclose(geom_bounds, new_bounds).all()


def test_geometry_array_medium2d_structure():
    """Test that GeometryArray works with Medium2D in a Structure."""
    box_2d = td.Box(center=(0, 0, 0.5), size=(1, 1, 0))
    offsets = np.array([[0, 0, 0], [2, 0, 0]])
    array = td.GeometryArray(geometry=box_2d, offsets=offsets)

    med2d = td.Medium2D(ss=td.Medium(), tt=td.Medium())
    struct = td.Structure(geometry=array, medium=med2d)
    assert struct is not None
