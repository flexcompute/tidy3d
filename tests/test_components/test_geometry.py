"""Tests Geometry objects."""

import warnings

import gdspy
import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import shapely
import tidy3d as td
import trimesh
from tidy3d.components.geometry.base import Planar
from tidy3d.components.geometry.mesh import AREA_SIZE_THRESHOLD
from tidy3d.components.geometry.utils import flatten_groups, traverse_geometries
from tidy3d.constants import LARGE_NUMBER
from tidy3d.exceptions import SetupError, Tidy3dKeyError, ValidationError

from ..utils import AssertLogLevel

GEO = td.Box(size=(1, 1, 1))
GEO_INF = td.Box(size=(1, 1, td.inf))
BOX = td.Box(size=(1, 1, 1))
BOX_2D = td.Box(size=(1, 0, 1))
POLYSLAB = td.PolySlab(vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(-0.5, 0.5), axis=2)
SPHERE = td.Sphere(radius=1)
CYLINDER = td.Cylinder(axis=2, length=1, radius=1)

GROUP = td.GeometryGroup(
    geometries=[
        td.Box(center=(-0.25, 0, 0), size=(0.5, 1, 1)),
        td.Box(center=(0.25, 0, 0), size=(0.5, 1, 1)),
    ]
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


GEO_TYPES = [
    BOX,
    CYLINDER,
    SPHERE,
    POLYSLAB,
    UNION,
    INTERSECTION,
    DIFFERENCE,
    SYM_DIFFERENCE,
    GROUP,
    TRANSFORMED,
]

_, AX = plt.subplots()


@pytest.mark.parametrize("component", GEO_TYPES)
def test_plot(component):
    _ = component.plot(z=0, ax=AX)
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


def test_planar_bounds():
    _ = Planar.bounds.fget(CYLINDER)


@pytest.mark.parametrize("component", GEO_TYPES)
def test_inside(component):
    _ = component.inside(0, 0, 0)
    _ = component.inside(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
    _ = component.inside(np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]))


def test_zero_dims():
    assert BOX.zero_dims == []
    assert BOX_2D.zero_dims == [1]


def test_inside_polyslab_sidewall():
    ps = POLYSLAB.copy(update=dict(sidewall_angle=0.1))
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


def test_bounds_base():
    assert all(a == b for a, b in zip(Planar.bounds.fget(POLYSLAB), POLYSLAB.bounds))


def test_center_not_inf_validate():
    with pytest.raises(pydantic.ValidationError):
        _ = td.Box(center=(td.inf, 0, 0))
    with pytest.raises(pydantic.ValidationError):
        _ = td.Box(center=(-td.inf, 0, 0))


def test_radius_not_inf_validate():
    with pytest.raises(pydantic.ValidationError):
        _ = td.Sphere(radius=td.inf)
    with pytest.raises(pydantic.ValidationError):
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.Cylinder(
            radius=1,
            center=(0, 0, 0),
            axis=1,
            length=td.inf,
            sidewall_angle=0.1,
            reference_plane="top",
        )
    with pytest.raises(pydantic.ValidationError):
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


def test_polyslab_center_axis():
    """Test the handling of center_axis in a polyslab having (-td.inf, td.inf) bounds."""
    ps = POLYSLAB.copy(update=dict(slab_bounds=(-td.inf, td.inf)))
    assert ps.center_axis == 0


@pytest.mark.parametrize(
    "lower_bound, upper_bound", ((-td.inf, td.inf), (-1, td.inf), (-td.inf, 1))
)
def test_polyslab_inf_bounds(lower_bound, upper_bound):
    """Test the handling of various operations in a polyslab having inf bounds."""
    ps = POLYSLAB.copy(update=dict(slab_bounds=(lower_bound, upper_bound)))
    # catch any runtime warning related to inf operations
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = ps.bounds
        ps.intersections_plane(x=0.5)
        ps.intersections_plane(z=0)


def test_polyslab_bounds():
    with pytest.raises(pydantic.ValidationError):
        td.PolySlab(vertices=((0, 0), (1, 0), (1, 1)), slab_bounds=(0.5, -0.5), axis=2)


def test_validate_polyslab_vertices_valid():
    with pytest.raises(pydantic.ValidationError):
        POLYSLAB.copy(update=dict(vertices=(1, 2, 3)))
    with pytest.raises(pydantic.ValidationError):
        crossing_verts = ((0, 0), (1, 1), (0, 1), (1, 0))
        POLYSLAB.copy(update=dict(vertices=crossing_verts))


def test_sidewall_failed_validation():
    with pytest.raises(pydantic.ValidationError):
        POLYSLAB.copy(update=dict(sidewall_angle=1000))


def test_surfaces():
    with pytest.raises(SetupError):
        td.Box.surfaces(size=(1, 0, 1), center=(0, 0, 0))

    td.FluxMonitor.surfaces(
        size=(1, 1, 1), center=(0, 0, 0), normal_dir="+", name="test", freqs=[1e12]
    )
    td.Box.surfaces(size=(1, 1, 1), center=(0, 0, 0), normal_dir="+")


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


def test_gdspy_cell():
    gds_cell = gdspy.Cell("name")
    gds_cell.add(gdspy.Rectangle((0, 0), (1, 1)))
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0)
    with pytest.raises(Tidy3dKeyError):
        td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1)


def make_geo_group():
    """Make a generic Geometry Group."""
    boxes = [td.Box(size=(1, 1, 1), center=(i, 0, 0)) for i in range(-5, 5)]
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.GeometryGroup(geometries=[])


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
    assert BOX + GROUP == td.GeometryGroup(geometries=(BOX,) + GROUP.geometries)
    assert GROUP + CYLINDER == td.GeometryGroup(geometries=GROUP.geometries + (CYLINDER,))

    assert BOX | CYLINDER == td.GeometryGroup(geometries=(BOX, CYLINDER))
    assert BOX | UNION == td.GeometryGroup(geometries=(BOX, UNION.geometry_a, UNION.geometry_b))
    assert UNION | CYLINDER == td.GeometryGroup(
        geometries=(UNION.geometry_a, UNION.geometry_b, CYLINDER)
    )
    assert BOX | GROUP == td.GeometryGroup(geometries=(BOX,) + GROUP.geometries)
    assert GROUP | CYLINDER == td.GeometryGroup(geometries=GROUP.geometries + (CYLINDER,))

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
    geo = geo.translated(-2, 0, 0).rotated(-np.pi * 0.4, 1)
    assert (geo.inside(*xyz) == (True, False, True, False, True)).all()
    assert len(geo.intersections_plane(x=0)) == 1
    assert len(geo.intersections_plane(z=0)) == 3


def test_general_rotation():
    assert np.allclose(td.Transformed.rotation(0.1, 0), td.Transformed.rotation(0.1, [2, 0, 0]))
    assert np.allclose(td.Transformed.rotation(0.2, 1), td.Transformed.rotation(0.2, [0, 3, 0]))
    assert np.allclose(td.Transformed.rotation(0.3, 2), td.Transformed.rotation(0.3, [0, 0, 4]))


def test_flattening():
    flat = list(
        flatten_groups(
            td.GeometryGroup(
                geometries=[
                    td.Box(size=(1, 1, 1)),
                    td.Box(size=(0, 1, 0)),
                    td.ClipOperation(
                        operation="union",
                        geometry_a=td.Box(size=(0, 0, 1)),
                        geometry_b=td.GeometryGroup(
                            geometries=[
                                td.Box(size=(2, 2, 2)),
                                td.GeometryGroup(
                                    geometries=[td.Box(size=(3, 3, 3)), td.Box(size=(3, 0, 3))]
                                ),
                            ]
                        ),
                    ),
                ]
            )
        )
    )
    assert len(flat) == 6
    assert all(isinstance(g, td.Box) for g in flat)

    flat = list(
        flatten_groups(
            td.GeometryGroup(
                geometries=[
                    td.Box(size=(1, 1, 1)),
                    td.Box(size=(0, 1, 0)),
                    td.ClipOperation(
                        operation="intersection",
                        geometry_a=td.Box(size=(0, 0, 1)),
                        geometry_b=td.GeometryGroup(
                            geometries=[
                                td.Box(size=(2, 2, 2)),
                                td.GeometryGroup(
                                    geometries=[td.Box(size=(3, 3, 3)), td.Box(size=(3, 0, 3))]
                                ),
                            ]
                        ),
                    ),
                ]
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
        geometries=[
            td.Box(size=(1, 0, 0)),
            td.ClipOperation(
                operation="intersection",
                geometry_a=td.GeometryGroup(
                    geometries=[
                        td.Box(size=(5, 0, 0)),
                        td.Box(size=(6, 0, 0)),
                    ]
                ),
                geometry_b=td.ClipOperation(
                    operation="difference",
                    geometry_a=td.Box(size=(7, 0, 0)),
                    geometry_b=td.Box(size=(8, 0, 0)),
                ),
            ),
            td.GeometryGroup(
                geometries=[
                    td.Box(size=(3, 0, 0)),
                    td.Box(size=(4, 0, 0)),
                ]
            ),
            td.Box(size=(2, 0, 0)),
        ]
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=-1, length=1)
    with pytest.raises(pydantic.ValidationError):
        _ = td.PolySlab(radius=1, center=(0, 0, 0), axis=-1, slab_bounds=(-0.5, 0.5))
    with pytest.raises(pydantic.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError):
        _ = td.PolySlab(radius=1, center=(0, 0, 0), axis=3, slab_bounds=(-0.5, 0.5))

    # make sure negative values error
    with pytest.raises(pydantic.ValidationError):
        _ = td.Sphere(radius=-1, center=(0, 0, 0))
    with pytest.raises(pydantic.ValidationError):
        _ = td.Cylinder(radius=-1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError):
        _ = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=-1)


def test_geometry_sizes():
    # negative in size kwargs errors
    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError):
            _ = td.Box(size=size, center=(0, 0, 0))
        with pytest.raises(pydantic.ValidationError):
            _ = td.Simulation(size=size, run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

    # negative grid sizes error?
    with pytest.raises(pydantic.ValidationError):
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
    assert len(polyslabs_gap) == 2, "untouching polylsabs were merged incorrectly."

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
    poly = poly.updated_copy(slab_bounds=[-td.inf, 0])
    assert len(poly.intersections_plane(x=0)) == 1
    assert poly.intersections_plane(x=0)[0] == shapely.box(-1, -LARGE_NUMBER, 1, 0)


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


def test_custom_surface_geometry(tmp_path, log_capture):
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
    faces = np.array([[2, 1, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel(log_capture, "WARNING"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    with AssertLogLevel(log_capture, None):
        geom = geom.fix_winding()

    # test non-watertight mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel(log_capture, "WARNING"):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)
    with AssertLogLevel(log_capture, None):
        geom = geom.fill_holes()

    # test zero area triangles
    vertices = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with AssertLogLevel(log_capture, "WARNING"):
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


def test_geo_group_sim():
    geo_grp = td.TriangleMesh.from_stl("tests/data/two_boxes_separate.stl")
    geos_orig = list(geo_grp.geometries)
    geo_grp_full = geo_grp.updated_copy(geometries=geos_orig + [td.Box(size=(1, 1, 1))])

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
    with pytest.raises(pydantic.ValidationError):
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


def test_triangulation_with_collinear_vertices():
    xr = np.linspace(0, 1, 6)
    a = np.array([[x, -0.5] for x in xr] + [[x, 0.5] for x in xr[::-1]])
    assert len(td.components.geometry.triangulation.triangulate(a)) == 10
