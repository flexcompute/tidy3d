"""Tests Geometry objects."""

import pytest
import pydantic
import numpy as np
import shapely
import matplotlib.pylab as plt
import gdstk
import gdspy
import trimesh

import tidy3d as td
from tidy3d.exceptions import ValidationError, SetupError, Tidy3dKeyError
from tidy3d.components.geometry import Geometry, Planar
from ..utils import assert_log_level, prepend_tmp, log_capture


GEO = td.Box(size=(1, 1, 1))
GEO_INF = td.Box(size=(1, 1, td.inf))
BOX = td.Box(size=(1, 1, 1))
BOX_2D = td.Box(size=(1, 0, 1))
POLYSLAB = td.PolySlab(vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(-0.5, 0.5), axis=2)
SPHERE = td.Sphere(radius=1)
CYLINDER = td.Cylinder(axis=2, length=1, radius=1)

GEO_TYPES = [BOX, CYLINDER, SPHERE, POLYSLAB]

_, AX = plt.subplots()


@pytest.mark.parametrize("component", GEO_TYPES)
def test_plot(component):
    _ = component.plot(z=0, ax=AX)


def test_base_inside():
    assert Geometry.inside(GEO, x=0, y=0, z=0)
    assert np.all(Geometry.inside(GEO, np.array([0, 0]), np.array([0, 0]), np.array([0, 0])))
    assert np.all(Geometry.inside(GEO, np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])))


def test_base_inside_meshgrid():
    assert np.all(Geometry.inside_meshgrid(GEO, x=[0], y=[0], z=[0]))
    assert np.all(Geometry.inside_meshgrid(GEO, [0, 0], [0, 0], [0, 0]))
    # Input dimensions different than 1 error for ``inside_meshgrid``.
    with pytest.raises(ValueError):
        b = Geometry.inside_meshgrid(GEO, x=0, y=0, z=0)
    with pytest.raises(ValueError):
        b = Geometry.inside_meshgrid(GEO, [[0, 0]], [[0, 0]], [[0, 0]])


def test_bounding_box():
    assert GEO.bounding_box == GEO
    assert GEO_INF.bounding_box == GEO_INF


def test_strip_coords_multi():
    lat_point_list = [0, 0, 0, 5, 9, 11, 7, 3, 9, 0, 0, 0]
    lon_point_list = [438, 428, 427, 428, 434, 439, 443, 446, 448, 452, 452, 449]

    polygon_geom = shapely.geometry.Polygon(zip(lon_point_list, lat_point_list))
    multipolygon_geom = shapely.geometry.MultiPolygon([polygon_geom])
    ext_coords, list_int_coords = Geometry.strip_coords(multipolygon_geom)
    assert len(list_int_coords) == 0
    assert np.allclose(
        np.array(ext_coords)[:-1], np.array(list(zip(lon_point_list, lat_point_list)))
    )


def test_map_to_coords_not_polygon():
    assert Geometry.map_to_coords(lambda x: None, "test") == "test"


@pytest.mark.parametrize("points_shape", [(3,), (3, 10)])
def test_rotate_points(points_shape):
    points = np.random.random(points_shape)
    points_rotated = Geometry.rotate_points(points=points, axis=(0, 0, 1), angle=2 * np.pi)
    assert np.allclose(points, points_rotated)
    points_rotated = Geometry.rotate_points(points=points, axis=(0, 0, 1), angle=np.pi)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflect_points(axis):
    points = np.random.random((3, 10))
    pr = GEO.reflect_points(points=points, polar_axis=2, angle_theta=2 * np.pi, angle_phi=0)
    assert np.allclose(pr, points)
    pr = GEO.reflect_points(points=points, polar_axis=2, angle_theta=0, angle_phi=2 * np.pi)
    assert np.allclose(pr, points)


@pytest.mark.parametrize("component", GEO_TYPES)
def test_volume(component):
    v = component.volume()
    v = component.volume(bounds=GEO.bounds)
    v = component.volume(bounds=((-100, -100, -100), (100, 100, 100)))
    v = component.volume(bounds=((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    v = component.volume(bounds=((-100, -100, -100), (-10, -10, -10)))
    v = component.volume(bounds=((10, 10, 10), (100, 100, 100)))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_surface_area(component):
    sa = component.surface_area()
    sa = component.surface_area(bounds=GEO.bounds)
    sa = component.surface_area(bounds=((-100, -100, -100), (100, 100, 100)))
    sa = component.surface_area(bounds=((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    sa = component.surface_area(bounds=((-100, -100, -100), (-10, -10, -10)))
    sa = component.surface_area(bounds=((10, 10, 10), (100, 100, 100)))


@pytest.mark.parametrize("component", GEO_TYPES)
def test_bounds(component):
    b = component.bounds


def test_planar_bounds():
    _ = Planar.bounds.fget(CYLINDER)


@pytest.mark.parametrize("component", GEO_TYPES)
def test_inside(component):
    b = component.inside(0, 0, 0)
    bs = component.inside(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
    bss = component.inside(np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]))


def test_zero_dims():
    assert BOX.zero_dims == []
    assert BOX_2D.zero_dims == [1]


def test_inside_polyslab_sidewall():
    ps = POLYSLAB.copy(update=dict(sidewall_angle=0.1))
    ps.inside(x=0, y=0, z=0)


# TODO: Weiliang fix this test? doesnt work when sidewall non-zero
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


def test_bounds_base():
    assert all(a == b for a, b in zip(Planar.bounds.fget(POLYSLAB), POLYSLAB.bounds))


def test_center_not_inf_validate():
    with pytest.raises(pydantic.ValidationError):
        g = td.Box(center=(td.inf, 0, 0))
    with pytest.raises(pydantic.ValidationError):
        g = td.Box(center=(-td.inf, 0, 0))


def test_radius_not_inf_validate():
    with pytest.raises(pydantic.ValidationError):
        g = td.Sphere(radius=td.inf)
    with pytest.raises(pydantic.ValidationError):
        g = td.Cylinder(radius=td.inf, center=(0, 0, 0), axis=1, length=1)


def test_slanted_cylinder_infinite_length_validate():
    g = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf)
    g = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="top")
    g = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="bottom")
    g = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=td.inf, reference_plane="middle")
    g = td.Cylinder(
        radius=1,
        center=(0, 0, 0),
        axis=1,
        length=td.inf,
        sidewall_angle=0.1,
        reference_plane="middle",
    )
    with pytest.raises(pydantic.ValidationError):
        g = td.Cylinder(
            radius=1,
            center=(0, 0, 0),
            axis=1,
            length=td.inf,
            sidewall_angle=0.1,
            reference_plane="top",
        )
    with pytest.raises(pydantic.ValidationError):
        g = td.Cylinder(
            radius=1,
            center=(0, 0, 0),
            axis=1,
            length=td.inf,
            sidewall_angle=0.1,
            reference_plane="bottom",
        )


def test_box_from_bounds():
    b = td.Box.from_bounds(rmin=(-td.inf, 0, 0), rmax=(td.inf, 0, 0))
    assert b.center[0] == 0.0

    with pytest.raises(SetupError):
        b = td.Box.from_bounds(rmin=(0, 0, 0), rmax=(td.inf, 0, 0))

    b = td.Box.from_bounds(rmin=(-1, -1, -1), rmax=(1, 1, 1))
    assert b.center == (0, 0, 0)


def test_polyslab_center_axis():
    ps = POLYSLAB.copy(update=dict(slab_bounds=(-td.inf, td.inf)))
    assert ps.center_axis == 0


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


def make_geo_group():
    """Make a generic Geometry Group."""
    boxes = [td.Box(size=(1, 1, 1), center=(i, 0, 0)) for i in range(-5, 5)]
    return td.GeometryGroup(geometries=boxes)


def test_surfaces():
    with pytest.raises(SetupError):
        td.Box.surfaces(size=(1, 0, 1), center=(0, 0, 0))

    td.FluxMonitor.surfaces(
        size=(1, 1, 1), center=(0, 0, 0), normal_dir="+", name="test", freqs=[1]
    )
    td.Box.surfaces(size=(1, 1, 1), center=(0, 0, 0), normal_dir="+")


def test_arrow_both_dirs():
    _, ax = plt.subplots()
    GEO._plot_arrow(direction=(1, 2, 3), x=0, both_dirs=True, ax=ax)


def test_gds_cell():
    gds_cell = gdstk.Cell("name")
    gds_cell.add(gdstk.rectangle((0, 0), (1, 1)))
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0)
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0, gds_dtype=0)
    with pytest.raises(pydantic.ValidationError):
        td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1)
    with pytest.raises(pydantic.ValidationError):
        td.PolySlab.from_gds(
            gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1, gds_dtype=0
        )


def test_geo_group_initialize():
    """make sure you can construct one."""
    geo_group = make_geo_group()


def test_geo_group_structure():
    """make sure you can construct a structure using GeometryGroup."""

    geo_group = make_geo_group()
    structure = td.Structure(geometry=geo_group, medium=td.Medium())


def test_geo_group_methods():
    """Tests the geometry methods of geo group."""

    geo_group = make_geo_group()
    geo_group.inside(0, 1, 2)
    geo_group.inside(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    geo_group.inside_meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    geo_group.intersections_plane(y=0)
    geo_group.intersects(td.Box(size=(1, 1, 1)))
    rmin, rmax = geo_group.bounds


def test_geo_group_empty():
    """dont allow empty geometry list."""

    with pytest.raises(pydantic.ValidationError):
        geo_group = td.GeometryGroup(geometries=[])


def test_geo_group_volume():
    geo_group = make_geo_group()
    geo_group._volume(bounds=GEO.bounds)


def test_geo_group_surface_area():
    geo_group = make_geo_group()
    geo_group._surface_area(bounds=GEO.bounds)


""" geometry """


def test_geometry():

    b = td.Box(size=(1, 1, 1), center=(0, 0, 0))
    s = td.Sphere(radius=1, center=(0, 0, 0))
    s = td.Cylinder(radius=1, center=(0, 0, 0), axis=1, length=1)
    s = td.PolySlab(vertices=((1, 2), (3, 4), (5, 4)), slab_bounds=(-1, 1), axis=2)
    # vertices_np = np.array(s.vertices)
    # s_np = PolySlab(vertices=vertices_np, slab_bounds=(-1, 1), axis=1)

    # make sure wrong axis arguments error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Cylinder(radius=1, center=(0, 0, 0), axis=-1, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.PolySlab(radius=1, center=(0, 0, 0), axis=-1, slab_bounds=(-0.5, 0.5))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.PolySlab(radius=1, center=(0, 0, 0), axis=3, slab_bounds=(-0.5, 0.5))

    # make sure negative values error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Sphere(radius=-1, center=(0, 0, 0))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Cylinder(radius=-1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Cylinder(radius=1, center=(0, 0, 0), axis=3, length=-1)


def test_geometry_sizes():

    # negative in size kwargs errors
    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError) as e_info:
            a = td.Box(size=size, center=(0, 0, 0))
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = td.Simulation(size=size, run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

    # negative grid sizes error?
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.uniform(dl=-1.0), run_time=1e-12)


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
    assert len(polyslabs_touching) == 1, "polyslabs didnt merge correctly."


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
    assert ps.inside(point[0], point[1], point[2]) == False

    # intersections
    plane_coord = [None] * 3
    plane_coord[axis] = 3
    assert ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2]) == False
    plane_coord[axis] = -3
    assert ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2]) == False


def test_polyslab_merge():
    """make sure polyslabs from gds get merged when they should."""

    import gdspy

    def make_polyslabs(gap_size):
        """Construct two rectangular polyslabs separated by a gap."""
        lib = gdspy.GdsLibrary()
        cell = lib.new_cell(f"polygons_{gap_size:.2f}")
        rect1 = gdspy.Rectangle((gap_size / 2, 0), (1, 1))
        rect2 = gdspy.Rectangle((-1, 0), (-gap_size / 2, 1))
        cell.add(rect1)
        cell.add(rect2)
        return td.PolySlab.from_gds(gds_cell=cell, gds_layer=0, axis=2, slab_bounds=(-1, 1))

    polyslabs_gap = make_polyslabs(gap_size=0.3)
    assert len(polyslabs_gap) == 2, "untouching polylsabs were merged incorrectly."

    polyslabs_touching = make_polyslabs(gap_size=0)
    assert len(polyslabs_touching) == 1, "polyslabs didnt merge correctly."


def test_gds_cell():
    gds_cell = gdspy.Cell("name")
    gds_cell.add(gdspy.Rectangle((0, 0), (1, 1)))
    td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=0)
    with pytest.raises(Tidy3dKeyError):
        td.PolySlab.from_gds(gds_cell=gds_cell, axis=2, slab_bounds=(-1, 1), gds_layer=1)


def test_custom_surface_geometry():
    # create tetrahedron STL
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    geom = td.TriangleMesh.from_trimesh(tetrahedron)

    # test import
    import_geom = td.TriangleMesh.from_stl("tests/data/tetrahedron.stl")
    assert np.allclose(import_geom.triangles, geom.triangles)

    # test export and then import
    geom.trimesh.export(prepend_tmp("export.stl"))
    import_geom = td.TriangleMesh.from_stl(prepend_tmp("export.stl"))
    assert np.allclose(import_geom.triangles, geom.triangles)

    # assert np.array_equal(tetrahedron.vectors, export_vectors)

    areas = [0.5 * np.sqrt(2) * np.sqrt(1 + 2 * 0.5**2), 0.5, 0.5, 0.5]
    unit_normals_unnormalized = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    unit_normals = [n / np.linalg.norm(n) for n in unit_normals_unnormalized]
    normals = [n * a for (n, a) in zip(unit_normals, areas)]

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

    # test inconsistent winding
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[2, 1, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    # we currently just log a warning
    # with pytest.raises(ValidationError):
    geom = td.TriangleMesh.from_trimesh(tetrahedron)

    # test non-watertight mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    # we currently just log a warning
    # with pytest.raises(ValidationError):
    geom = td.TriangleMesh.from_trimesh(tetrahedron)

    # test zero area triangles
    vertices = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    tetrahedron = trimesh.Trimesh(vertices, faces)
    with pytest.raises(pydantic.ValidationError):
        geom = td.TriangleMesh.from_trimesh(tetrahedron)

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
