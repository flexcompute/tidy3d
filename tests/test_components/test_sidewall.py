"""test slanted polyslab can be correctly setup and visualized. """
from typing import Dict
import pytest
import numpy as np
import pydantic
from shapely.geometry import Polygon, Point

import tidy3d as td
from tidy3d.constants import fp_eps
from tidy3d.exceptions import ValidationError, SetupError

np.random.seed(4)
_BUFFER_PARAM = {"join_style": 2, "mitre_limit": 1e10}


def offset_distance_to_base(reference_plane, length_axis: float, tan_angle: float) -> float:
    """
    A convenient function that returns the distance needed to offset the cross section
    from reference plane to the base.

    Parameters
    ----------
    reference_plane : PlanePosition
        The position of the plane where the vertices of the polygon are supplied.
    length_axis : float
        The overall length of PolySlab along extrusion direction.
    tan_angle : float
        tan(sidewall angle)

    Returns
    -------
    float
        Offset distance.
    """

    if reference_plane == "top":
        return length_axis * tan_angle

    if reference_plane == "middle":
        return length_axis * tan_angle / 2

    # bottom
    return 0


def setup_polyslab(vertices, dilation, angle, bounds, axis=2, reference_plane="bottom"):
    """Setup slanted polyslab"""
    s = td.PolySlab(
        vertices=vertices,
        slab_bounds=bounds,
        axis=axis,
        dilation=dilation,
        sidewall_angle=angle,
        reference_plane=reference_plane,
    )
    return s


def convert_polyslab_other_reference_plane(poly, reference_plane):
    """Convert a polyslab defined at ``bottom`` to other plane"""
    offset_distance = -offset_distance_to_base(reference_plane, poly.length_axis, poly._tanq)
    vertices = poly._shift_vertices(poly.base_polygon, offset_distance)[0]
    return poly.copy(update={"vertices": vertices, "reference_plane": reference_plane})


def minimal_edge_length(vertices):
    """compute the minimal edge length in a polygon"""
    vs = vertices.T.copy()
    vsp = np.roll(vs.copy(), axis=-1, shift=-1)
    edge_length = np.linalg.norm(vsp - vs, axis=0)
    return np.min(edge_length)


def convert_valid_polygon(vertices):
    """Given vertices that might have intersecting edges, converted to
    vertices of a valid polygon
    """
    poly = Polygon(vertices).buffer(0, **_BUFFER_PARAM)  # make sure no intersecting edges
    if type(poly) is not Polygon:
        poly = poly.geoms[0]

    vertices_n = np.array(poly.exterior.coords[:])
    return vertices_n


def validate_poly_bound(poly):
    """validate bound based polyslab's base and top polygon"""
    xmin1, ymin1 = np.amin(poly.base_polygon, axis=0)
    xmax1, ymax1 = np.amax(poly.base_polygon, axis=0)

    xmin2, ymin2 = np.amin(poly.top_polygon, axis=0)
    xmax2, ymax2 = np.amax(poly.top_polygon, axis=0)

    xmin, ymin = min(xmin1, xmin2), min(ymin1, ymin2)
    xmax, ymax = max(xmax1, xmax2), max(ymax1, ymax2)

    bound_tidy = poly.bounds
    assert bound_tidy[0][0] <= xmin + fp_eps
    assert bound_tidy[0][1] <= ymin + fp_eps
    assert bound_tidy[1][0] >= xmax - fp_eps
    assert bound_tidy[1][1] >= ymax - fp_eps


# default values
bounds = (0, 0.5)
dilation = 0.0
angle = 0


def test_remove_duplicate_poly():
    """
    Make sure redundant neighboring vertices are removed
    """
    vertices = np.random.random((10, 2))
    vertices[0, :] = vertices[9, :]
    vertices[1, :] = vertices[0, :]
    vertices[5, :] = vertices[6, :]

    vertices = td.PolySlab._remove_duplicate_vertices(vertices)
    assert vertices.shape[0] == 7


def test_valid_polygon():
    """No intersecting edges"""

    # area = 0
    vertices = ((0, 0), (1, 0), (2, 0))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # only two points
    vertices = ((0, 0), (1, 0), (1, 0))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # intersecting edges
    vertices = ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, -1))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)


def test_crossing_square_poly():
    """
    Vertices crossing detection for a simple square
    """

    # allows self-intersection right at the top/base
    vertices = ((0, 0), (1, 0), (1, -1), (0, -1))
    dilation = 0.0
    angle = np.pi / 4
    s = setup_polyslab(vertices, dilation, angle, bounds)

    # fully eroded
    dilation = -1.1
    angle = 0
    for ref_plane in ["bottom", "middle", "top"]:
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane=ref_plane)

    # angle too large, self-intersecting
    dilation = 0
    angle = np.pi / 3
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)
        s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="top")
    # middle plane
    angle = np.arctan(1.999)
    s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")

    # angle too large for middle reference plane
    angle = np.arctan(2.001)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")

    # combines both
    dilation = -0.1
    angle = np.pi / 4
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)


def test_crossing_concave_poly():
    """
    Vertices crossing during dilation for a concave polygon
    """
    bounds = (0, 0.5)

    # self-intersecting during dilation, making a hole/island
    vertices = ((-0.5, 1), (-0.5, -1), (1, -1), (0, -0.1), (0, 0.1), (1, 1))
    dilation = 0.5
    angle = 0
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # polygon splitting
    dilation = -0.3
    angle = 0
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # polygon fully eroded
    dilation = -0.5
    angle = 0
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # # or, effectively
    dilation = 0
    angle = -np.pi / 4
    for bounds in [(0, 0.3), (0, 0.5)]:
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = setup_polyslab(vertices, dilation, angle, bounds)
            s = setup_polyslab(vertices, dilation, -angle, bounds, reference_plane="top")

    # middle plane
    angle = np.pi / 4
    bounds = (0, 0.44)
    s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")
    s = setup_polyslab(vertices, dilation, -angle, bounds, reference_plane="middle")
    with pytest.raises(pydantic.ValidationError) as e_info:
        # vertices degenerate
        bounds = (0, 0.45)
        s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")
        s = setup_polyslab(vertices, dilation, -angle, bounds, reference_plane="middle")
        # polygon splitting
        bounds = (0, 0.6)
        s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")
        s = setup_polyslab(vertices, dilation, -angle, bounds, reference_plane="middle")
        # fully eroded
        bounds = (0, 1)
        s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane="middle")
        s = setup_polyslab(vertices, dilation, -angle, bounds, reference_plane="middle")


def test_edge_events():
    """Make sure edge events error properly."""

    # No edge events here; however, if using shapely's buffer directly in detecting
    # edge events, it can error in mistake.
    Nx = 1000
    coord = np.linspace(-1, 1, Nx)
    vertice1 = [(f, -1) for f in coord]
    vertice2 = [(1, f) for f in coord[1:]]
    vertice3 = [(f, 1) for f in np.flip(coord)[1:]]
    vertice4 = [(-1, f) for f in np.flip(coord)[1:-1]]
    vertices = vertice1 + vertice2 + vertice3 + vertice4

    angle = -np.pi / 20
    s = td.PolySlab(
        vertices=vertices,
        axis=0,
        slab_bounds=(-1, 1),
        sidewall_angle=angle,
        reference_plane="bottom",
    )


def test_max_erosion_polygon():
    """
    Maximal erosion distance validation
    """
    N = 10  # number of vertices
    for i in range(50):
        vertices = convert_valid_polygon(np.random.random((N, 2)) * 10)

        dilation = 0
        angle = 0
        bounds = (0, 0.5)
        s = setup_polyslab(vertices, dilation, angle, bounds)

        # compute maximal allowed erosion distance
        max_dist = s._neighbor_vertices_crossing_detection(s.reference_polygon, -100)
        # verify it is indeed maximal allowed
        dilation = -max_dist + 1e-10
        # avoid polygon splitting etc. case
        if s._edge_events_detection(s.reference_polygon, dilation, ignore_at_dist=False):
            continue
        s = setup_polyslab(vertices, dilation, angle, bounds)
        assert np.isclose(minimal_edge_length(s.reference_polygon), 0, atol=1e-4)

        # verify it is indeed maximal allowed
        dilation = 0.0
        bounds = (0, max_dist - 1e-10)
        angle = np.pi / 4
        s = setup_polyslab(vertices, dilation, angle, bounds)
        assert np.isclose(minimal_edge_length(s.top_polygon), 0, atol=1e-4)


def test_shift_height_poly():
    """Make sure a list of height where the plane will intersect with the vertices
    works properly
    """
    N = 10  # number of vertices
    Lx = 10.0
    for i in range(50):
        vertices = convert_valid_polygon(np.random.random((N, 2)) * Lx)
        dilation = 0
        angle = 0
        bounds = (0, 1)
        s = setup_polyslab(vertices, dilation, angle, bounds)
        # set up proper thickness
        max_dist = s._neighbor_vertices_crossing_detection(s.base_polygon, -100)
        dilation = 0.0
        bounds = (0, max_dist * 0.99)
        angle = np.pi / 4
        # avoid vertex-edge crossing case
        try:
            s = setup_polyslab(vertices, dilation, angle, bounds)
        except:
            continue

        for axis in (0, 1):
            position = np.random.random(1)[0] * Lx - Lx / 2
            height = s._find_intersecting_height(position, axis)
            for h in height:
                bounds = (0, h)
                s = setup_polyslab(vertices, dilation, angle, bounds)
                diff = s.top_polygon[:, axis] - position
                assert np.any(np.isclose(diff, 0)) == True


def test_intersection_with_inside_poly():
    """Make sure intersection produces the same result as inside"""

    N = 10  # number of vertices
    Lx = 10  # maximal length in x,y direction
    angle_list = [-np.pi / 4, np.pi / 4]
    dilation = 0.0

    # generate vertices for testing
    Ntest = 20
    vertices_list = []
    # triangle
    vertices_list.append([[-1, -1], [0, -1], [1, -1], [0, 1]])
    # multiple vertices touching axis
    vertices_list.append([[0, -1], [0, 0], [0, 1], [0, 2], [-1, 2], [-1, -1]])
    # random vertices
    for i in range(Ntest):
        vertices_list.append(np.array(convert_valid_polygon(np.random.random((N, 2)) * Lx)))

    # different polyslab axis
    for axis in range(3):
        # sidewall angles
        for angle in angle_list:
            for vertices in vertices_list:
                max_dist = 5
                # for erosion type, setup appropriate bounds
                if angle > 0:
                    angle_tmp = 0
                    bounds = (0, 1)
                    s_bottom = setup_polyslab(vertices, dilation, angle_tmp, bounds, axis=axis)
                    # set up proper thickness
                    max_dist = s_bottom._neighbor_vertices_crossing_detection(s.base_polygon, -100)

                bounds = (-(max_dist * 0.95) / 2, (max_dist * 0.95) / 2)

                # avoid vertex-edge crossing case
                try:
                    s_bottom = setup_polyslab(vertices, dilation, angle, bounds, axis=axis)
                except:
                    continue
                s_top = convert_polyslab_other_reference_plane(s_bottom, "top")
                s_middle = convert_polyslab_other_reference_plane(s_bottom, "middle")
                s_list = [s_bottom, s_top, s_middle]
                xyz = np.random.random((10, 3)) * 2 * Lx - Lx

                # keep track for checking the consistency between different
                # reference plane
                res_inside_array = []
                for ind, s in enumerate(s_list):
                    ### side x
                    xp = 0
                    yp = xyz[:, 1]
                    zp = xyz[:, 2]
                    shape_intersect = s.intersections_plane(x=xp)

                    xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                    res_inside_array.append(s.inside(xarray, yarray, zarray))

                    for i in range(len(yp)):
                        for j in range(len(zp)):
                            # inside
                            res_inside = s.inside(xp, yp[i], zp[j])
                            assert res_inside_array[3 * ind][0, i, j] == res_inside
                            # intersect
                            res_inter = False
                            for shape in shape_intersect:
                                if shape.covers(Point(yp[i], zp[j])):
                                    res_inter = True
                            # if res_inter != res_inside:
                            #     print(repr(vertices))
                            #     print(repr(s.base_polygon))
                            #     print(bounds)
                            #     print(xp, yp[i], zp[j])
                            assert res_inter == res_inside

                    ### side y
                    xp = xyz[:, 0]
                    yp = 0
                    zp = xyz[:, 2]
                    shape_intersect = s.intersections_plane(y=yp)

                    xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                    res_inside_array.append(s.inside(xarray, yarray, zarray))

                    for i in range(len(xp)):
                        for j in range(len(zp)):
                            # inside
                            res_inside = s.inside(xp[i], yp, zp[j])
                            assert res_inside == res_inside_array[3 * ind + 1][i, 0, j]
                            # intersect
                            res_inter = False
                            for shape in shape_intersect:
                                if shape.covers(Point(xp[i], zp[j])):
                                    res_inter = True
                            assert res_inter == res_inside

                    ### norm z
                    xp = xyz[:, 0]
                    yp = xyz[:, 1]
                    zp = 0
                    shape_intersect = s.intersections_plane(z=zp)

                    xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                    res_inside_array.append(s.inside(xarray, yarray, zarray))

                    for i in range(len(xp)):
                        for j in range(len(yp)):
                            # inside
                            res_inside = s.inside(xp[i], yp[j], zp)
                            assert res_inside == res_inside_array[3 * ind + 2][i, j, 0]
                            # intersect
                            res_inter = False
                            for shape in shape_intersect:
                                if shape.covers(Point(xp[i], yp[j])):
                                    res_inter = True
                            assert res_inter == res_inside
                # consistency between different reference plane
                for ind_pol in range(3):
                    for ind_poly in range(2):
                        assert np.allclose(
                            res_inside_array[3 * ind_poly + ind_pol],
                            res_inside_array[3 * (1 + ind_poly) + ind_pol],
                        )


def test_bound_poly():
    """
    Make sure bound works, even though it might not be tight.
    """
    N = 10  # number of vertices
    Lx = 10  # maximal length in x,y direction
    for i in range(50):
        for reference_plane in ["bottom", "middle", "top"]:
            vertices = convert_valid_polygon(np.random.random((N, 2)) * Lx)
            vertices = np.array(vertices)  # .astype("float32")

            ### positive dilation
            dilation = 0
            angle = 0
            bounds = (0, 1)
            s = setup_polyslab(vertices, dilation, angle, bounds, reference_plane=reference_plane)
            max_dist = s._neighbor_vertices_crossing_detection(s.base_polygon, 100)
            # verify it is indeed maximal allowed
            dilation = 1
            if max_dist is not None:
                dilation = max_dist - 1e-10
            bounds = (0, 1)
            angle = 0.0
            # avoid vertex-edge crossing case
            try:
                s = setup_polyslab(
                    vertices, dilation, angle, bounds, reference_plane=reference_plane
                )
            except:
                continue
            validate_poly_bound(s)

            ## sidewall
            dilation = 0
            angle = 0
            bounds = (0, 1)
            s = setup_polyslab(vertices, dilation, angle, bounds)
            # set up proper thickness
            max_dist = s._neighbor_vertices_crossing_detection(s.base_polygon, -100)
            dilation = 0.0
            bounds = (0, (max_dist * 0.95))
            angle = np.pi / 4
            # avoid vertex-edge crossing case
            try:
                s = setup_polyslab(vertices, dilation, angle, bounds)
            except:
                continue
            s = convert_polyslab_other_reference_plane(s, reference_plane)
            validate_poly_bound(s)


def test_normal_intersection_with_inside_cylinder():
    """Make sure intersection_normal produces the same result as inside"""

    radius = 1.1
    length = 0.9

    num_test_point = 10
    num_test_z = 10

    # a list of cylinders
    cyl = td.Cylinder(
        center=(0.1, 0.2, 0.3),
        radius=radius,
        length=length,
        axis=2,
        sidewall_angle=0,
        reference_plane="bottom",
    )
    s_list = [cyl]
    s_list.append(cyl.copy(update={"sidewall_angle": np.pi / 2.1}))
    s_list.append(cyl.copy(update={"sidewall_angle": np.pi / 4}))
    s_list.append(cyl.copy(update={"reference_plane": "top", "sidewall_angle": np.pi / 4.1}))
    s_list.append(cyl.copy(update={"reference_plane": "middle", "sidewall_angle": np.pi / 4}))

    x_list = np.linspace(-3 * radius, 3 * radius, num_test_point)
    y_list = np.linspace(-3 * radius, 3 * radius, num_test_point)
    z_list = np.linspace(-length, length, num_test_z)
    for s in s_list:
        for z in z_list:
            shape_intersect = s.intersections_plane(z=z)
            for x in x_list:
                for y in y_list:
                    # inside
                    res_inside = s.inside(x, y, z)
                    # intersect
                    res_inter = False
                    for shape in shape_intersect:
                        if shape.covers(Point(x, y)):
                            res_inter = True
                    if res_inter != res_inside:
                        print(x, y, z)
                        print(Point(0.1, 0.2).buffer(1.1).covers(Point(x, y)))
                    assert res_inter == res_inside


def test_side_intersection_cylinder():
    """Make sure intersection_side produces correct result"""

    radius = 1.0
    length = 1.0

    # a list of cylinders
    cyl = td.Cylinder(
        center=(0, 0, 0),
        radius=radius,
        length=length,
        axis=2,
        sidewall_angle=np.pi / 4,
        reference_plane="middle",
    )

    shape_intersect = cyl.intersections_plane(x=0)
    assert shape_intersect[0].covers(Point(1.25, 0.4)) == False
    assert shape_intersect[0].covers(Point(1.25, -0.4)) == True

    shape_intersect = cyl.intersections_plane(x=np.sqrt(3) / 2)
    assert shape_intersect[0].covers(Point(1.25, -0.4)) == False
    assert shape_intersect[0].covers(Point(0.7, -0.4)) == True


def test_slanted_infinite_cylinder():
    """Make sure infinite slanted cylinder works."""

    radius = 1.0
    length = td.inf

    # a list of cylinders
    cyl = td.Cylinder(
        center=(0, 0, 0),
        radius=radius,
        length=length,
        axis=2,
        sidewall_angle=np.pi / 4,
        reference_plane="middle",
    )

    shape_intersect = cyl.intersections_plane(x=0)
    assert shape_intersect[0].covers(Point(1.25, 0.4)) == False
    assert shape_intersect[0].covers(Point(1.25, -0.4)) == True

    shape_intersect = cyl.intersections_plane(x=np.sqrt(3) / 2)
    assert shape_intersect[0].covers(Point(1.25, -0.4)) == False
