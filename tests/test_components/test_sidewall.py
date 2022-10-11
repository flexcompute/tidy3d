"""test slanted polyslab can be correctly setup and visualized. """
from typing import Dict
import pytest
import numpy as np
import pydantic
from shapely.geometry import Polygon, Point

import tidy3d as td
from tidy3d.log import ValidationError, SetupError

np.random.seed(4)


def setup_polyslab(vertices, dilation, angle, bounds, axis=2):
    """Setup slanted polyslab"""
    s = td.PolySlab(
        vertices=vertices,
        slab_bounds=bounds,
        axis=axis,
        dilation=dilation,
        sidewall_angle=angle,
    )
    return s


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
    poly = Polygon(vertices).buffer(0)  # make sure no intersecting edges
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
    assert bound_tidy[0][0] <= xmin
    assert bound_tidy[0][1] <= ymin
    assert bound_tidy[1][0] >= xmax
    assert bound_tidy[1][1] >= ymax


# default values
bounds = (0, 0.5)
dilation = 0.0
angle = 0


def test_remove_duplicate():
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
    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # only two points
    vertices = ((0, 0), (1, 0), (1, 0))
    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # intersecting edges
    vertices = ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, -1))

    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)


def test_crossing_square():
    """
    Vertices crossing detection for a simple square
    """
    vertices = ((0, 0), (1, 0), (1, -1), (0, -1))
    dilation = 0.0
    angle = np.pi / 4
    s = setup_polyslab(vertices, dilation, angle, bounds)

    # dilation too significant
    dilation = -1.1
    angle = 0
    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # angle too large
    dilation = 0
    angle = np.pi / 3
    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)

    # combines both
    dilation = -0.1
    angle = np.pi / 4
    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)


def test_crossing_concave_poly():
    """
    Vertices crossing during dilation for a concave polygon
    """

    # self-intersecting, concave polygon
    vertices = ((-1, 1), (-1, -1), (1, -1), (0, -0.1), (0, 0.1), (1, 1))
    dilation = 0.5
    angle = 0

    with pytest.raises(SetupError) as e_info:
        s = setup_polyslab(vertices, dilation, angle, bounds)


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
        _, max_dist = s._crossing_detection(s.base_polygon, -100)

        # verify it is indeed maximal allowed
        dilation = -max_dist + 1e-10
        # avoid vertex-edge crossing case
        try:
            s = setup_polyslab(vertices, dilation, angle, bounds)
        except:
            continue
        assert np.isclose(minimal_edge_length(s.base_polygon), 0, atol=1e-4)

        # verify it is indeed maximal allowed
        dilation = 0.0
        bounds = (0, max_dist - 1e-10)
        angle = np.pi / 4

        # avoid vertex-edge crossing case
        try:
            s = setup_polyslab(vertices, dilation, angle, bounds)
        except:
            continue
        assert np.isclose(minimal_edge_length(s.top_polygon), 0, atol=1e-4)


def test_shift_height():
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
        _, max_dist = s._crossing_detection(s.base_polygon, -100)
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


def test_intersection_with_inside():
    """Make sure intersection produces the same result as inside"""

    N = 10  # number of vertices
    Lx = 10  # maximal length in x,y direction
    angle_list = [-np.pi / 4, np.pi / 4]
    dilation = 0.0

    # generate vertices for testing
    Ntest = 50
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
                    s = setup_polyslab(vertices, dilation, angle_tmp, bounds, axis=axis)
                    # set up proper thickness
                    _, max_dist = s._crossing_detection(s.base_polygon, -100)

                bounds = (-(max_dist * 0.95) / 2, (max_dist * 0.95) / 2)

                # avoid vertex-edge crossing case
                try:
                    s = setup_polyslab(vertices, dilation, angle, bounds, axis=axis)
                except:
                    continue

                xyz = np.random.random((10, 3)) * 2 * Lx - Lx
                ### side x
                xp = 0
                yp = xyz[:, 1]
                zp = xyz[:, 2]
                shape_intersect = s.intersections(x=xp)

                xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                res_inside_array = s.inside(xarray, yarray, zarray)

                for i in range(len(yp)):
                    for j in range(len(zp)):
                        # inside
                        res_inside = s.inside(xp, yp[i], zp[j])
                        assert res_inside_array[0, i, j] == res_inside
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
                shape_intersect = s.intersections(y=yp)

                xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                res_inside_array = s.inside(xarray, yarray, zarray)

                for i in range(len(xp)):
                    for j in range(len(zp)):
                        # inside
                        res_inside = s.inside(xp[i], yp, zp[j])
                        assert res_inside == res_inside_array[i, 0, j]
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
                shape_intersect = s.intersections(z=zp)

                xarray, yarray, zarray = np.meshgrid(xp, yp, zp, indexing="ij")
                res_inside_array = s.inside(xarray, yarray, zarray)

                for i in range(len(xp)):
                    for j in range(len(yp)):
                        # inside
                        res_inside = s.inside(xp[i], yp[j], zp)
                        assert res_inside == res_inside_array[i, j, 0]
                        # intersect
                        res_inter = False
                        for shape in shape_intersect:
                            if shape.covers(Point(xp[i], yp[j])):
                                res_inter = True
                        assert res_inter == res_inside


def test_bound():
    """
    Make sure bound works, even though it might not be tight.
    """
    N = 10  # number of vertices
    Lx = 10  # maximal length in x,y direction
    for i in range(50):
        vertices = convert_valid_polygon(np.random.random((N, 2)) * Lx)
        vertices = np.array(vertices)  # .astype("float32")

        ### positive dilation
        dilation = 0
        angle = 0
        bounds = (0, 1)
        s = setup_polyslab(vertices, dilation, angle, bounds)
        _, max_dist = s._crossing_detection(s.base_polygon, 100)
        # verify it is indeed maximal allowed
        dilation = 1
        if max_dist is not None:
            dilation = max_dist - 1e-10
        bounds = (0, 1)
        angle = 0.0
        # avoid vertex-edge crossing case
        try:
            s = setup_polyslab(vertices, dilation, angle, bounds)
        except:
            continue
        validate_poly_bound(s)

        ## sidewall
        dilation = 0
        angle = 0
        bounds = (0, 1)
        s = setup_polyslab(vertices, dilation, angle, bounds)
        # set up proper thickness
        _, max_dist = s._crossing_detection(s.base_polygon, -100)
        dilation = 0.0
        bounds = (0, (max_dist * 0.95))
        angle = np.pi / 4
        # avoid vertex-edge crossing case
        try:
            s = setup_polyslab(vertices, dilation, angle, bounds)
        except:
            continue
        validate_poly_bound(s)
