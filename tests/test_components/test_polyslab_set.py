"""Tests for custom-medium contour extraction and polyslab-set workflow helpers."""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import gdstk
import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry import Polygon as ShapelyPolygon

import tidy3d as td
import tidy3d.plugins.invdes.polyslab_set as polyslab_set_module
from tidy3d.components.geometry.contour_conversion import (
    _dataarray_to_polyslab_data_and_permittivity_bounds,
    _geometry_to_polygons,
    contours_to_polyslab_data,
    dataarray_to_polyslab_data,
    gdstk_contours_from_custom_medium,
)
from tidy3d.components.grid.grid import Coords
from tidy3d.components.structure import resolve_foreground_background_media
from tidy3d.plugins.autograd import make_curvature_penalty
from tidy3d.plugins.invdes import PolySlabSet, curvature_penalty, smooth_polygon_vertices


def make_custom_medium_from_eps(eps_xyz, x, y, z):
    """Construct a ``td.CustomMedium`` from 3D permittivity values."""
    data = np.asarray(eps_xyz, dtype=float)[..., None]
    coords = {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
        "f": np.array([td.C_0]),
    }
    eps_data = td.ScalarFieldDataArray(data, coords=coords)
    eps_dataset = td.PermittivityDataset(eps_xx=eps_data, eps_yy=eps_data, eps_zz=eps_data)
    return td.CustomMedium(eps_dataset=eps_dataset)


def make_2d_custom_medium_from_mask(mask, x=None, y=None, eps_hi=12.0, eps_lo=1.0):
    """Construct a z-singleton custom medium from a 2D mask."""
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    x = np.asarray(np.arange(nx) if x is None else x, dtype=float)
    y = np.asarray(np.arange(ny) if y is None else y, dtype=float)
    eps_xy = np.where(mask, eps_hi, eps_lo).astype(float)
    eps_xyz = eps_xy[:, :, None]
    return make_custom_medium_from_eps(eps_xyz=eps_xyz, x=x, y=y, z=[0.0])


def make_scalar_medium(permittivity):
    """Construct a uniform scalar medium."""
    return td.Medium(permittivity=permittivity)


def make_2d_anisotropic_custom_medium_from_mask(mask, x=None, y=None, eps_hi=12.0, eps_lo=1.0):
    """Construct a z-singleton anisotropic custom medium with mixed interp methods."""
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    x = np.asarray(np.arange(nx) if x is None else x, dtype=float)
    y = np.asarray(np.arange(ny) if y is None else y, dtype=float)
    coords = {"x": x, "y": y, "z": np.array([0.0], dtype=float)}
    eps = np.where(mask, eps_hi, eps_lo).astype(float)[..., None]
    xx = td.CustomMedium(
        permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="nearest"
    )
    yy = td.CustomMedium(
        permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="linear"
    )
    zz = td.CustomMedium(
        permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="nearest"
    )
    return td.CustomAnisotropicMedium(xx=xx, yy=yy, zz=zz)


def make_2d_spatial_data_from_mask(mask, x=None, y=None, eps_hi=12.0, eps_lo=1.0):
    """Construct a z-singleton permittivity data array from a 2D mask."""
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    x = np.asarray(np.arange(nx) if x is None else x, dtype=float)
    y = np.asarray(np.arange(ny) if y is None else y, dtype=float)
    eps = np.where(mask, eps_hi, eps_lo).astype(float)[..., None]
    return td.SpatialDataArray(eps, coords={"x": x, "y": y, "z": np.array([0.0])})


def make_2d_custom_debye_from_mask(mask, x=None, y=None, eps_hi=3.0, eps_lo=1.0):
    """Construct a z-singleton dispersive custom Debye medium from a 2D mask."""
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    x = np.asarray(np.arange(nx) if x is None else x, dtype=float)
    y = np.asarray(np.arange(ny) if y is None else y, dtype=float)
    coords = {"x": x, "y": y, "z": np.array([0.0], dtype=float)}

    eps_inf = td.SpatialDataArray(
        np.where(mask, eps_hi, eps_lo).astype(float)[..., None], coords=coords
    )
    delta_eps = td.SpatialDataArray(np.full((nx, ny, 1), 0.5, dtype=float), coords=coords)
    tau = td.SpatialDataArray(np.full((nx, ny, 1), 1e-15, dtype=float), coords=coords)
    return td.CustomDebye(eps_inf=eps_inf, coeffs=((delta_eps, tau),))


def make_2d_triangular_grid_custom_medium(permittivity=12.0):
    """Construct a 2D unstructured custom medium."""
    points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )
    cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )
    values = td.IndexedDataArray([permittivity] * 4, dims=("index",))
    grid = td.TriangularGridDataset(
        normal_axis=2,
        normal_pos=0.0,
        points=points,
        cells=cells,
        values=values,
    )
    return td.CustomMedium(permittivity=grid)


def polygons_area(polygons):
    """Total area of a polygon list."""
    if not polygons:
        return 0.0
    return float(sum(poly.area() for poly in polygons))


def symmetric_difference_area(polygons_a, polygons_b):
    """Area of symmetric difference between two polygon lists."""
    a_minus_b = gdstk.boolean(polygons_a, polygons_b, "not")
    b_minus_a = gdstk.boolean(polygons_b, polygons_a, "not")
    if not a_minus_b and not b_minus_a:
        return 0.0
    sym = gdstk.boolean(a_minus_b, b_minus_a, "or")
    return polygons_area(sym)


def polyslab_set_to_gdstk_merged(polyslab_set):
    """Reconstruct filled polygons from polyslab-set solids and holes."""
    solids = [
        gdstk.Polygon(np.asarray(ps.vertices, dtype=float)) for ps in polyslab_set.solid_polyslabs
    ]
    holes = [
        gdstk.Polygon(np.asarray(ps.vertices, dtype=float)) for ps in polyslab_set.hole_polyslabs
    ]
    return gdstk.boolean(solids, holes, "not")


def max_ring_edge(vertices):
    """Maximum edge length of one closed ring."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] < 2:
        return 0.0
    edges = np.roll(vertices, -1, axis=0) - vertices
    return float(np.max(np.linalg.norm(edges, axis=1)))


def make_polyslab_set_from_contours(
    contours,
    *,
    slab_bounds,
    axis,
    boundary_step,
    frame_bounds,
    in_plane_step,
    min_hole_area=0.0,
    min_island_area=0.0,
    smooth_sigma=0.0,
):
    """Construct a ``PolySlabSet`` from raw contours via the shared conversion path."""
    contour_data = contours_to_polyslab_data(
        contours=contours,
        slab_bounds=slab_bounds,
        axis=axis,
        boundary_step=boundary_step,
        frame_bounds=frame_bounds,
        in_plane_step=in_plane_step,
        min_hole_area=min_hole_area,
        min_island_area=min_island_area,
    )
    return PolySlabSet.from_contour_data(contour_data, smooth_sigma=smooth_sigma)


def cyclic_triplet_penalties(vertices, penalty_fn):
    """Evaluate a local penalty on each cyclic 3-point window of a closed ring."""
    ring = np.asarray(vertices)
    return np.array(
        [
            penalty_fn(np.stack((ring[idx - 1], ring[idx], ring[(idx + 1) % len(ring)])))
            for idx in range(len(ring))
        ]
    )


def test_smooth_polygon_vertices():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]], dtype=float)

    no_smooth = smooth_polygon_vertices(vertices, sigma=0.0)
    assert np.allclose(no_smooth, vertices)

    smoothed = smooth_polygon_vertices(vertices, sigma=1.0)
    assert smoothed.shape == vertices.shape
    assert not np.allclose(smoothed, vertices)

    with pytest.raises(ValueError):
        _ = smooth_polygon_vertices(vertices, sigma=-1.0)


def test_gdstk_contours_from_custom_medium_pixel_exact():
    x = np.array([-1.0, 0.0, 2.0], dtype=float)
    y = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    medium = make_2d_custom_medium_from_mask(mask=mask, x=x, y=y)

    contours, frame_bounds, in_plane_step, eps_min, eps_max, threshold = (
        gdstk_contours_from_custom_medium(
            medium=medium,
            axis=2,
            plane_position=0.0,
            bounds_xyz=((x[0], y[0], 0.0), (x[-1], y[-1], 0.0)),
            permittivity_threshold=6.0,
            frequency=td.C_0,
            pixel_exact=True,
        )
    )

    assert len(contours) == int(mask.sum())
    assert frame_bounds[0] == pytest.approx((x[0], y[0]))
    assert frame_bounds[1] == pytest.approx((x[-1], y[-1]))
    assert in_plane_step == pytest.approx(0.5)
    assert eps_min == pytest.approx(1.0)
    assert eps_max == pytest.approx(12.0)
    assert threshold == pytest.approx(6.0)


def test_gdstk_contours_from_custom_medium_non_pixel_exact_avoids_double_interpolation(
    monkeypatch,
):
    medium = make_2d_custom_medium_from_mask(mask=np.ones((5, 5), dtype=bool))

    call_count = 0
    orig_spatial_interp = Coords.spatial_interp

    def counting_spatial_interp(self, array, interp_method, fill_value="extrapolate"):
        nonlocal call_count
        call_count += 1
        return orig_spatial_interp(self, array, interp_method, fill_value)

    monkeypatch.setattr(Coords, "spatial_interp", counting_spatial_interp)

    contours, *_ = gdstk_contours_from_custom_medium(
        medium=medium,
        axis=2,
        plane_position=0.0,
        bounds_xyz=((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)),
        permittivity_threshold=6.0,
        frequency=td.C_0,
        pixel_exact=False,
    )

    assert contours
    assert call_count == 1


def test_geometry_to_polygons_rejects_non_polygon_geometry_collection():
    geometry = GeometryCollection(
        [
            ShapelyPolygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (1.0, 1.0)]),
        ]
    )

    with pytest.raises(TypeError, match="Expected polygonal geometry, got GeometryCollection"):
        _geometry_to_polygons(geometry)


def test_contours_to_polyslab_set_holes_and_min_hole_area():
    outer = gdstk.rectangle((0.0, 0.0), (5.0, 5.0))
    hole_large = gdstk.rectangle((1.0, 1.0), (2.0, 2.0))
    hole_tiny = gdstk.rectangle((3.0, 3.0), (3.4, 3.4))
    contours = gdstk.boolean([outer], [hole_large, hole_tiny], "not")

    polyslab_set = make_polyslab_set_from_contours(
        contours=contours,
        slab_bounds=(-0.1, 0.1),
        axis=2,
        boundary_step=0.5,
        frame_bounds=((0.0, 0.0), (5.0, 5.0)),
        in_plane_step=0.5,
        min_hole_area=0.3,
    )

    assert len(polyslab_set.solid_polyslabs) == 1
    assert len(polyslab_set.hole_polyslabs) == 1
    assert len(polyslab_set.frame_boundary_vertex_mask) == 2
    assert np.any(polyslab_set.solid_frame_boundary_vertex_mask[0])
    assert not np.any(polyslab_set.hole_frame_boundary_vertex_mask[0])


def test_contours_to_polyslab_set_hole_boundary_mask_matches_hole_vertices():
    outer = gdstk.rectangle((0.0, 0.0), (5.0, 5.0))
    hole = gdstk.Polygon(((1.0, 1.0), (1.6, 2.2), (2.8, 1.4), (2.1, 0.6)))
    contours = gdstk.boolean([outer], [hole], "not")

    polyslab_set = make_polyslab_set_from_contours(
        contours=contours,
        slab_bounds=(-0.1, 0.1),
        axis=2,
        boundary_step=10.0,
        frame_bounds=((0.0, 0.0), (5.0, 2.2)),
        in_plane_step=0.5,
        min_hole_area=0.1,
    )

    assert len(polyslab_set.hole_polyslabs) == 1
    hole_vertices = np.asarray(polyslab_set.hole_polyslabs[0].vertices, dtype=float)
    tol = max(0.5 * 1e-3, 1e-12)
    expected_mask = (
        (np.abs(hole_vertices[:, 0] - 0.0) <= tol)
        | (np.abs(hole_vertices[:, 0] - 5.0) <= tol)
        | (np.abs(hole_vertices[:, 1] - 0.0) <= tol)
        | (np.abs(hole_vertices[:, 1] - 2.2) <= tol)
    )
    np.testing.assert_array_equal(polyslab_set.hole_frame_boundary_vertex_mask[0], expected_mask)


def test_contours_to_polyslab_set_separate_hole_and_island_area_thresholds():
    outer = gdstk.rectangle((0.0, 0.0), (5.0, 5.0))
    hole_large = gdstk.rectangle((1.0, 1.0), (2.0, 2.0))
    hole_tiny = gdstk.rectangle((3.0, 3.0), (3.4, 3.4))
    island_tiny = gdstk.rectangle((6.0, 1.0), (6.4, 1.4))
    contours = gdstk.boolean([outer], [hole_large, hole_tiny], "not")
    contours.append(island_tiny)

    polyslab_set = make_polyslab_set_from_contours(
        contours=contours,
        slab_bounds=(-0.1, 0.1),
        axis=2,
        boundary_step=0.5,
        frame_bounds=((0.0, 0.0), (7.0, 5.0)),
        in_plane_step=0.5,
        min_hole_area=0.3,
        min_island_area=0.1,
    )

    assert len(polyslab_set.solid_polyslabs) == 2
    assert len(polyslab_set.hole_polyslabs) == 1
    assert polyslab_set.ring_types == ("solid", "hole", "solid")
    structures = polyslab_set.to_structures(
        foreground_medium=make_scalar_medium(12.0),
        background_medium=make_scalar_medium(1.5),
        name_prefix="ordered",
    )
    assert [structure.name for structure in structures] == [
        "ordered_solid_0",
        "ordered_hole_0",
        "ordered_solid_1",
    ]


def test_contours_to_polyslab_set_preserves_nested_order_for_thin_frame():
    outer = gdstk.rectangle((0.0, 0.0), (10.0, 10.0))
    hole = gdstk.rectangle((0.2, 0.2), (9.8, 9.8))
    island = gdstk.rectangle((3.0, 3.0), (7.0, 7.0))
    contours = gdstk.boolean([outer], [hole], "not")
    contours.append(island)

    polyslab_set = make_polyslab_set_from_contours(
        contours=contours,
        slab_bounds=(-0.1, 0.1),
        axis=2,
        boundary_step=10.0,
        frame_bounds=((0.0, 0.0), (10.0, 10.0)),
        in_plane_step=0.5,
        min_hole_area=0.1,
        min_island_area=0.1,
    )

    assert len(polyslab_set.solid_polyslabs) == 2
    assert len(polyslab_set.hole_polyslabs) == 1
    assert polyslab_set.ring_types == ("solid", "hole", "solid")
    structures = polyslab_set.to_structures(
        foreground_medium=make_scalar_medium(12.0),
        background_medium=make_scalar_medium(1.5),
        name_prefix="thin_frame",
    )
    assert [structure.name for structure in structures] == [
        "thin_frame_solid_0",
        "thin_frame_hole_0",
        "thin_frame_solid_1",
    ]


def test_custom_medium_to_polyslabs_infers_axis_and_boundary_step():
    x = np.arange(6, dtype=float) * 0.5
    y = np.arange(5, dtype=float) * 0.25
    mask = np.zeros((len(x), len(y)), dtype=bool)
    mask[1:5, 1:4] = True
    medium = make_2d_custom_medium_from_mask(mask=mask, x=x, y=y)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=False,
    )

    assert len(polyslab_set.solid_polyslabs) >= 1
    assert all(polyslab.axis == 2 for polyslab in polyslab_set.polyslabs)
    assert polyslab_set.max_edge_length() <= 0.25 + 1e-12


def test_dataarray_to_polyslab_data_infers_axis_and_boundary_step():
    x = np.arange(6, dtype=float) * 0.5
    y = np.arange(5, dtype=float) * 0.25
    mask = np.zeros((len(x), len(y)), dtype=bool)
    mask[1:5, 1:4] = True
    data = make_2d_spatial_data_from_mask(mask=mask, x=x, y=y)

    contour_data = dataarray_to_polyslab_data(
        data,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=False,
    )

    assert contour_data.solid_polyslabs
    assert all(polyslab.axis == 2 for polyslab in contour_data.solid_polyslabs)
    assert contour_data.in_plane_step == pytest.approx(0.25)


def test_dataarray_to_polyslab_data_private_helper_returns_derived_min_max():
    x = np.arange(6, dtype=float) * 0.5
    y = np.arange(5, dtype=float) * 0.25
    mask = np.zeros((len(x), len(y)), dtype=bool)
    mask[1:5, 1:4] = True
    data = make_2d_spatial_data_from_mask(mask=mask, x=x, y=y)

    contour_data, permittivity_min, permittivity_max = (
        _dataarray_to_polyslab_data_and_permittivity_bounds(
            data,
            slab_bounds=(-0.2, 0.2),
            threshold=6.0,
            pixel_exact=False,
        )
    )

    assert contour_data.solid_polyslabs
    assert permittivity_min == pytest.approx(1.0)
    assert permittivity_max == pytest.approx(12.0)


def test_custom_medium_to_polyslabs_reuses_precomputed_eps_components(monkeypatch):
    medium = make_2d_custom_medium_from_mask(mask=np.ones((5, 5), dtype=bool))
    original_eps_dataarray_freq = type(medium).eps_dataarray_freq
    call_count = 0

    def wrapped_eps_dataarray_freq(self, frequency):
        nonlocal call_count
        call_count += 1
        return original_eps_dataarray_freq(self, frequency)

    monkeypatch.setattr(type(medium), "eps_dataarray_freq", wrapped_eps_dataarray_freq)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=True,
    )

    assert polyslab_set.solid_polyslabs
    assert call_count == 1


def test_custom_medium_to_polyslabs_separate_hole_and_island_area_thresholds():
    x = np.arange(7, dtype=float)
    y = np.arange(7, dtype=float)
    mask = np.zeros((len(x), len(y)), dtype=bool)
    mask[1:5, 1:5] = True
    mask[2, 2] = False
    mask[5, 5] = True
    medium = make_2d_custom_medium_from_mask(mask=mask, x=x, y=y)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=True,
        min_hole_area=1.1,
        min_island_area=0.9,
    )

    assert len(polyslab_set.solid_polyslabs) == 2
    assert len(polyslab_set.hole_polyslabs) == 0


def test_custom_medium_to_polyslabs_rejects_non_2d_medium():
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 2.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    eps_xyz = np.ones((len(x), len(y), len(z)), dtype=float) * 12.0
    medium = make_custom_medium_from_eps(eps_xyz=eps_xyz, x=x, y=y, z=z)

    with pytest.raises(ValueError, match="supports only 2D media"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
        )


def test_custom_medium_to_polyslabs_rejects_unstructured_medium():
    medium = make_2d_triangular_grid_custom_medium(permittivity=12.0)

    with pytest.raises(NotImplementedError, match="does not support unstructured datasets"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_hole_area": -0.1}, "min_hole_area"),
        ({"min_island_area": -0.1}, "min_island_area"),
    ],
)
def test_custom_medium_to_polyslabs_rejects_negative_separate_area_thresholds(kwargs, message):
    medium = make_2d_custom_medium_from_mask(mask=np.ones((3, 3), dtype=bool))

    with pytest.raises(ValueError, match=message):
        _ = PolySlabSet.from_custom_medium(
            medium,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
            **kwargs,
        )


def test_custom_medium_to_polyslabs_rejects_effectively_1d_medium_even_with_axis():
    x = np.array([0.0], dtype=float)
    y = np.array([0.0], dtype=float)
    z = np.array([-0.2, 0.0, 0.2], dtype=float)
    eps_xyz = np.ones((len(x), len(y), len(z)), dtype=float) * 12.0
    medium = make_custom_medium_from_eps(eps_xyz=eps_xyz, x=x, y=y, z=z)

    with pytest.raises(ValueError, match="exactly one coordinate dimension"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            axis=0,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
        )


def test_custom_medium_to_polyslabs_dispersive_default_frequency_succeeds():
    x = np.linspace(-0.5, 0.5, 31)
    y = np.linspace(-0.5, 0.5, 31)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= 0.12**2
    medium = make_2d_custom_debye_from_mask(mask=mask, x=x, y=y)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.1, 0.1),
        threshold=2.0,
        pixel_exact=True,
    )

    assert len(polyslab_set.solid_polyslabs) >= 1


def test_custom_medium_to_polyslabs_dispersive_with_frequency_succeeds():
    x = np.linspace(-0.5, 0.5, 31)
    y = np.linspace(-0.5, 0.5, 31)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= 0.12**2
    medium = make_2d_custom_debye_from_mask(mask=mask, x=x, y=y)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.1, 0.1),
        threshold=2.0,
        pixel_exact=True,
        frequency=2e14,
    )

    assert len(polyslab_set.solid_polyslabs) >= 1


@pytest.mark.parametrize("frequency", [float("nan"), -float("inf")])
def test_custom_medium_to_polyslabs_frequency_validation(frequency):
    x = np.linspace(-0.5, 0.5, 31)
    y = np.linspace(-0.5, 0.5, 31)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= 0.12**2
    medium = make_2d_custom_debye_from_mask(mask=mask, x=x, y=y)

    with pytest.raises(ValueError, match="must be finite or \\+inf"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            slab_bounds=(-0.1, 0.1),
            threshold=2.0,
            pixel_exact=True,
            frequency=frequency,
        )


@pytest.mark.parametrize("kwarg_name", ["foreground_medium", "background_medium"])
def test_polyslab_set_to_structures_rejects_custom_medium(kwarg_name):
    medium = make_2d_custom_medium_from_mask(mask=np.ones((3, 3), dtype=bool))
    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.1, 0.1),
        threshold=6.0,
    )
    varying_medium = make_2d_custom_medium_from_mask(
        mask=np.array([[False, True, False], [True, True, True], [False, True, False]])
    )
    kwargs = {
        "foreground_medium": make_scalar_medium(12.0),
        "background_medium": make_scalar_medium(1.0),
    }
    kwargs[kwarg_name] = varying_medium

    with pytest.raises(ValueError, match="non-custom"):
        _ = polyslab_set.to_structures(**kwargs, name_prefix="invalid")


def test_custom_medium_to_polyslabs_threshold_validation():
    medium = make_2d_custom_medium_from_mask(mask=np.ones((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="threshold.*finite"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            slab_bounds=(-0.1, 0.1),
            threshold=float("inf"),
        )


def test_custom_medium_to_polyslabs_axis_mismatch_rejected():
    x = np.linspace(-0.5, 0.5, 31)
    y = np.linspace(-0.5, 0.5, 31)
    z = np.asarray([0.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= 0.15**2
    eps = np.where(mask, 2.0, 1.0).astype(float)[..., None]
    medium = td.CustomMedium(permittivity=td.SpatialDataArray(eps, coords={"x": x, "y": y, "z": z}))

    with pytest.raises(
        ValueError, match="must correspond to a coordinate dimension with exactly one"
    ):
        _ = PolySlabSet.from_custom_medium(
            medium,
            axis=1,
            slab_bounds=(-0.2, 0.2),
            threshold=1.5,
            pixel_exact=True,
            boundary_step=0.05,
        )


def test_custom_medium_to_polyslabs_boundary_step_controls_edge_length():
    x = np.linspace(-1.0, 1.0, 61)
    y = np.linspace(-1.0, 1.0, 61)
    z = np.asarray([0.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx + 0.1) ** 2 + (yy - 0.05) ** 2 <= 0.65**2

    eps = np.where(mask, 2.0, 1.0).astype(float)[..., None]
    medium = td.CustomMedium(permittivity=td.SpatialDataArray(eps, coords={"x": x, "y": y, "z": z}))

    coarse = PolySlabSet.from_custom_medium(
        medium,
        axis=2,
        slab_bounds=(-0.1, 0.1),
        threshold=1.5,
        pixel_exact=True,
        boundary_step=0.20,
    )
    fine = PolySlabSet.from_custom_medium(
        medium,
        axis=2,
        slab_bounds=(-0.1, 0.1),
        threshold=1.5,
        pixel_exact=True,
        boundary_step=0.04,
    )

    coarse_max = max(max_ring_edge(np.asarray(ps.vertices)) for ps in coarse.polyslabs)
    fine_max = max(max_ring_edge(np.asarray(ps.vertices)) for ps in fine.polyslabs)

    assert coarse_max <= 0.20 + 1e-9
    assert fine_max <= 0.04 + 1e-9
    assert fine_max <= coarse_max + 1e-12


def test_custom_medium_to_polyslabs_slab_bounds_required():
    x = np.linspace(-0.5, 0.5, 31)
    y = np.linspace(-0.5, 0.5, 31)
    z = np.asarray([0.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= 0.15**2
    eps = np.where(mask, 2.0, 1.0).astype(float)[..., None]
    medium = td.CustomMedium(permittivity=td.SpatialDataArray(eps, coords={"x": x, "y": y, "z": z}))

    with pytest.raises(ValueError, match="slab_bounds"):
        _ = PolySlabSet.from_custom_medium(
            medium,
            threshold=1.5,
            pixel_exact=True,
            boundary_step=0.05,
        )

    inferred = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.3, 0.3),
        threshold=1.5,
        pixel_exact=True,
        boundary_step=0.05,
    )
    assert inferred.solid_polyslabs
    assert inferred.solid_polyslabs[0].slab_bounds == (-0.3, 0.3)


def test_structure_list_from_custom_medium_defaults_media_and_supports_smoothing():
    x = np.linspace(-1.0, 1.0, 61)
    y = np.linspace(-1.0, 1.0, 61)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx + 0.1) ** 2 + (yy - 0.05) ** 2 <= 0.55**2
    medium = make_2d_custom_medium_from_mask(mask=mask, x=x, y=y, eps_hi=12.0, eps_lo=1.0)

    unsmoothed = td.Structure.list_from_custom_medium(
        medium=medium,
        slab_bounds=(-0.1, 0.1),
        threshold=6.0,
        pixel_exact=False,
        smooth_sigma=0.0,
    )
    smoothed = td.Structure.list_from_custom_medium(
        medium=medium,
        slab_bounds=(-0.1, 0.1),
        threshold=6.0,
        pixel_exact=False,
        smooth_sigma=1.0,
    )

    assert unsmoothed
    assert {structure.medium.permittivity for structure in unsmoothed} == {12.0}
    assert not np.allclose(
        np.asarray(unsmoothed[0].geometry.vertices),
        np.asarray(smoothed[0].geometry.vertices),
    )


def test_structure_list_from_custom_medium_accepts_explicit_infinite_frequency():
    medium = make_2d_custom_medium_from_mask(mask=np.ones((5, 5), dtype=bool))

    structures = td.Structure.list_from_custom_medium(
        medium=medium,
        slab_bounds=(-0.1, 0.1),
        threshold=6.0,
        frequency=float("inf"),
    )

    assert len(structures) == 1
    assert structures[0].medium.permittivity == pytest.approx(12.0)


def test_structure_list_from_custom_medium_rejects_uniform_custom_medium_override():
    medium = make_2d_custom_medium_from_mask(mask=np.ones((5, 5), dtype=bool))
    foreground_medium = make_2d_custom_medium_from_mask(
        mask=np.ones((5, 5), dtype=bool), eps_hi=2.5, eps_lo=2.5
    )

    with pytest.raises(ValueError, match="non-custom"):
        _ = td.Structure.list_from_custom_medium(
            medium=medium,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
            foreground_medium=foreground_medium,
            background_medium=make_scalar_medium(1.0),
        )


def test_resolve_foreground_background_media_validates_selected_defaults():
    with pytest.raises(ValueError, match="default_foreground_permittivity"):
        _ = resolve_foreground_background_media(
            default_foreground_permittivity=float("nan"),
            default_background_permittivity=1.0,
            missing_message="unused",
        )

    with pytest.raises(ValueError, match="default_background_permittivity"):
        _ = resolve_foreground_background_media(
            default_foreground_permittivity=12.0,
            default_background_permittivity=float("inf"),
            missing_message="unused",
        )


def test_polyslab_set_from_custom_medium_supports_mixed_component_interp_methods():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    medium = make_2d_anisotropic_custom_medium_from_mask(mask=mask, eps_hi=2.0, eps_lo=1.0)

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.1, 0.1),
        threshold=1.5,
        pixel_exact=False,
    )

    assert polyslab_set.solid_polyslabs


def test_polyslab_set_from_dataarray_allows_subunity_permittivity_slice():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    data = make_2d_spatial_data_from_mask(mask=mask, eps_hi=0.8, eps_lo=0.5)

    polyslab_set = PolySlabSet.from_dataarray(
        data,
        slab_bounds=(-0.1, 0.1),
        threshold=0.65,
        pixel_exact=False,
    )

    assert polyslab_set.solid_polyslabs


@pytest.mark.parametrize("kwarg_name", ["foreground_medium", "background_medium"])
def test_structure_list_from_custom_medium_rejects_custom_medium_override(kwarg_name):
    medium = make_2d_custom_medium_from_mask(mask=np.ones((5, 5), dtype=bool))
    varying_medium = make_2d_custom_medium_from_mask(
        mask=np.array(
            [
                [False, False, True, False, False],
                [False, True, True, True, False],
                [True, True, True, True, True],
                [False, True, True, True, False],
                [False, False, True, False, False],
            ]
        )
    )

    with pytest.raises(ValueError, match="non-custom"):
        _ = td.Structure.list_from_custom_medium(
            medium=medium,
            slab_bounds=(-0.1, 0.1),
            threshold=6.0,
            **{kwarg_name: varying_medium},
        )


def test_structure_list_from_dataarray_rejects_subunity_inferred_medium():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    data = make_2d_spatial_data_from_mask(mask=mask, eps_hi=0.8, eps_lo=0.5)

    with pytest.raises(ValueError, match="Provide 'foreground_medium' explicitly"):
        _ = td.Structure.list_from_dataarray(
            data=data,
            slab_bounds=(-0.1, 0.1),
            threshold=0.65,
        )


def test_structure_list_from_dataarray_supports_area_thresholds():
    x = np.arange(7, dtype=float)
    y = np.arange(7, dtype=float)
    mask = np.zeros((len(x), len(y)), dtype=bool)
    mask[1:5, 1:5] = True
    mask[2, 2] = False
    mask[5, 5] = True
    data = make_2d_spatial_data_from_mask(mask=mask, x=x, y=y, eps_hi=12.0, eps_lo=1.0)

    structures = td.Structure.list_from_dataarray(
        data=data,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=True,
        min_hole_area=1.1,
        min_island_area=0.9,
    )

    assert len(structures) == 2
    assert {structure.medium.permittivity for structure in structures} == {12.0}


def test_polyslab_set_from_dataarray_matches_custom_medium():
    mask = np.zeros((6, 5), dtype=bool)
    mask[1:5, 1:4] = True
    data = make_2d_spatial_data_from_mask(mask=mask, eps_hi=12.0, eps_lo=1.0)
    medium = make_2d_custom_medium_from_mask(mask=mask, eps_hi=12.0, eps_lo=1.0)

    from_data = PolySlabSet.from_dataarray(
        data,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=False,
    )
    from_medium = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.2, 0.2),
        threshold=6.0,
        pixel_exact=False,
    )

    assert len(from_data.solid_polyslabs) == len(from_medium.solid_polyslabs)
    assert len(from_data.hole_polyslabs) == len(from_medium.hole_polyslabs)
    assert from_data.max_edge_length() == pytest.approx(from_medium.max_edge_length())


def test_polyslab_set_helpers():
    solid = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    hole = td.PolySlab(
        vertices=np.array([[0.5, 0.5], [0.5, 1.0], [1.0, 1.0], [1.0, 0.5]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(hole,),
        solid_frame_boundary_vertex_mask=(np.array([True, False, False, True]),),
        hole_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        frame_bounds=((0.0, 0.0), (2.0, 2.0)),
        in_plane_step=0.5,
    )

    flat = polyslab_set.flatten_ring_vertices()
    assert flat.shape == (16,)

    displaced = flat + 0.2
    displaced_set = polyslab_set.with_flat_ring_vertices(displaced)
    updated_set = polyslab_set.update(
        displaced,
        freeze_boundary=True,
        respect_bounds=True,
        smooth_sigma=0.0,
    )
    clipped_set = displaced_set.clip_to_bounds(((0.0, 0.0), (2.0, 2.0)))
    np.testing.assert_allclose(
        updated_set.flatten_ring_vertices()[polyslab_set.frame_boundary_mask_flat(repeat_xy=True)],
        polyslab_set.flatten_ring_vertices()[polyslab_set.frame_boundary_mask_flat(repeat_xy=True)],
    )
    for ring in clipped_set.ring_vertices:
        assert np.all(ring[:, 0] >= 0.0)
        assert np.all(ring[:, 0] <= 2.0)
        assert np.all(ring[:, 1] >= 0.0)
        assert np.all(ring[:, 1] <= 2.0)

    flat_mask = clipped_set.frame_boundary_mask_flat(repeat_xy=True)
    assert flat_mask.shape == clipped_set.flatten_ring_vertices().shape

    assert clipped_set.smooth(sigma=0.0) is clipped_set
    smoothed = clipped_set.smooth(sigma=1.0)
    assert smoothed is not clipped_set
    assert smoothed.ring_vertex_counts == clipped_set.ring_vertex_counts

    def penalty_fn(points):
        return float(np.mean(points[:, 0]))

    penalty_no_ignore = curvature_penalty(
        polyslab_set, penalty_fn=penalty_fn, ignore_boundary_vertices=False
    )
    penalty_ignore = curvature_penalty(
        polyslab_set, penalty_fn=penalty_fn, ignore_boundary_vertices=True
    )
    expected_triplet_penalties = []
    expected_ignore_penalties = []
    for polyslab, frame_mask in zip(
        polyslab_set.polyslabs, polyslab_set.frame_boundary_vertex_mask
    ):
        vertices = np.asarray(polyslab.vertices)
        ring_triplet_penalties = cyclic_triplet_penalties(vertices, penalty_fn)
        expected_triplet_penalties.append(ring_triplet_penalties)
        expected_ignore_penalties.append(ring_triplet_penalties[~np.asarray(frame_mask)])

    assert penalty_no_ignore == pytest.approx(
        np.sum([np.sum(values) for values in expected_triplet_penalties])
        / np.sum([values.size for values in expected_triplet_penalties])
    )
    assert penalty_ignore == pytest.approx(
        np.sum([np.sum(values) for values in expected_ignore_penalties])
        / np.sum([values.size for values in expected_ignore_penalties])
    )

    with pytest.raises(TypeError):
        _ = polyslab_set.to_structures(name_prefix="contour")

    structures = polyslab_set.to_structures(
        foreground_medium=make_scalar_medium(12.0),
        background_medium=make_scalar_medium(1.5),
        name_prefix="contour",
    )
    assert len(structures) == 2
    assert structures[0].medium.permittivity == pytest.approx(12.0)
    assert structures[1].medium.permittivity == pytest.approx(1.5)
    assert structures[0].name == "contour_solid_0"
    assert structures[1].name == "contour_hole_0"


def test_polyslab_set_flatten_ring_vertices_is_autograd_safe():
    solid = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(),
        solid_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        hole_frame_boundary_vertex_mask=(),
        frame_bounds=((0.0, 0.0), (2.0, 1.0)),
        in_plane_step=1.0,
    )
    flat0 = anp.array(polyslab_set.flatten_ring_vertices())

    def objective(flat_vertices):
        updated = polyslab_set.with_flat_ring_vertices(flat_vertices)
        flattened = updated.flatten_ring_vertices()
        return anp.sum(flattened**2)

    grad = ag.grad(objective)(flat0)
    assert np.allclose(np.asarray(grad), 2.0 * np.asarray(flat0))


def test_polyslab_set_update_keeps_frozen_boundary_vertices_after_smoothing():
    solid = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(),
        solid_frame_boundary_vertex_mask=(np.array([True, False, False, False]),),
        hole_frame_boundary_vertex_mask=(),
        frame_bounds=((0.0, 0.0), (2.0, 2.0)),
        in_plane_step=1.0,
    )

    updated = polyslab_set.update(
        polyslab_set.flatten_ring_vertices() + 0.1,
        freeze_boundary=True,
        respect_bounds=True,
        smooth_sigma=1.0,
    )

    np.testing.assert_allclose(updated.ring_vertices[0][0], polyslab_set.ring_vertices[0][0])


def test_polyslab_set_caches_ring_refs(monkeypatch):
    solid = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    hole = td.PolySlab(
        vertices=np.array([[0.5, 0.5], [0.5, 1.0], [1.0, 1.0], [1.0, 0.5]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(hole,),
        solid_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        hole_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        frame_bounds=((0.0, 0.0), (2.0, 2.0)),
        in_plane_step=0.5,
    )

    call_count = 0
    original = polyslab_set_module.ordered_ring_refs_by_area

    def wrapped(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(polyslab_set_module, "ordered_ring_refs_by_area", wrapped)

    _ = polyslab_set.ring_types
    _ = polyslab_set.polyslabs
    _ = polyslab_set.ring_vertices
    _ = polyslab_set.ring_vertex_counts
    _ = polyslab_set.frame_boundary_vertex_mask
    _ = polyslab_set.flatten_ring_vertices()
    _ = polyslab_set.to_structures(
        foreground_medium=make_scalar_medium(12.0),
        background_medium=make_scalar_medium(1.5),
        name_prefix="cached",
    )

    assert call_count == 1


def test_polyslab_set_to_structures_is_autograd_safe():
    solid = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(),
        solid_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        hole_frame_boundary_vertex_mask=(),
        frame_bounds=((0.0, 0.0), (2.0, 1.0)),
        in_plane_step=1.0,
    )
    flat0 = anp.array(polyslab_set.flatten_ring_vertices())

    def objective(flat_vertices):
        updated = polyslab_set.with_flat_ring_vertices(flat_vertices)
        _ = updated.to_structures(
            foreground_medium=make_scalar_medium(12.0),
            background_medium=make_scalar_medium(1.5),
            name_prefix="autograd",
        )
        return anp.sum(updated.flatten_ring_vertices() ** 2)

    value, grad = ag.value_and_grad(objective)(flat0)
    assert np.isfinite(value)
    assert np.allclose(np.asarray(grad), 2.0 * np.asarray(flat0))


def test_polyslab_set_curvature_penalty_masks_after_penalty_eval():
    solid = td.PolySlab(
        vertices=np.array(
            [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [2.0, 3.0], [0.0, 2.0]], dtype=float
        ),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    boundary_mask = np.array([False, True, False, False, False])
    polyslab_set = PolySlabSet(
        solid_polyslabs=(solid,),
        hole_polyslabs=(),
        solid_frame_boundary_vertex_mask=(boundary_mask,),
        hole_frame_boundary_vertex_mask=(),
        frame_bounds=((0.0, 0.0), (3.0, 3.0)),
        in_plane_step=1.0,
    )
    penalty_fn = make_curvature_penalty(min_radius=0.1)
    vertices = np.asarray(solid.vertices)
    triplet_penalties = cyclic_triplet_penalties(vertices, penalty_fn)
    active_centers = ~boundary_mask

    penalty_no_ignore = curvature_penalty(
        polyslab_set, penalty_fn=penalty_fn, ignore_boundary_vertices=False
    )
    penalty_ignore = curvature_penalty(
        polyslab_set, penalty_fn=penalty_fn, ignore_boundary_vertices=True
    )

    assert np.isfinite(penalty_ignore)
    assert penalty_no_ignore == pytest.approx(np.mean(triplet_penalties))
    assert penalty_ignore == pytest.approx(np.mean(triplet_penalties[active_centers]))
    assert penalty_ignore != pytest.approx(penalty_fn(vertices[~boundary_mask]))


def test_curvature_penalty_accepts_single_polyslab():
    solid = td.PolySlab(
        vertices=np.array(
            [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [2.0, 3.0], [0.0, 2.0]], dtype=float
        ),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )

    def penalty_fn(points):
        return float(np.mean(points[:, 0]))

    triplet_penalties = cyclic_triplet_penalties(np.asarray(solid.vertices), penalty_fn)
    penalty = curvature_penalty(
        PolySlabSet.from_polyslab(solid),
        penalty_fn=penalty_fn,
        ignore_boundary_vertices=True,
    )

    assert penalty == pytest.approx(np.mean(triplet_penalties))


def test_polyslab_set_to_structures_preserves_nested_topology_order():
    outer = td.PolySlab(
        vertices=np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    island = td.PolySlab(
        vertices=np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    hole = td.PolySlab(
        vertices=np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]], dtype=float),
        slab_bounds=(-0.2, 0.2),
        axis=2,
    )
    polyslab_set = PolySlabSet(
        solid_polyslabs=(outer, island),
        hole_polyslabs=(hole,),
        solid_frame_boundary_vertex_mask=(
            np.array([False, False, False, False]),
            np.array([False, False, False, False]),
        ),
        hole_frame_boundary_vertex_mask=(np.array([False, False, False, False]),),
        frame_bounds=((0.0, 0.0), (6.0, 6.0)),
        in_plane_step=1.0,
    )

    structures = polyslab_set.to_structures(
        foreground_medium=make_scalar_medium(12.0),
        background_medium=make_scalar_medium(1.5),
        name_prefix="nested",
    )

    assert [structure.name for structure in structures] == [
        "nested_solid_0",
        "nested_hole_0",
        "nested_solid_1",
    ]
    assert [structure.medium.permittivity for structure in structures] == pytest.approx(
        [12.0, 1.5, 12.0]
    )


@pytest.mark.parametrize("pixel_exact", [False, True])
def test_custom_medium_to_polyslabs_matches_structure_to_gds(pixel_exact):
    x = np.arange(9, dtype=float) * 0.3
    y = np.arange(8, dtype=float) * 0.25
    mask = np.zeros((len(x), len(y)), dtype=bool)

    # Main component.
    mask[1:8, 1:7] = True
    # Hole in main component.
    mask[3:5, 3:5] = False
    # Separate island.
    mask[0:2, 6:8] = True

    medium = make_2d_custom_medium_from_mask(mask=mask, x=x, y=y, eps_hi=12.0, eps_lo=1.0)
    geometry = td.Box(
        center=((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0, 0.0),
        size=(x[-1] - x[0], y[-1] - y[0], 0.0),
    )
    structure = td.Structure(geometry=geometry, medium=medium)

    polygons_gds = structure.to_gdstk(
        z=0.0,
        permittivity_threshold=6.0,
        frequency=td.C_0,
        pixel_exact=pixel_exact,
    )

    polyslab_set = PolySlabSet.from_custom_medium(
        medium,
        slab_bounds=(-0.1, 0.1),
        axis=2,
        threshold=6.0,
        pixel_exact=pixel_exact,
    )
    polygons_contour = polyslab_set_to_gdstk_merged(polyslab_set)

    area_gds = polygons_area(polygons_gds)
    area_contour = polygons_area(polygons_contour)
    assert area_contour == pytest.approx(area_gds, rel=1e-10, abs=1e-10)

    sym_area = symmetric_difference_area(polygons_gds, polygons_contour)
    assert sym_area == pytest.approx(0.0, abs=1e-10)
