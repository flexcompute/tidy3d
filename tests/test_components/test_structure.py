from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import gdstk
import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td


def test_to_gds(tmp_path):
    geometry = td.Box(size=(2, 2, 2))
    medium = td.Medium()
    structure = td.Structure(geometry=geometry, medium=medium)

    fname = str(tmp_path / "structure.gds")
    structure.to_gds_file(fname, x=0, gds_cell_name="X")
    cell = gdstk.read_gds(fname).cells[0]
    assert cell.name == "X"
    assert len(cell.polygons) == 1
    assert np.allclose(cell.polygons[0].area(), 4.0)


def test_custom_medium_to_gds(tmp_path):
    geometry = td.Box(size=(2, 2, 2))

    nx, ny, nz = 100, 90, 80
    x = np.linspace(0, 2, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    f = np.array([td.C_0])
    mx, my, mz, _ = np.meshgrid(x, y, z, f, indexing="ij", sparse=True)
    data = 1 + 1 / (1 + (mx - 1) ** 2 + my**2 + mz**2)
    eps_diagonal_data = td.ScalarFieldDataArray(data, coords={"x": x, "y": y, "z": z, "f": f})
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")
    structure = td.Structure(geometry=geometry, medium=medium)

    fname = str(tmp_path / "structure-custom-x.gds")
    structure.to_gds_file(fname, x=1, permittivity_threshold=1.5, frequency=td.C_0)
    cell = gdstk.read_gds(fname).cells[0]
    assert np.allclose(cell.area(), np.pi, atol=1e-2)

    fname = str(tmp_path / "structure-custom-z.gds")
    structure.to_gds_file(fname, z=0, permittivity_threshold=1.5, frequency=td.C_0)
    cell = gdstk.read_gds(fname).cells[0]
    assert np.allclose(cell.area(), np.pi / 2, atol=3e-2)

    fname = str(tmp_path / "structure-empty.gds")
    structure.to_gds_file(fname, x=-0.1, permittivity_threshold=1.5, frequency=td.C_0)
    cell = gdstk.read_gds(fname).cells[0]
    assert len(cell.polygons) == 0


def test_lower_dimension_custom_medium_to_gds(tmp_path):
    geometry = td.Box(size=(2, 0, 2))

    nx, nz = 100, 80
    x = np.linspace(0, 2, nx)
    y = np.array([0.0])
    z = np.linspace(-1, 1, nz)
    f = np.array([td.C_0])
    mx, my, mz, _ = np.meshgrid(x, y, z, f, indexing="ij", sparse=True)
    data = 1 + 1 / (1 + (mx - 1) ** 2 + mz**2)
    eps_diagonal_data = td.ScalarFieldDataArray(data, coords={"x": x, "y": y, "z": z, "f": f})
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")
    structure = td.Structure(geometry=geometry, medium=medium)

    fname = str(tmp_path / "structure-custom-y.gds")
    structure.to_gds_file(fname, y=0, permittivity_threshold=1.5, frequency=td.C_0)
    cell = gdstk.read_gds(fname).cells[0]
    assert np.allclose(cell.area(), np.pi / 2, atol=3e-2)


def test_non_symmetric_custom_medium_to_gds(tmp_path):
    geometry = td.Box(size=(1, 2, 1), center=(0.5, 0, 2.5))

    nx, ny, nz = 150, 80, 180
    x = np.linspace(0, 2, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(2, 3, nz)
    f = np.array([td.C_0])
    mx, my, mz, _ = np.meshgrid(x, y, z, f, indexing="ij", sparse=True)
    data = 1 + mx + 0 * my + (mz - 2) ** 2
    print(data.min(), data.max())

    eps_diagonal_data = td.ScalarFieldDataArray(data, coords={"x": x, "y": y, "z": z, "f": f})
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")
    structure = td.Structure(geometry=geometry, medium=medium)

    fname = str(tmp_path / "structure-non-symmetric.gds")
    structure.to_gds_file(fname, y=0, permittivity_threshold=2.0, frequency=td.C_0)
    cell = gdstk.read_gds(fname).cells[0]
    assert np.allclose(cell.bounding_box(), ((0, 2), (1, 3)), atol=0.1)
    assert gdstk.inside([(0.1, 2.1), (0.5, 2.5), (0.9, 2.9)], cell.polygons) == (False, False, True)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_invalid_polyslab(axis):
    medium = td.Medium()
    vertices = [(-1, -2), (-1, 1), (1, 2), (2, 1), (0, 1), (0, 0), (1.5, -0.5), (0, -1), (0, -2)]
    i = (axis + 1) % 3
    j = (axis + 2) % 3

    ps = td.PolySlab(vertices=vertices, slab_bounds=(-1, 1), sidewall_angle=0.1, axis=axis)
    box = td.Box(size=(1, 1, 1))

    geo0 = ps.rotated(-np.pi / 3, axis)
    _ = td.Structure(geometry=geo0, medium=medium)

    geo1 = ps.rotated(-np.pi / 3, i).scaled(2, 2, 2).translated(-1, 0.5, 2).rotated(np.pi / 3, i)
    _ = td.Structure(geometry=geo1, medium=medium)

    geo2 = ps.rotated(np.pi / 4, i)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo2, medium=medium)

    geo3 = ps.rotated(np.pi / 5, j)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo3, medium=medium)

    geo4 = ps.rotated(np.pi / 6, (1, 1, 1))
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo4, medium=medium)

    geo5 = td.GeometryGroup(geometries=[ps]).rotated(np.pi / 2, j)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo5, medium=medium)

    geo6 = td.GeometryGroup(geometries=[ps - box]).rotated(np.pi / 2, i)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo6, medium=medium)

    geo7 = td.GeometryGroup(geometries=[(ps - box).rotated(np.pi / 4, j)]).rotated(-np.pi / 4, j)
    _ = td.Structure(geometry=geo7, medium=medium)

    # normal with zero component along the slab axis
    n1 = (axis, 2 * (axis - 1), 3 * (axis - 2))
    # normal with non-zero component along the slab axis
    n2 = (2 * (axis - 1), 3 * (axis - 2), axis)

    geo8 = ps.reflected(n1)
    _ = td.Structure(geometry=geo8, medium=medium)

    geo9 = ps.reflected(n2).scaled(2, 2, 2).translated(-1, 0.5, 2).reflected(n2)
    _ = td.Structure(geometry=geo9, medium=medium)

    geo10 = ps.reflected(n1)
    _ = td.Structure(geometry=geo10, medium=medium)

    geo11 = ps.reflected(n2)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo11, medium=medium)

    geo12 = td.GeometryGroup(geometries=[ps]).reflected(n2)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=geo12, medium=medium)

    geo13 = td.GeometryGroup(geometries=[(ps - box).reflected(n2)]).reflected(n2)
    _ = td.Structure(geometry=geo13, medium=medium)


def test_validation_of_structures_with_2d_materials():
    med2d = td.Medium2D(ss=td.PEC, tt=td.PEC)

    box2d = td.Box(size=(1, 0, 1))
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

    allowed_geometries = [
        box2d,
        cylinder2d,
        polyslab2d,
        geo_group2d,
        clip2d,
        shift,
        shift_rotate,
        transformed_2d,
    ]

    for geom in allowed_geometries:
        _ = td.Structure(geometry=geom, medium=med2d)

    # Of course 3d objects should be NOT compatible with 2d materials
    box3d = td.Box(size=(1, 1, 1))
    polyslab3d = td.PolySlab(
        vertices=((0, 0), (1, 0), (1, 1), (0, 1)), slab_bounds=(0, 0.5), axis=2
    )
    cylinder3d = td.Cylinder(axis=2, length=1.0, radius=1)
    sphere = td.Sphere(center=(0, 1, 2), radius=2)

    # Some test transformations that do NOT preserve the normal
    rotate_bad = td.Transformed.rotation(angle=np.pi * (1 / 8), axis=0)
    transformed_2d_bad = td.Transformed(geometry=cylinder2d, transform=rotate_bad)

    # Geometries composed of sub geometries that are not coplanar
    cylinder2d = td.Cylinder(axis=2, length=0, radius=1, center=(0, 0, 0.0))
    geo_group2d_not_coplanar = td.GeometryGroup(geometries=(cylinder2d, polyslab2d))
    clip2d_not_coplanar = td.ClipOperation(
        operation="union", geometry_a=cylinder2d, geometry_b=polyslab2d
    )

    # Geometries composed of sub geometries that do not have the same normal
    cylinder2d = td.Cylinder(axis=0, length=0, radius=1, center=(0, 0, 0.5))
    geo_group2d_not_aligned = td.GeometryGroup(geometries=(cylinder2d, polyslab2d))
    clip2d_not_aligned = td.ClipOperation(
        operation="union", geometry_a=cylinder2d, geometry_b=polyslab2d
    )

    not_allowed_geometries = [
        box3d,
        polyslab3d,
        cylinder3d,
        sphere,
        transformed_2d_bad,
        geo_group2d_not_coplanar,
        clip2d_not_coplanar,
        geo_group2d_not_aligned,
        clip2d_not_aligned,
    ]

    for geom in not_allowed_geometries:
        with pytest.raises(ValidationError):
            _ = td.Structure(geometry=geom, medium=med2d)


def test_from_permittivity_array():
    nx, ny, nz = 10, 1, 12
    box = td.Box(size=(2, td.inf, 4), center=(0, 0, 0))

    eps_data = 1.0 + np.random.random((nx, ny, nz))

    structure = td.Structure.from_permittivity_array(geometry=box, eps_data=eps_data, name="test")

    assert structure.name == "test"

    assert np.all(structure.medium.permittivity.values == eps_data)

    def f(x):
        eps_data = (1 + x) * (1 + np.random.random((nx, ny, nz)))
        structure = td.Structure.from_permittivity_array(
            geometry=box, eps_data=eps_data, name="test"
        )
        return anp.sum(structure.medium.permittivity).item()

    grad = ag.grad(f)(1.0)
    assert not np.isclose(grad, 0.0)


def test_to_gdstk_pixel_exact(tmp_path):
    box = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    nx, ny = 50, 50
    arr = np.ones((nx, ny, 1))
    arr[20:30, 20:30] = 2
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.asarray([0])
    coords = {"x": x, "y": y, "z": z}
    permittivity = td.SpatialDataArray(arr, coords=coords)

    medium = td.CustomMedium(permittivity=permittivity)
    structure = td.Structure(geometry=box, medium=medium)
    polygons = structure.to_gdstk(z=0, frequency=3e14, permittivity_threshold=1.5, pixel_exact=True)
    assert polygons

    fname = str(tmp_path / "structure-exact.gds")
    structure.to_gds_file(fname, z=0, permittivity_threshold=1.5, frequency=3e14, pixel_exact=True)
    cells = gdstk.read_gds(fname).cells
    cell = cells[0]
    assert np.allclose(cell.bounding_box(), ((-0.2, -0.2), (0.2, 0.2)), atol=0.01)
    assert gdstk.inside([(0.1, 0.9), (0.5, 2.5), (0, 0)], cell.polygons) == (False, False, True)


def test_to_gdstk_pixel_exact_supports_3d_custom_medium():
    box = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    nx, ny, nz = 41, 39, 5
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-0.6, 0.6, nz)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    eps_xyz = np.ones((nx, ny, nz), dtype=float)
    for iz, z_val in enumerate(z):
        radius = 0.15 + 0.25 * max(0.0, 1.0 - abs(z_val) / float(np.max(np.abs(z))))
        eps_xyz[:, :, iz] = np.where(xx**2 + yy**2 <= radius**2, 2.0, 1.0)

    f = np.array([td.C_0])
    eps_diagonal_data = td.ScalarFieldDataArray(
        eps_xyz[..., None], coords={"x": x, "y": y, "z": z, "f": f}
    )
    eps_dataset = td.PermittivityDataset(
        eps_xx=eps_diagonal_data,
        eps_yy=eps_diagonal_data,
        eps_zz=eps_diagonal_data,
    )
    structure = td.Structure(geometry=box, medium=td.CustomMedium(eps_dataset=eps_dataset))

    polygons = structure.to_gdstk(
        z=0.0,
        frequency=td.C_0,
        permittivity_threshold=1.5,
        pixel_exact=True,
    )

    assert polygons
    assert sum(poly.area() for poly in polygons) > 0.0


def test_to_gdstk_pixel_exact_respects_geometry_extent_override():
    box = td.Box(center=(0, 0, 0), size=(4, 4, 2))
    x = np.array([-0.5, 0.5], dtype=float)
    y = np.array([-0.5, 0.5], dtype=float)
    z = np.array([0.0], dtype=float)
    arr = np.full((len(x), len(y), len(z)), 2.0, dtype=float)
    permittivity = td.SpatialDataArray(arr, coords={"x": x, "y": y, "z": z})

    structure = td.Structure(geometry=box, medium=td.CustomMedium(permittivity=permittivity))
    polygons = structure.to_gdstk(
        z=0.0, frequency=3e14, permittivity_threshold=1.5, pixel_exact=True
    )

    assert polygons
    bboxes = np.asarray([poly.bounding_box() for poly in polygons], dtype=float)
    bbox = (tuple(np.min(bboxes[:, 0, :], axis=0)), tuple(np.max(bboxes[:, 1, :], axis=0)))
    assert np.allclose(bbox, ((-2.0, -2.0), (2.0, 2.0)))
    assert np.isclose(sum(poly.area() for poly in polygons), 16.0)


def test_to_gdstk_supports_mixed_component_interp_methods():
    box = td.Box(center=(0, 0, 0), size=(2, 2, 0))
    x = np.linspace(-1.0, 1.0, 9)
    y = np.linspace(-1.0, 1.0, 9)
    z = np.array([0.0], dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    eps = np.where(xx**2 + yy**2 <= 0.45**2, 2.0, 1.0).astype(float)[..., None]
    coords = {"x": x, "y": y, "z": z}
    medium = td.CustomAnisotropicMedium(
        xx=td.CustomMedium(
            permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="nearest"
        ),
        yy=td.CustomMedium(
            permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="linear"
        ),
        zz=td.CustomMedium(
            permittivity=td.SpatialDataArray(eps, coords=coords), interp_method="nearest"
        ),
    )
    structure = td.Structure(geometry=box, medium=medium)

    polygons = structure.to_gdstk(
        z=0.0,
        frequency=td.C_0,
        permittivity_threshold=1.5,
        pixel_exact=False,
    )

    assert polygons
