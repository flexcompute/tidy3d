import pytest
import gdstk
import numpy as np
import tidy3d as td
import pydantic.v1 as pd


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
    eps_diagonal_data = td.ScalarFieldDataArray(data, coords=dict(x=x, y=y, z=z, f=f))
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

    eps_diagonal_data = td.ScalarFieldDataArray(data, coords=dict(x=x, y=y, z=z, f=f))
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
    with pytest.raises(pd.ValidationError):
        _ = td.Structure(geometry=geo2, medium=medium)

    geo3 = ps.rotated(np.pi / 5, j)
    with pytest.raises(pd.ValidationError):
        _ = td.Structure(geometry=geo3, medium=medium)

    geo4 = ps.rotated(np.pi / 6, (1, 1, 1))
    with pytest.raises(pd.ValidationError):
        _ = td.Structure(geometry=geo4, medium=medium)

    geo5 = td.GeometryGroup(geometries=[ps]).rotated(np.pi / 2, j)
    with pytest.raises(pd.ValidationError):
        _ = td.Structure(geometry=geo5, medium=medium)

    geo6 = td.GeometryGroup(geometries=[ps - box]).rotated(np.pi / 2, i)
    with pytest.raises(pd.ValidationError):
        _ = td.Structure(geometry=geo6, medium=medium)

    geo7 = td.GeometryGroup(geometries=[(ps - box).rotated(np.pi / 4, j)]).rotated(-np.pi / 4, j)
    _ = td.Structure(geometry=geo7, medium=medium)
