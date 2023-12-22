"""Tests tidy3d/components/data/dataset.py"""
import pytest
import numpy as np
import pydantic.v1 as pd
from matplotlib import pyplot as plt


np.random.seed(4)


@pytest.mark.parametrize("ds_name", ["test123", None])
def test_triangular_dataset(tmp_path, ds_name):

    import tidy3d as td
    from tidy3d.components.types import vtk
    from tidy3d.exceptions import DataError, Tidy3dImportError

    # basic create
    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=("index"),
        name=ds_name,
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    # test name redirect
    assert tri_grid.name == ds_name

    # wrong points dimensionality
    with pytest.raises(pd.ValidationError):

        tri_grid_points_bad = td.PointDataArray(
            np.random.random((4, 3)),
            coords=dict(index=np.arange(4), axis=np.arange(3)),
        )

        _ = td.TriangularGridDataset(
            normal_axis=0,
            normal_pos=10,
            points=tri_grid_points_bad,
            cells=tri_grid_cells,
            values=tri_grid_values,
        )

    # grid with degenerate cells
    tri_grid_cells_bad = td.CellDataArray(
        [[0, 1, 1], [1, 2, 3]],
        coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
    )

    _ = td.TriangularGridDataset(
        normal_axis=2,
        normal_pos=-3,
        points=tri_grid_points,
        cells=tri_grid_cells_bad,
        values=tri_grid_values,
    )

    # invalid cell connections
    with pytest.raises(pd.ValidationError):

        tri_grid_cells_bad = td.CellDataArray(
            [[0, 1, 2, 3]],
            coords=dict(cell_index=np.arange(1), vertex_index=np.arange(4)),
        )

        _ = td.TriangularGridDataset(
            normal_axis=2,
            normal_pos=-3,
            points=tri_grid_points,
            cells=tri_grid_cells_bad,
            values=tri_grid_values,
        )

    with pytest.raises(pd.ValidationError):

        tri_grid_cells_bad = td.CellDataArray(
            [[0, 1, 5], [1, 2, 3]],
            coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
        )

        _ = td.TriangularGridDataset(
            normal_axis=2,
            normal_pos=-3,
            points=tri_grid_points,
            cells=tri_grid_cells_bad,
            values=tri_grid_values,
        )

    # wrong number of values
    with pytest.raises(pd.ValidationError):

        tri_grid_values_bad = td.IndexedDataArray(
            [1.0, 2.0, 3.0],
            coords=dict(index=np.arange(3)),
        )

        _ = td.TriangularGridDataset(
            normal_axis=0,
            normal_pos=0,
            points=tri_grid_points,
            cells=tri_grid_cells,
            values=tri_grid_values_bad,
        )

    # some auxiliary properties
    assert tri_grid.bounds == ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    assert np.all(tri_grid._vtk_offsets == np.array([0, 3, 6]))

    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid._vtk_cells
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid._vtk_points
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid._vtk_obj
    else:
        _ = tri_grid._vtk_cells
        _ = tri_grid._vtk_points
        _ = tri_grid._vtk_obj

    # plane slicing
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid.plane_slice(axis=2, pos=0.5)
    else:
        result = tri_grid.plane_slice(axis=2, pos=0.5)

        assert result.name == ds_name

        # can't slice parallel grid plane
        with pytest.raises(DataError):
            _ = tri_grid.plane_slice(axis=1, pos=0.5)

        # can't slice outside of bounds
        with pytest.raises(DataError):
            _ = tri_grid.plane_slice(axis=0, pos=2)

    # clipping by a box
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
    else:
        result = tri_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
        assert result.name == ds_name

        # can't clip outside of grid
        with pytest.raises(DataError):
            _ = tri_grid.box_clip([[0.1, 0.1, 0.3], [0.2, 0.2, 0.9]])

    # interpolation
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            invariant = tri_grid.interp(
                x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333
            )
    else:
        # default = invariant along normal direction
        invariant = tri_grid.interp(x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333)
        assert np.all(invariant.isel(y=0).data == invariant.isel(y=1).data)
        assert invariant.name == ds_name

        # no invariance
        out_of_plane = tri_grid.interp(
            x=0.4, y=[1], z=np.linspace(0.2, 0.6, 10), fill_value=123, ignore_normal_pos=False
        )
        assert np.all(out_of_plane.data == 123)
        assert out_of_plane.name == ds_name

        # outside of grid
        invariant_no_intersection = tri_grid.interp(
            x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10), fill_value=909
        )
        assert np.all(invariant_no_intersection.data == 909)
        assert invariant_no_intersection.name == ds_name

    # renaming
    tri_grid_renamed = tri_grid.rename("renamed")
    assert tri_grid_renamed.name == "renamed"

    # plotting
    _ = tri_grid.plot()
    plt.close()

    _ = tri_grid.plot(grid=False)
    plt.close()

    _ = tri_grid.plot(field=False)
    plt.close()

    _ = tri_grid.plot(cbar=False)
    plt.close()

    _ = tri_grid.plot(vmin=-20, vmax=100)
    plt.close()

    _ = tri_grid.plot(cbar_kwargs=dict(label="test"))
    plt.close()

    _ = tri_grid.plot(cmap="RdBu")
    plt.close()

    _ = tri_grid.plot(shading="flat")
    plt.close()

    with pytest.raises(DataError):
        _ = tri_grid.plot(field=False, grid=False)

    # generalized selection method
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid.sel(x=0.2)
    else:
        _ = tri_grid.sel(x=0.2)
        _ = tri_grid.sel(x=0.2, z=[0.3, 0.4, 0.5])
        result = tri_grid.sel(x=np.linspace(0, 1, 3), y=tri_grid.normal_pos, z=[0.3, 0.4, 0.5])
        assert result.name == ds_name

        # can't select out of plane
        with pytest.raises(DataError):
            _ = tri_grid.sel(x=np.linspace(0, 1, 3), y=1.2, z=[0.3, 0.4, 0.5])

    # writing/reading .vtu
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            tri_grid.to_vtu(tmp_path / "tri_grid_test.vtu")
        with pytest.raises(Tidy3dImportError):
            tri_grid_loaded = td.TriangularGridDataset.from_vtu(tmp_path / "tri_grid_test.vtu")
    else:
        tri_grid.to_vtu(tmp_path / "tri_grid_test.vtu")
        tri_grid_loaded = td.TriangularGridDataset.from_vtu(tmp_path / "tri_grid_test.vtu")

        assert tri_grid == tri_grid_loaded

    # test ariphmetic operations
    def operation(arr):
        return 5 + (arr * 2 + arr.imag / 3) ** 2 / arr.real + np.log10(arr.abs)

    result = operation(tri_grid)
    result_values = operation(tri_grid.values)

    assert np.allclose(result.values, result_values)
    assert result.name == ds_name


@pytest.mark.parametrize("ds_name", ["test123", None])
def test_tetrahedral_dataset(tmp_path, ds_name):

    import tidy3d as td
    from tidy3d.components.types import vtk
    from tidy3d.exceptions import DataError, Tidy3dImportError

    # basic create
    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=("index"),
        name=ds_name,
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    # wrong points dimensionality
    with pytest.raises(pd.ValidationError):

        tet_grid_points_bad = td.PointDataArray(
            np.random.random((5, 2)),
            coords=dict(index=np.arange(5), axis=np.arange(2)),
        )

        _ = td.TetrahedralGridDataset(
            points=tet_grid_points_bad,
            cells=tet_grid_cells,
            values=tet_grid_values,
        )

    # grid with degenerate cells
    tet_grid_cells_bad = td.CellDataArray(
        [[0, 1, 1, 4], [1, 2, 3, 4]],
        coords=dict(cell_index=np.arange(2), vertex_index=np.arange(4)),
    )

    _ = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells_bad,
        values=tet_grid_values,
    )

    # invalid cell connections
    with pytest.raises(pd.ValidationError):

        tet_grid_cells_bad = td.CellDataArray(
            [[0, 1, 2], [1, 2, 3]],
            coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
        )

        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values,
        )

    with pytest.raises(pd.ValidationError):

        tet_grid_cells_bad = td.CellDataArray(
            [[0, 1, 2, 6], [1, 2, 3, 4]],
            coords=dict(cell_index=np.arange(2), vertex_index=np.arange(4)),
        )

        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values,
        )

    # wrong number of values
    with pytest.raises(pd.ValidationError):

        tet_grid_values_bad = td.IndexedDataArray(
            [1.0, 2.0, 3.0],
            coords=dict(index=np.arange(3)),
        )

        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values_bad,
        )

    # some auxiliary properties
    assert tet_grid.bounds == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert np.all(tet_grid._vtk_offsets == np.array([0, 4, 8]))
    assert tet_grid.name == ds_name

    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid._vtk_cells
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid._vtk_points
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid._vtk_obj
    else:
        _ = tet_grid._vtk_cells
        _ = tet_grid._vtk_points
        _ = tet_grid._vtk_obj

    # plane slicing
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.plane_slice(axis=2, pos=0.5)
    else:
        result = tet_grid.plane_slice(axis=2, pos=0.5)
        assert result.name == ds_name

        # can't slice outside of bounds
        with pytest.raises(DataError):
            _ = tet_grid.plane_slice(axis=1, pos=2)

    # clipping by a box
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
    else:
        result = tet_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
        assert result.name == ds_name

        # can't clip outside of grid
        with pytest.raises(DataError):
            _ = tet_grid.box_clip([[0.1, 1.1, 0.3], [0.2, 1.2, 0.9]])

    # interpolation
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.interp(x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333)
    else:
        # default = invariant along normal direction
        result = tet_grid.interp(x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333)
        assert result.name == ds_name

        # outside of grid
        no_intersection = tet_grid.interp(
            x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10), fill_value=909
        )
        assert np.all(no_intersection.data == 909)
        assert no_intersection.name == ds_name

    # generalized selection method
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.sel(x=0.2)
    else:
        _ = tet_grid.sel(x=0.2)
        _ = tet_grid.sel(x=0.2, y=0.4)
        result = tet_grid.sel(x=np.linspace(0, 1, 3), y=0.55, z=[0.3, 0.4, 0.5])
        assert result.name == ds_name

        # can't do plane slicing with array of values
        with pytest.raises(DataError):
            _ = tet_grid.sel(x=0.2, z=[0.3, 0.4, 0.5])

    # writing/reading .vtu
    if vtk["mod"] is None:
        with pytest.raises(Tidy3dImportError):
            tet_grid.to_vtu(tmp_path / "tet_grid_test.vtu")
        with pytest.raises(Tidy3dImportError):
            tet_grid_loaded = td.TetrahedralGridDataset.from_vtu(tmp_path / "tet_grid_test.vtu")
    else:
        tet_grid.to_vtu(tmp_path / "tet_grid_test.vtu")
        tet_grid_loaded = td.TetrahedralGridDataset.from_vtu(tmp_path / "tet_grid_test.vtu")

        assert tet_grid == tet_grid_loaded

    # test ariphmetic operations
    def operation(arr):
        return 5 + (arr * 2 + arr.imag / 3) ** 2 / arr.real + np.log10(arr.abs)

    result = operation(tet_grid)
    result_values = operation(tet_grid.values)

    assert np.allclose(result.values, result_values)
    assert result.name == ds_name
