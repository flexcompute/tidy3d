"""Tests tidy3d/components/data/unstructured"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pydantic import ValidationError

from tidy3d.components.data.data_array import IndexedDataArrayTypes

from ..utils import AssertLogLevel, cartesian_to_unstructured

np.random.seed(4)

# Extract actual types from the Union for parametrization
INDEXED_DATA_ARRAY_TYPES = get_args(IndexedDataArrayTypes)

# Map each indexed data array type to its extra dimensions
TYPE_TO_EXTRA_DIMS = {}
import tidy3d as td

TYPE_TO_EXTRA_DIMS[td.IndexedDataArray] = {}
TYPE_TO_EXTRA_DIMS[td.IndexedVoltageDataArray] = {"voltage": [0, 1, 2]}
TYPE_TO_EXTRA_DIMS[td.IndexedTimeDataArray] = {"t": [0, 1, 2]}
TYPE_TO_EXTRA_DIMS[td.IndexedFreqDataArray] = {"f": [1e14, 2e14, 3e14]}
TYPE_TO_EXTRA_DIMS[td.IndexedFieldVoltageDataArray] = {"axis": [0, 1, 2], "voltage": [0, 1, 2]}
TYPE_TO_EXTRA_DIMS[td.IndexedSurfaceFreqDataArray] = {
    "side": ["outside", "inside"],
    "f": [1e14, 2e14, 3e14],
}
TYPE_TO_EXTRA_DIMS[td.IndexedSurfaceTimeDataArray] = {"side": ["outside", "inside"], "t": [0, 1, 2]}
TYPE_TO_EXTRA_DIMS[td.IndexedSurfaceFieldDataArray] = {
    "side": ["outside", "inside"],
    "axis": [0, 1, 2],
    "f": [1e14, 2e14, 3e14],
}
TYPE_TO_EXTRA_DIMS[td.IndexedSurfaceFieldTimeDataArray] = {
    "side": ["outside", "inside"],
    "axis": [0, 1, 2],
    "t": [0, 1, 2],
}
TYPE_TO_EXTRA_DIMS[td.IndexedFieldDataArray] = {
    "axis": [0, 1, 2],
    "f": [1e14, 2e14, 3e14],
}
TYPE_TO_EXTRA_DIMS[td.IndexedFieldTimeDataArray] = {
    "axis": [0, 1, 2],
    "t": [0, 1, 2],
}
TYPE_TO_EXTRA_DIMS[td.PointDataArray] = {"axis": [0, 1, 2]}


@pytest.mark.parametrize("values_type", TYPE_TO_EXTRA_DIMS.keys())
@pytest.mark.parametrize("ds_name", ["test123", None])
def test_triangular_dataset(tmp_path, ds_name, values_type, no_vtk=False):
    import tidy3d as td
    from tidy3d.exceptions import DataError, Tidy3dImportError

    extra_dims = TYPE_TO_EXTRA_DIMS[values_type]

    # basic create
    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = values_type(
        np.random.rand(4, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(4), **extra_dims),
        name=ds_name,
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )
    assert not tri_grid.is_uniform
    # test name redirect
    assert tri_grid.name == ds_name

    # wrong points dimensionality
    with pytest.raises(ValidationError):
        tri_grid_points_bad = td.PointDataArray(
            np.random.random((4, 3)),
            coords={"index": np.arange(4), "axis": np.arange(3)},
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
        coords={"cell_index": np.arange(2), "vertex_index": np.arange(3)},
    )

    with AssertLogLevel("WARNING"):
        tri_grid_with_degenerates = td.TriangularGridDataset(
            normal_axis=2,
            normal_pos=-3,
            points=tri_grid_points,
            cells=tri_grid_cells_bad,
            values=tri_grid_values,
        )

    # removal of degenerate cells

    # only removing degenerate cells will result in unsude points in this case
    with AssertLogLevel("WARNING"):
        tri_grid_with_fixed = tri_grid_with_degenerates.clean(
            remove_degenerate_cells=True, remove_unused_points=False
        )
    assert np.all(tri_grid_with_fixed.cells.values == [[1, 2, 3]])

    # once we remove those, no warning should occur
    with AssertLogLevel(None):
        tri_grid_with_fixed = tri_grid_with_fixed.clean(
            remove_degenerate_cells=False, remove_unused_points=True
        )
    assert np.all(tri_grid_with_fixed.cells.values == [[0, 1, 2]])

    # doing both at the same time
    with AssertLogLevel(None):
        tri_grid_with_fixed = tri_grid_with_degenerates.clean()
    assert np.all(tri_grid_with_fixed.cells.values == [[0, 1, 2]])

    # invalid cell connections
    tri_grid_cells_bad = td.CellDataArray(
        [[0, 1, 2, 3]],
        coords={"cell_index": np.arange(1), "vertex_index": np.arange(4)},
    )
    with pytest.raises(ValidationError):
        _ = td.TriangularGridDataset(
            normal_axis=2,
            normal_pos=-3,
            points=tri_grid_points,
            cells=tri_grid_cells_bad,
            values=tri_grid_values,
        )

    tri_grid_cells_bad = td.CellDataArray(
        [[0, 1, 5], [1, 2, 3]],
        coords={"cell_index": np.arange(2), "vertex_index": np.arange(3)},
    )
    with pytest.raises(ValidationError):
        _ = td.TriangularGridDataset(
            normal_axis=2,
            normal_pos=-3,
            points=tri_grid_points,
            cells=tri_grid_cells_bad,
            values=tri_grid_values,
        )

    # wrong number of values
    tri_grid_values_bad = values_type(
        np.random.rand(3, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(3), **extra_dims),
    )
    with pytest.raises(ValidationError):
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

    if no_vtk:
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
    if no_vtk:
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

        # slicing along edges
        _ = tri_grid.plane_slice(axis=0, pos=1)
        _ = tri_grid.plane_slice(axis=0, pos=0)
        _ = tri_grid.plane_slice(axis=2, pos=1)
        _ = tri_grid.plane_slice(axis=2, pos=0)

    # clipping by a box
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
    else:
        result = tri_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
        assert result.name == ds_name

        # can't clip outside of grid
        with pytest.raises(DataError):
            _ = tri_grid.box_clip([[0.1, 0.1, 0.3], [0.2, 0.2, 0.9]])

    # interpolation
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            tri_grid.interp(
                x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
            )
    else:
        interp = tri_grid.interp(x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333)
        assert np.all(interp.isel(y=0).data == interp.isel(y=1).data)
        assert interp.name == ds_name

        if len(extra_dims) == 0:
            interp_vtk = tri_grid.interp(
                x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
            )
            assert np.all(interp_vtk.isel(y=0).data == interp_vtk.isel(y=1).data)
            assert interp_vtk.name == ds_name
            assert np.allclose(interp_vtk, interp)
        else:
            with pytest.raises(DataError):
                interp_vtk = tri_grid.interp(
                    x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
                )

        # outside of grid
        no_intersection = tri_grid.interp(
            x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10), fill_value=909
        )
        assert np.all(no_intersection.data == 909)
        assert no_intersection.name == ds_name

        # test default fill_value="extrapolate" - should use nearest neighbor extrapolation
        # for points outside the grid (not zeros)
        extrapolated = tri_grid.interp(x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10))
        assert not np.all(extrapolated.data == 0)  # should extrapolate, not fill with zeros

        # fill_value=None should behave the same as fill_value="extrapolate"
        extrapolated_none = tri_grid.interp(
            x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10), fill_value=None
        )
        assert np.allclose(extrapolated.data, extrapolated_none.data)

    # renaming
    tri_grid_renamed = tri_grid.rename("renamed")
    assert tri_grid_renamed.name == "renamed"

    if len(extra_dims) > 0:
        with pytest.raises(DataError):
            _ = tri_grid.plot()

        tri_grid_one_field = tri_grid.sel(**{key: value[0] for key, value in extra_dims.items()})
    else:
        tri_grid_one_field = tri_grid

    # plotting
    _ = tri_grid_one_field.plot()
    plt.close()

    _ = tri_grid_one_field.plot(grid=False)
    plt.close()

    _ = tri_grid_one_field.plot(field=False)
    plt.close()

    _ = tri_grid_one_field.plot(cbar=False)
    plt.close()

    _ = tri_grid_one_field.plot(vmin=-20, vmax=100)
    plt.close()

    _ = tri_grid_one_field.plot(cbar_kwargs={"label": "test"})
    plt.close()

    _ = tri_grid_one_field.plot(cmap="RdBu")
    plt.close()

    _ = tri_grid_one_field.plot(shading="flat")
    plt.close()

    # test max_cells decimation
    if not no_vtk:
        _ = tri_grid_one_field.plot(max_cells=1)
        plt.close()

        # max_cells larger than mesh should be a no-op
        _ = tri_grid_one_field.plot(max_cells=10_000)
        plt.close()

        # grid-only decimation must not warn about missing point data
        with AssertLogLevel("INFO"):
            _ = tri_grid_one_field.plot(field=False, grid=True, max_cells=1)
        plt.close()

    # max_cells <= 0 must be rejected
    with pytest.raises(DataError):
        _ = tri_grid_one_field.plot(max_cells=0)
    with pytest.raises(DataError):
        _ = tri_grid_one_field.plot(max_cells=-1)

    # multi-field guard fires before decimation
    if len(extra_dims) > 0 and not no_vtk:
        with pytest.raises(DataError):
            _ = tri_grid.plot(max_cells=1)

    with pytest.raises(DataError):
        _ = tri_grid.plot(field=False, grid=False)

    # generalized selection method
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = tri_grid.sel(x=0.2)
    else:
        _ = tri_grid.sel(x=0.2)
        _ = tri_grid.sel(x=0.2, z=[0.3, 0.4, 0.5])
        result = tri_grid.sel(x=np.linspace(0, 1, 3), y=tri_grid.normal_pos, z=[0.3, 0.4, 0.5])
        assert result.name == ds_name

        # selecting only along the normal axis is a no-op and should return self
        result_normal_only = tri_grid.sel(y=tri_grid.normal_pos)
        assert result_normal_only == tri_grid

        # can't select out of plane
        with pytest.raises(DataError):
            _ = tri_grid.sel(x=np.linspace(0, 1, 3), y=1.2, z=[0.3, 0.4, 0.5])

    # writing/reading
    tri_grid.to_file(tmp_path / "tri_grid_test.hdf5")

    tri_grid_loaded = td.TriangularGridDataset.from_file(tmp_path / "tri_grid_test.hdf5")
    assert tri_grid == tri_grid_loaded

    # writing/reading .vtu
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            tri_grid.to_vtu(tmp_path / "tri_grid_test.vtu")
        with pytest.raises(Tidy3dImportError):
            tri_grid_loaded = td.TriangularGridDataset.from_vtu(tmp_path / "tri_grid_test.vtu")
    else:
        tri_grid.to_vtu(tmp_path / "tri_grid_test.vtu")

        if len(extra_dims) == 0:
            tri_grid_loaded = td.TriangularGridDataset.from_vtu(tmp_path / "tri_grid_test.vtu")
            assert tri_grid == tri_grid_loaded

            custom_name = "newname"
            tri_grid_renamed = tri_grid.rename(custom_name)
            tri_grid_renamed.to_vtu(tmp_path / "tri_grid_test.vtu")

            tri_grid_loaded = td.TriangularGridDataset.from_vtu(
                tmp_path / "tri_grid_test.vtu", field=custom_name
            )
            assert tri_grid == tri_grid_loaded

        with pytest.raises(AttributeError):
            tri_grid_loaded = td.TriangularGridDataset.from_vtu(
                tmp_path / "tri_grid_test.vtu", field="blah"
            )

    # test ariphmetic operations
    def operation(arr):
        return 5 + (arr * 2 + arr.imag / 3) ** 2 / arr.real + np.log10(arr.abs)

    result = operation(tri_grid)
    result_values = operation(tri_grid.values)

    assert np.allclose(result.values, result_values)
    assert result.name == ds_name

    # test conjugate
    assert np.allclose(tri_grid.conj().values, np.conjugate(tri_grid.values))

    # test norm
    if "axis" in extra_dims:
        axis_index = 1 + list(extra_dims.keys()).index("axis")
        assert np.allclose(
            tri_grid.norm(dim="axis").values, np.linalg.norm(tri_grid.values, axis=axis_index)
        )


@pytest.mark.parametrize("values_type", TYPE_TO_EXTRA_DIMS.keys())
@pytest.mark.parametrize("ds_name", ["test123", None])
def test_tetrahedral_dataset(tmp_path, ds_name, values_type, no_vtk=False):
    import tidy3d as td
    from tidy3d.exceptions import DataError, Tidy3dImportError

    extra_dims = TYPE_TO_EXTRA_DIMS[values_type]

    # basic create
    tet_grid_points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 3, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7], [0, 4, 5, 7], [0, 1, 5, 7]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = values_type(
        np.random.rand(8, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(8), **extra_dims),
        name=ds_name,
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    # wrong points dimensionality
    tet_grid_points_bad = td.PointDataArray(
        np.random.random((8, 2)),
        coords={"index": np.arange(8), "axis": np.arange(2)},
    )
    with pytest.raises(ValidationError):
        _ = td.TetrahedralGridDataset(
            points=tet_grid_points_bad,
            cells=tet_grid_cells,
            values=tet_grid_values,
        )

    # grid with degenerate cells
    tet_grid_cells_bad = td.CellDataArray(
        [[0, 1, 1, 7], [0, 2, 3, 7], [0, 2, 2, 7], [0, 4, 6, 7], [0, 4, 5, 7], [0, 5, 5, 7]],
        coords={"cell_index": np.arange(6), "vertex_index": np.arange(4)},
    )

    with AssertLogLevel("WARNING"):
        tet_grid_with_degenerates = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values,
        )

    # removal of degenerate cells

    # only removing degenerate cells will result in unsude points in this case
    with AssertLogLevel("WARNING"):
        tet_grid_with_fixed = tet_grid_with_degenerates.clean(
            remove_degenerate_cells=True, remove_unused_points=False
        )
    assert np.all(tet_grid_with_fixed.cells.values == [[0, 2, 3, 7], [0, 4, 6, 7], [0, 4, 5, 7]])

    # once we remove those, no warning should occur
    with AssertLogLevel(None):
        tet_grid_with_fixed = tet_grid_with_fixed.clean(
            remove_degenerate_cells=False, remove_unused_points=True
        )
    assert np.all(tet_grid_with_fixed.cells.values == [[0, 1, 2, 6], [0, 3, 5, 6], [0, 3, 4, 6]])

    # doing both at the same time
    with AssertLogLevel(None):
        tet_grid_with_fixed = tet_grid_with_degenerates.clean()
    assert np.all(tet_grid_with_fixed.cells.values == [[0, 1, 2, 6], [0, 3, 5, 6], [0, 3, 4, 6]])

    # invalid cell connections
    tet_grid_cells_bad = td.CellDataArray(
        [[0, 1, 3], [0, 2, 3], [0, 2, 6], [0, 4, 6], [0, 4, 5], [0, 1, 5]],
        coords={"cell_index": np.arange(6), "vertex_index": np.arange(3)},
    )
    with pytest.raises(ValidationError):
        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values,
        )

    tet_grid_cells_bad = td.CellDataArray(
        [[0, 1, 3, 17], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7], [0, 4, 5, 7], [0, 1, 5, 7]],
        coords={"cell_index": np.arange(6), "vertex_index": np.arange(4)},
    )
    with pytest.raises(ValidationError):
        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values,
        )

    # wrong number of values
    tet_grid_values_bad = values_type(
        np.random.rand(5, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(5), **extra_dims),
    )
    with pytest.raises(ValidationError):
        _ = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells_bad,
            values=tet_grid_values_bad,
        )

    # some auxiliary properties
    assert tet_grid.bounds == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert np.all(tet_grid._vtk_offsets == np.array([0, 4, 8, 12, 16, 20, 24]))
    assert tet_grid.name == ds_name

    if no_vtk:
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
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.plane_slice(axis=2, pos=0.5)
    else:
        result = tet_grid.plane_slice(axis=2, pos=0.5)
        assert result.name == ds_name

        # can't slice outside of bounds
        with pytest.raises(DataError):
            _ = tet_grid.plane_slice(axis=1, pos=2)

        # slicing along faces
        for axis in range(3):
            for pos in [0, 1]:
                _ = tet_grid.plane_slice(axis=axis, pos=pos)

        # slicing along edges
        for axis in range(3):
            for pos1 in [0, 0.4, 1]:
                for pos2 in [0, 0.7, 1]:
                    pos = [pos1, pos2]
                    pos.insert(axis, 0)

                    _ = tet_grid.line_slice(axis=axis, pos=pos)

    # clipping by a box
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
    else:
        result = tet_grid.box_clip([[0.1, -0.2, 0.1], [0.2, 0.2, 0.9]])
        assert result.name == ds_name

        # can't clip outside of grid
        with pytest.raises(DataError):
            _ = tet_grid.box_clip([[0.1, 1.1, 0.3], [0.2, 1.2, 0.9]])

    # interpolation
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = tet_grid.interp(
                x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
            )
    else:
        result = tet_grid.interp(x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333)
        assert result.name == ds_name

        if len(extra_dims) == 0:
            result_vtk = tet_grid.interp(
                x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
            )
            assert result.name == ds_name
            assert np.allclose(result_vtk, result)
        else:
            with pytest.raises(DataError):
                result_vtk = tet_grid.interp(
                    x=0.4, y=[0, 1], z=np.linspace(0.2, 0.6, 10), fill_value=-333, use_vtk=True
                )

        # outside of grid
        no_intersection = tet_grid.interp(
            x=[1.5, 2], y=2, z=np.linspace(0.2, 0.6, 10), fill_value=909
        )
        assert np.all(no_intersection.data == 909)
        assert no_intersection.name == ds_name

    # generalized selection method
    if no_vtk:
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

    # writing/reading
    tet_grid.to_file(tmp_path / "tri_grid_test.hdf5")

    tet_grid_loaded = td.TetrahedralGridDataset.from_file(tmp_path / "tri_grid_test.hdf5")
    assert tet_grid == tet_grid_loaded

    # writing/reading .vtu
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            tet_grid.to_vtu(tmp_path / "tet_grid_test.vtu")
        with pytest.raises(Tidy3dImportError):
            tet_grid_loaded = td.TetrahedralGridDataset.from_vtu(tmp_path / "tet_grid_test.vtu")
    else:
        tet_grid.to_vtu(tmp_path / "tet_grid_test.vtu")

        if len(extra_dims) == 0:
            tet_grid_loaded = td.TetrahedralGridDataset.from_vtu(tmp_path / "tet_grid_test.vtu")
            assert tet_grid == tet_grid_loaded

            custom_name = "newname"
            tet_grid_renamed = tet_grid.rename(custom_name)
            tet_grid_renamed.to_vtu(tmp_path / "tet_grid_test.vtu")

            tet_grid_loaded = td.TetrahedralGridDataset.from_vtu(
                tmp_path / "tet_grid_test.vtu", field=custom_name
            )
            assert tet_grid == tet_grid_loaded

        with pytest.raises(AttributeError):
            td.TetrahedralGridDataset.from_vtu(tmp_path / "tet_grid_test.vtu", field="blah")

    # plotting
    if not no_vtk:
        from tidy3d.exceptions import DataError as _DataError

        if len(extra_dims) > 0:
            tet_grid_one_field = tet_grid.sel(
                **{key: value[0] for key, value in extra_dims.items()}
            )
        else:
            tet_grid_one_field = tet_grid

        _ = tet_grid_one_field.plot(z=0.5)
        plt.close()

        _ = tet_grid_one_field.plot(x=0.5, grid=False)
        plt.close()

        _ = tet_grid_one_field.plot(y=0.5, cmap="hot", vmin=-1, vmax=2)
        plt.close()

        _ = tet_grid_one_field.plot(z=0.5, cbar_kwargs={"label": "test"})
        plt.close()

        _ = tet_grid_one_field.plot(z=0.5, cbar=False, shading="flat")
        plt.close()

        # test max_cells decimation pass-through
        _ = tet_grid_one_field.plot(z=0.5, max_cells=1, grid=False)
        plt.close()

        # verify axis limits are clipped to tet bounds
        ax = tet_grid_one_field.plot(x=0.5)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        tet_bounds = tet_grid_one_field.bounds
        assert np.isclose(xlim[0], tet_bounds[0][1])
        assert np.isclose(xlim[1], tet_bounds[1][1])
        assert np.isclose(ylim[0], tet_bounds[0][2])
        assert np.isclose(ylim[1], tet_bounds[1][2])
        plt.close()

        # must provide exactly one spatial kwarg
        with pytest.raises(_DataError):
            tet_grid_one_field.plot()
        with pytest.raises(_DataError):
            tet_grid_one_field.plot(x=0.5, y=0.5)

    # test ariphmetic operations
    def operation(arr):
        return 5 + (arr * 2 + arr.imag / 3) ** 2 / arr.real + np.log10(arr.abs)

    result = operation(tet_grid)
    result_values = operation(tet_grid.values)

    assert np.allclose(result.values, result_values)
    assert result.name == ds_name

    # test conjugate
    assert np.allclose(tet_grid.conj().values, np.conjugate(tet_grid.values))

    # test norm
    if "axis" in extra_dims:
        axis_index = 1 + list(extra_dims.keys()).index("axis")
        assert np.allclose(
            tet_grid.norm(dim="axis").values, np.linalg.norm(tet_grid.values, axis=axis_index)
        )


@pytest.mark.parametrize("values_type", TYPE_TO_EXTRA_DIMS.keys())
@pytest.mark.parametrize("ds_name", ["test123", None])
def test_triangular_surface_dataset(tmp_path, ds_name, values_type, no_vtk=False):
    import tidy3d as td
    from tidy3d.exceptions import DataError, Tidy3dImportError

    extra_dims = TYPE_TO_EXTRA_DIMS[values_type]

    # basic create
    surf_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dims=("index", "axis"),
    )

    surf_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    surf_grid_values = values_type(
        np.random.rand(4, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(4), **extra_dims),
        name=ds_name,
    )

    surf_grid = td.TriangularSurfaceDataset(
        points=surf_grid_points,
        cells=surf_grid_cells,
        values=surf_grid_values,
    )
    assert not surf_grid.is_uniform
    # test name redirect
    assert surf_grid.name == ds_name

    # wrong points dimensionality (should be 3D, not 2D)
    with pytest.raises(ValidationError):
        surf_grid_points_bad = td.PointDataArray(
            np.random.random((4, 2)),
            coords={"index": np.arange(4), "axis": np.arange(2)},
        )

        _ = td.TriangularSurfaceDataset(
            points=surf_grid_points_bad,
            cells=surf_grid_cells,
            values=surf_grid_values,
        )

    # grid with degenerate cells
    surf_grid_cells_bad = td.CellDataArray(
        [[0, 1, 1], [1, 2, 3]],
        coords={"cell_index": np.arange(2), "vertex_index": np.arange(3)},
    )

    with AssertLogLevel("WARNING"):
        surf_grid_with_degenerates = td.TriangularSurfaceDataset(
            points=surf_grid_points,
            cells=surf_grid_cells_bad,
            values=surf_grid_values,
        )

    # removal of degenerate cells

    # only removing degenerate cells will result in unused points in this case
    with AssertLogLevel("WARNING"):
        surf_grid_with_fixed = surf_grid_with_degenerates.clean(
            remove_degenerate_cells=True, remove_unused_points=False
        )
    assert np.all(surf_grid_with_fixed.cells.values == [[1, 2, 3]])

    # once we remove those, no warning should occur
    with AssertLogLevel(None):
        surf_grid_with_fixed = surf_grid_with_fixed.clean(
            remove_degenerate_cells=False, remove_unused_points=True
        )
    assert np.all(surf_grid_with_fixed.cells.values == [[0, 1, 2]])

    # doing both at the same time
    with AssertLogLevel(None):
        surf_grid_with_fixed = surf_grid_with_degenerates.clean()
    assert np.all(surf_grid_with_fixed.cells.values == [[0, 1, 2]])

    # invalid cell connections
    surf_grid_cells_bad = td.CellDataArray(
        [[0, 1, 2, 3]],
        coords={"cell_index": np.arange(1), "vertex_index": np.arange(4)},
    )
    with pytest.raises(ValidationError):
        _ = td.TriangularSurfaceDataset(
            points=surf_grid_points,
            cells=surf_grid_cells_bad,
            values=surf_grid_values,
        )

    surf_grid_cells_bad = td.CellDataArray(
        [[0, 1, 5], [1, 2, 3]],
        coords={"cell_index": np.arange(2), "vertex_index": np.arange(3)},
    )
    with pytest.raises(ValidationError):
        _ = td.TriangularSurfaceDataset(
            points=surf_grid_points,
            cells=surf_grid_cells_bad,
            values=surf_grid_values,
        )

    # wrong number of values
    surf_grid_values_bad = values_type(
        np.random.rand(3, *[len(coord) for coord in extra_dims.values()]),
        coords=dict(index=np.arange(3), **extra_dims),
    )
    with pytest.raises(ValidationError):
        _ = td.TriangularSurfaceDataset(
            points=surf_grid_points,
            cells=surf_grid_cells,
            values=surf_grid_values_bad,
        )

    # some auxiliary properties
    assert surf_grid.bounds == ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    assert np.all(surf_grid._vtk_offsets == np.array([0, 3, 6]))

    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = surf_grid._vtk_cells
        with pytest.raises(Tidy3dImportError):
            _ = surf_grid._vtk_points
        with pytest.raises(Tidy3dImportError):
            _ = surf_grid._vtk_obj
    else:
        _ = surf_grid._vtk_cells
        _ = surf_grid._vtk_points
        _ = surf_grid._vtk_obj

    # plane slicing - not supported for surface datasets
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = surf_grid.plane_slice(axis=2, pos=0.5)
    else:
        with pytest.raises(td.exceptions.Tidy3dNotImplementedError):
            _ = surf_grid.plane_slice(axis=2, pos=0.5)

    # clipping by a box
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            _ = surf_grid.box_clip([[0.1, 0.1, -0.1], [0.9, 0.9, 0.1]])
    else:
        result = surf_grid.box_clip([[0.1, 0.1, -0.1], [0.9, 0.9, 0.1]])
        assert result.name == ds_name

        # can't clip outside of grid
        with pytest.raises(DataError):
            _ = surf_grid.box_clip([[0.1, 0.1, 0.5], [0.9, 0.9, 1.0]])

    # renaming
    surf_grid_renamed = surf_grid.rename("renamed")
    assert surf_grid_renamed.name == "renamed"

    # generalized selection method - spatial selection not supported for surface
    with pytest.raises(td.exceptions.Tidy3dNotImplementedError):
        _ = surf_grid.sel(x=0.5)

    # non-spatial selection should work
    if len(extra_dims) > 0:
        first_key = list(extra_dims.keys())[0]
        first_value = extra_dims[first_key][0]
        result = surf_grid.sel(**{first_key: first_value})
        assert result.name == ds_name
        result_isel = surf_grid.isel(**{first_key: 0})
        assert result == result_isel

    # cannot sel along index dimension
    with pytest.raises(DataError):
        _ = surf_grid.sel(index=0)
    with pytest.raises(DataError):
        _ = surf_grid.isel(index=0)

    # writing/reading
    surf_grid.to_file(tmp_path / "surf_grid_test.hdf5")

    surf_grid_loaded = td.TriangularSurfaceDataset.from_file(tmp_path / "surf_grid_test.hdf5")
    assert surf_grid == surf_grid_loaded

    # writing/reading .vtu
    if no_vtk:
        with pytest.raises(Tidy3dImportError):
            surf_grid.to_vtu(tmp_path / "surf_grid_test.vtu")
        with pytest.raises(Tidy3dImportError):
            surf_grid_loaded = td.TriangularSurfaceDataset.from_vtu(tmp_path / "surf_grid_test.vtu")
    else:
        surf_grid.to_vtu(tmp_path / "surf_grid_test.vtu")

        if len(extra_dims) == 0:
            surf_grid_loaded = td.TriangularSurfaceDataset.from_vtu(tmp_path / "surf_grid_test.vtu")
            assert surf_grid == surf_grid_loaded

            custom_name = "newname"
            surf_grid_renamed = surf_grid.rename(custom_name)
            surf_grid_renamed.to_vtu(tmp_path / "surf_grid_test.vtu")

            surf_grid_loaded = td.TriangularSurfaceDataset.from_vtu(
                tmp_path / "surf_grid_test.vtu", field=custom_name
            )
            assert surf_grid == surf_grid_loaded

        with pytest.raises(AttributeError):
            surf_grid_loaded = td.TriangularSurfaceDataset.from_vtu(
                tmp_path / "surf_grid_test.vtu", field="blah"
            )

    # test cell volumes (areas for surface)
    cell_volumes = surf_grid.get_cell_volumes()
    assert len(cell_volumes) == len(surf_grid.cells)
    assert np.all(cell_volumes > 0)

    # test arithmetic operations
    def operation(arr):
        return 5 + (arr * 2 + arr.imag / 3) ** 2 / arr.real + np.log10(arr.abs)

    result = operation(surf_grid)
    result_values = operation(surf_grid.values)

    assert np.allclose(result.values, result_values)
    assert result.name == ds_name

    # test conjugate
    assert np.allclose(surf_grid.conj().values, np.conjugate(surf_grid.values))

    # test norm
    if "axis" in extra_dims:
        axis_index = 1 + list(extra_dims.keys()).index("axis")
        assert np.allclose(
            surf_grid.norm(dim="axis").values, np.linalg.norm(surf_grid.values, axis=axis_index)
        )

    # test non-spatial interpolation works
    if len(extra_dims) > 0:
        first_key = list(extra_dims.keys())[0]
        first_coords = extra_dims[first_key]
        # interpolate to midpoint
        if isinstance(first_coords[0], str):
            # Can't interpolate string coordinates
            pass
        else:
            mid_value = (first_coords[0] + first_coords[1]) / 2
            result = surf_grid.interp(**{first_key: [mid_value]})
            assert result.name == ds_name
            assert len(result.values.coords[first_key]) == 1

    # test spatial interpolation raises error (not implemented for surfaces)
    with pytest.raises(td.exceptions.Tidy3dNotImplementedError):
        _ = surf_grid.interp(x=0.5, y=0.5, z=0.5)


def test_reflect_all_points_on_plane():
    """When every boundary point lies on the reflection plane, reflect should
    return the mesh unchanged (no duplicated cells or points)."""
    import tidy3d as td

    # flat surface at z=0: all points on the plane
    points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dims=("index", "axis"),
    )
    cells = td.CellDataArray(
        [[0, 1, 2], [1, 3, 2]],
        dims=("cell_index", "vertex_index"),
    )
    values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0],
        coords={"index": np.arange(4)},
    )
    surf = td.TriangularSurfaceDataset(points=points, cells=cells, values=values)

    reflected = surf.reflect(axis=2, center=0.0)

    assert len(reflected.cells) == len(surf.cells)
    assert len(reflected.points) == len(surf.points)
    assert np.allclose(reflected.values.data, surf.values.data)

    # cell areas must not change
    orig_areas = surf.get_cell_volumes()
    refl_areas = reflected.get_cell_volumes()
    assert np.allclose(orig_areas, refl_areas)


def test_reflect_mixed_on_and_off_plane():
    """When some cells lie entirely on the reflection plane and others do not,
    only the off-plane cells should be reflected; on-plane cells must not be
    duplicated."""
    import tidy3d as td

    # two triangles: one at z=0 (on-plane), one tilted up
    points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],  # 0 - on plane
            [1.0, 0.0, 0.0],  # 1 - on plane
            [0.0, 1.0, 0.0],  # 2 - on plane
            [1.0, 1.0, 0.5],  # 3 - off plane
        ],
        dims=("index", "axis"),
    )
    cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )
    values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0],
        coords={"index": np.arange(4)},
    )
    surf = td.TriangularSurfaceDataset(points=points, cells=cells, values=values)

    reflected = surf.reflect(axis=2, center=0.0)

    # original 2 cells + only the off-plane cell reflected = 3 total
    assert len(reflected.cells) == 3

    # only point 3 is off-plane → 4 original + 1 reflected = 5
    assert len(reflected.points) == 5
    assert len(reflected.values) == 5

    # the reflected copy of point 3 should be at z = -0.5
    reflected_z = reflected.points.sel(axis=2).data
    assert np.isclose(reflected_z[4], -0.5)


@pytest.mark.parametrize("fill_value", [0.23123, "extrapolate"])
@pytest.mark.parametrize("use_vtk", [True, False])
@pytest.mark.parametrize("nz", [13, 1])
def test_cartesian_to_unstructured(nz, use_vtk, fill_value):
    import tidy3d as td

    nx = 11
    ny = 12

    x = np.linspace(0, 0.3, nx)
    y = np.linspace(-0.4, 0, ny)
    z = np.linspace(-0.2, 0.15, nz)
    values = np.sin(x[:, None, None]) * np.cos(y[None, :, None]) * np.exp(z[None, None, :])

    arr_c = td.SpatialDataArray(values, coords={"x": x, "y": y, "z": z})

    arr_u_linear = cartesian_to_unstructured(arr_c, pert=0.1, method="linear", seed=123)
    arr_c_linear = arr_u_linear.interp(
        x=x, y=y, z=z, method="linear", use_vtk=use_vtk, fill_value=fill_value
    )

    print(np.max(np.abs(arr_c.values - arr_c_linear.values)))
    assert np.allclose(arr_c.values, arr_c_linear.values, atol=1e-4, rtol=1e-4)

    arr_u_nearest = cartesian_to_unstructured(arr_c, pert=0.1, method="nearest", seed=123)
    arr_c_nearest = arr_u_nearest.interp(
        x=x, y=y, z=z, method="nearest", use_vtk=use_vtk, fill_value=fill_value
    )

    assert np.all(arr_c.values == arr_c_nearest.values)

    sample_outside = arr_u_linear.interp(
        x=-1, y=-1, z=-1, method="linear", use_vtk=use_vtk, fill_value=fill_value
    )

    if fill_value == "extrapolate":
        assert sample_outside.values.item() == values[0, 0, 0]
    else:
        assert sample_outside.values.item() == fill_value


def test_triangular_dataset_uniform():
    import tidy3d as td

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
        [1.0, 1.0, 1.0, 1.0],
        dims=("index"),
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )
    assert tri_grid.is_uniform

    tri_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 1.0],
        dims=("index"),
    )

    tri_grid = tri_grid.updated_copy(values=tri_grid_values)
    assert not tri_grid.is_uniform


def test_cell_values():
    """Test whether the cell values are correctly calculated"""
    import tidy3d as td

    # start with a triangle grid
    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = td.IndexedVoltageDataArray(
        [[0.0, 0.0], [0, 0], [3, -3], [3, -3]],
        coords={"index": np.arange(4), "voltage": [-1, 1]},
        name="test",
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    cell_values = tri_grid.get_cell_values(voltage=-1)
    cell_vols = tri_grid.get_cell_volumes()
    assert np.dot(cell_values, cell_vols) == 1.5

    # Now repeat for a tet mesh
    tet_grid_points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 3, 7], [0, 2, 7, 3], [0, 2, 6, 7], [0, 4, 7, 6], [0, 4, 5, 7], [0, 1, 7, 5]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.IndexedDataArray(
        [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0], coords={"index": np.arange(8)}, name="test_tet"
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    # this will fail since we now have a single field (voltage isn't a coordinate)
    with pytest.raises(KeyError):
        _ = tet_grid.get_cell_values(voltage=1)

    cell_values = tet_grid.get_cell_values()
    cell_vols = tet_grid.get_cell_volumes()
    assert np.dot(cell_values, cell_vols) == 1.5


def test_from_vtk():
    """Test that 2D and 3D vtk data can be loaded if `ignore_invalid_cells==True`."""
    import tidy3d as td
    from tidy3d.exceptions import DataError

    _ = td.TetrahedralGridDataset.from_vtk("tests/data/gmsh.vtk", ignore_invalid_cells=True)

    with pytest.raises(DataError):
        _ = td.TetrahedralGridDataset.from_vtk("tests/data/gmsh.vtk")

    _ = td.TriangularGridDataset.from_vtk("tests/data/gmsh_2d.vtk", ignore_invalid_cells=True)

    with pytest.raises(DataError):
        _ = td.TriangularGridDataset.from_vtk("tests/data/gmsh_2d.vtk")


def test_triangular_from_vtu_near_planar_large_coordinates(tmp_path):
    """Regression for large-coordinate planar slices with small normal-axis jitter."""
    pytest.importorskip("vtk")
    import vtk

    import tidy3d as td

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0.0,
        points=td.PointDataArray(
            [
                [100000.0, 0.0],
                [101000.0, 0.0],
                [100000.0, 690.0],
                [101000.0, 690.0],
            ],
            dims=("index", "axis"),
        ),
        cells=td.CellDataArray([[0, 1, 2], [1, 2, 3]], dims=("cell_index", "vertex_index")),
        values=td.IndexedDataArray(
            [1.0, 2.0, 3.0, 4.0], coords={"index": np.arange(4)}, name="temp"
        ),
    )

    vtu_path = tmp_path / "near_planar_large_coords.vtu"
    tri_grid.to_vtu(vtu_path)

    # Inject tiny jitter in the nominally planar normal direction.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    grid = reader.GetOutput()
    points = grid.GetPoints()
    for ind in range(points.GetNumberOfPoints()):
        x, _, z = points.GetPoint(ind)
        points.SetPoint(ind, x, 6e-6 if (ind % 2) == 0 else -6e-6, z)
    points.Modified()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(vtu_path))
    writer.SetInputData(grid)
    writer.Write()

    loaded = td.TriangularGridDataset.from_vtu(vtu_path, field="temp")
    assert loaded.normal_axis == 1
    assert abs(float(loaded.normal_pos)) < 1e-4
    assert loaded.points.sizes["index"] == 4


def test_triangular_from_vtu_far_from_origin_small_extent(tmp_path):
    """Regression: tolerance must depend on extent, not absolute position."""
    pytest.importorskip("vtk")
    import vtk

    import tidy3d as td

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0.0,
        points=td.PointDataArray(
            [
                [1e9, -2e9],
                [1e9 + 2.0, -2e9],
                [1e9, -2e9 + 1.0],
                [1e9 + 2.0, -2e9 + 1.0],
            ],
            dims=("index", "axis"),
        ),
        cells=td.CellDataArray([[0, 1, 2], [1, 2, 3]], dims=("cell_index", "vertex_index")),
        values=td.IndexedDataArray(
            [1.0, 2.0, 3.0, 4.0], coords={"index": np.arange(4)}, name="temp"
        ),
    )

    vtu_path = tmp_path / "far_origin_small_extent.vtu"
    tri_grid.to_vtu(vtu_path)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    grid = reader.GetOutput()
    points = grid.GetPoints()
    for ind in range(points.GetNumberOfPoints()):
        x, _, z = points.GetPoint(ind)
        points.SetPoint(ind, x, 2e-7 if (ind % 2) == 0 else -2e-7, z)
    points.Modified()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(vtu_path))
    writer.SetInputData(grid)
    writer.Write()

    loaded = td.TriangularGridDataset.from_vtu(vtu_path, field="temp")
    assert loaded.normal_axis == 1
    assert abs(float(loaded.normal_pos)) < 1e-5
    assert loaded.points.sizes["index"] == 4


def test_tetrahedral_from_vtk_obj_without_cell_types_array():
    """Regression test for VTK objects with missing ``GetCellTypesArray`` output."""
    pytest.importorskip("vtk")
    import tidy3d as td

    tet_grid = td.TetrahedralGridDataset(
        points=td.PointDataArray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dims=("index", "axis"),
        ),
        cells=td.CellDataArray([[0, 1, 2, 3]], dims=("cell_index", "vertex_index")),
        values=td.IndexedDataArray([1.0, 2.0, 3.0, 4.0], dims=("index")),
    )

    class VtkObjWithoutCellTypesArray:
        """Proxy object that mimics a VTK grid with no cell-types array."""

        def __init__(self, vtk_obj):
            self._vtk_obj = vtk_obj

        def __getattr__(self, name):
            return getattr(self._vtk_obj, name)

        def GetCellTypesArray(self):
            return None

    converted = tet_grid._from_vtk_obj_internal(
        VtkObjWithoutCellTypesArray(tet_grid._vtk_obj),
        remove_degenerate_cells=False,
        remove_unused_points=False,
    )

    assert converted == tet_grid
