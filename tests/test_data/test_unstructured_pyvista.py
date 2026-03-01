"""Tests PyVista visualization for TriangularSurfaceDataset."""

from __future__ import annotations

import numpy as np
import pytest

np.random.seed(4)


def _create_simple_surface():
    """Create a simple triangulated surface for testing."""
    import tidy3d as td

    # Create a 2x2 grid of triangles forming a pyramid
    points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],  # Raised center point
        ],
        coords={"index": np.arange(5), "axis": np.arange(3)},
    )

    cells = td.CellDataArray(
        [
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4],
        ],
        coords={"cell_index": np.arange(4), "vertex_index": np.arange(3)},
    )

    # Scalar field values
    values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        coords={"index": np.arange(5)},
    )

    return td.TriangularSurfaceDataset(points=points, cells=cells, values=values)


def _create_vector_surface():
    """Create a surface with vector field for quiver testing."""
    import tidy3d as td

    # Same geometry as simple surface
    points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        coords={"index": np.arange(5), "axis": np.arange(3)},
    )

    cells = td.CellDataArray(
        [
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4],
        ],
        coords={"cell_index": np.arange(4), "vertex_index": np.arange(3)},
    )

    # Vector field (pointing outward from center)
    vectors = td.IndexedDataArray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        coords={"index": np.arange(5), "axis": np.arange(3)},
    )

    return td.TriangularSurfaceDataset(points=points, cells=cells, values=vectors)


def test_surface_plot_basic(no_pyvista=False):
    """Test basic plot functionality."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(show=False)
    else:
        import pyvista as pv

        plotter = dataset.plot(show=False)
        assert isinstance(plotter, pv.Plotter), "Should return PyVista Plotter"
        plotter.close()


def test_surface_plot_with_grid(no_pyvista=False):
    """Test plot with grid overlay."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(grid=True, show=False)
    else:
        import pyvista as pv

        plotter = dataset.plot(grid=True, grid_color="white", grid_width=2, show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_plot_customization(no_pyvista=False):
    """Test plot customization options."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(cmap="plasma", show=False)
    else:
        import pyvista as pv

        plotter = dataset.plot(cmap="plasma", vmin=0, vmax=6, opacity=0.8, show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_plot_error_nothing_to_plot(no_pyvista=False):
    """Test error when both field and grid are False."""
    from tidy3d.exceptions import DataError, Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(field=False, grid=False, show=False)
    else:
        with pytest.raises(DataError, match="Nothing to plot"):
            _ = dataset.plot(field=False, grid=False, show=False)


def test_surface_plot_grid_only(no_pyvista=False):
    """Test plot with only grid, no field."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(field=False, grid=True, show=False)
    else:
        import pyvista as pv

        plotter = dataset.plot(field=False, grid=True, show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_plot_plotter_reuse(no_pyvista=False):
    """Test reusing plotter for multiple surfaces."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(show=False)
    else:
        import pyvista as pv

        # Create plotter manually
        plotter = pv.Plotter(off_screen=True)

        # Add first surface
        result1 = dataset.plot(plotter=plotter, show=False)

        # Add second surface with different styling
        result2 = dataset.plot(plotter=plotter, opacity=0.5, show=False)

        # Add third surface with different styling and positional plotter
        result3 = dataset.plot(plotter, opacity=0.5, show=False)

        assert result1 is plotter, "Should return same plotter"
        assert result2 is plotter, "Should return same plotter"
        assert result3 is plotter, "Should return same plotter"

        plotter.close()


def test_surface_quiver_basic(no_pyvista=False):
    """Test basic quiver plot."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_vector_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.quiver(show=False)
    else:
        import pyvista as pv

        plotter = dataset.quiver(show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_quiver_customization(no_pyvista=False):
    """Test quiver customization options."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_vector_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.quiver(scale=0.2, show=False)
    else:
        import pyvista as pv

        plotter = dataset.quiver(scale=0.2, downsampling=1, color="red", cbar=False, show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_quiver_error_wrong_num_fields(no_pyvista=False):
    """Test error when not exactly 3 fields."""
    from tidy3d.exceptions import DataError, Tidy3dImportError

    # Use scalar dataset (not vector)
    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.quiver(show=False)
    else:
        with pytest.raises(DataError, match="exactly 3 fields"):
            _ = dataset.quiver(show=False)


def test_surface_quiver_magnitude_coloring(no_pyvista=False):
    """Test quiver with magnitude-based coloring."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_vector_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.quiver(color="magnitude", show=False)
    else:
        import pyvista as pv

        plotter = dataset.quiver(color="magnitude", cmap="viridis", show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_combined_plot_and_quiver(no_pyvista=False):
    """Test combining surface plot with vector field on same plotter."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_vector_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.quiver(show=False)
    else:
        import pyvista as pv

        # Create plotter manually to test combination
        plotter = pv.Plotter(off_screen=True)

        # Add quiver (in real use case, would add surface of scalar field first)
        dataset.quiver(plotter=plotter, show=False)

        assert isinstance(plotter, pv.Plotter)
        plotter.close()


@pytest.mark.parametrize("windowed", [True, False, None])
def test_surface_plot_windowed_parameter(windowed, no_pyvista=False):
    """Test windowed parameter for display mode control."""
    from tidy3d.exceptions import Tidy3dImportError

    dataset = _create_simple_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(windowed=windowed, show=False)
    else:
        import pyvista as pv

        plotter = dataset.plot(windowed=windowed, show=False)
        assert isinstance(plotter, pv.Plotter)
        plotter.close()


def test_surface_plot_unsupported_plotting(no_pyvista=False):
    """Test plot with more than one field."""
    from tidy3d.exceptions import DataError, Tidy3dImportError

    # test more than one field not supported
    dataset = _create_vector_surface()

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(field=True, grid=True, show=False)
    else:
        _ = dataset.isel(axis=0).plot(field=True, grid=True, show=False)
        with pytest.raises(DataError):
            _ = dataset.plot(field=True, grid=True, show=False)

    # test plot with complex field not supported
    dataset = _create_simple_surface()
    dataset = dataset.updated_copy(values=dataset.values * (1 + 1j))

    if no_pyvista:
        with pytest.raises(Tidy3dImportError):
            _ = dataset.plot(field=True, grid=True, show=False)
    else:
        with pytest.raises(DataError):
            _ = dataset.plot(field=True, grid=True, show=False)
