"""Tests PyVista visualization without pyvista installed."""

from __future__ import annotations

import builtins

import pytest

from ..test_data.test_unstructured_pyvista import (
    test_surface_combined_plot_and_quiver as _test_surface_combined_plot_and_quiver,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_basic as _test_surface_plot_basic,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_customization as _test_surface_plot_customization,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_error_nothing_to_plot as _test_surface_plot_error_nothing_to_plot,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_grid_only as _test_surface_plot_grid_only,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_plotter_reuse as _test_surface_plot_plotter_reuse,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_windowed_parameter as _test_surface_plot_windowed_parameter,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_plot_with_grid as _test_surface_plot_with_grid,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_quiver_basic as _test_surface_quiver_basic,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_quiver_customization as _test_surface_quiver_customization,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_quiver_error_wrong_num_fields as _test_surface_quiver_error_wrong_num_fields,
)
from ..test_data.test_unstructured_pyvista import (
    test_surface_quiver_magnitude_coloring as _test_surface_quiver_magnitude_coloring,
)


@pytest.fixture
def hide_pyvista(monkeypatch):
    """Hide pyvista module to test error handling when not installed."""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "pyvista":
            raise ImportError
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_basic_no_pyvista():
    _test_surface_plot_basic(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_with_grid_no_pyvista():
    _test_surface_plot_with_grid(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_customization_no_pyvista():
    _test_surface_plot_customization(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_error_nothing_to_plot_no_pyvista():
    _test_surface_plot_error_nothing_to_plot(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_grid_only_no_pyvista():
    _test_surface_plot_grid_only(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_plotter_reuse_no_pyvista():
    _test_surface_plot_plotter_reuse(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_quiver_basic_no_pyvista():
    _test_surface_quiver_basic(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_quiver_customization_no_pyvista():
    _test_surface_quiver_customization(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_quiver_error_wrong_num_fields_no_pyvista():
    _test_surface_quiver_error_wrong_num_fields(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_quiver_magnitude_coloring_no_pyvista():
    _test_surface_quiver_magnitude_coloring(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_combined_plot_and_quiver_no_pyvista():
    _test_surface_combined_plot_and_quiver(no_pyvista=True)


@pytest.mark.usefixtures("hide_pyvista")
def test_surface_plot_windowed_parameter_no_pyvista():
    _test_surface_plot_windowed_parameter(None, no_pyvista=True)
