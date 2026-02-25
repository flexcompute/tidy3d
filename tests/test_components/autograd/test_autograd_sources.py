"""
Analytical tests for source VJP computation.

Tests for ``td.CustomCurrentSource._compute_derivatives`` and
``td.CustomFieldSource._compute_derivatives`` using analytical solutions
for simple geometries and field distributions.

Test coverage:
 - Rectangular sources with uniform field distributions
 - Gaussian field distributions
 - Different source orientations and sizes
 - Edge cases with zero fields and boundary conditions
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import autograd as ag
import autograd.numpy as anp
import numpy as np
import numpy.testing as npt
import pytest

import tidy3d as td
from tidy3d.web import run

from .test_autograd import use_emulated_run  # noqa: F401


class DummySourceDI:
    """Stand-in for DerivativeInfo for source testing."""

    def __init__(
        self,
        *,
        paths,
        E_adj: dict,
        H_adj: dict | None = None,
        frequencies: np.ndarray,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        background_medium: td.Medium | None = None,
        eps_data: dict | None = None,
    ) -> None:
        self.paths = paths
        self.E_adj = E_adj
        self.H_adj = H_adj or {}
        self.frequencies = frequencies
        self.bounds = bounds
        self.E_fwd = {}  # Not used for sources
        self.D_adj = {}
        self.D_fwd = {}
        self.eps_data = eps_data
        self.eps_in = None
        self.eps_out = None
        self.eps_background = None
        self.eps_no_structure = None
        self.eps_inf_structure = None
        self.bounds_intersect = bounds
        self.background_medium = background_medium


def create_uniform_field_data(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    field_value: float = 1.0,
    num_points: int = 10,
) -> td.FieldDataset:
    """Create uniform field data for testing."""

    # Create grid coordinates
    x_min, x_max = center[0] - size[0] / 2, center[0] + size[0] / 2
    y_min, y_max = center[1] - size[1] / 2, center[1] + size[1] / 2
    z_min, z_max = center[2] - size[2] / 2, center[2] + size[2] / 2

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    z = np.linspace(z_min, z_max, max(1, num_points // 10))  # Fewer z points for 2D sources
    f = [2e14]  # Single frequency

    coords = {"x": x, "y": y, "z": z, "f": f}

    # Create uniform field data
    data_shape = (len(x), len(y), len(z), len(f))
    field_data = field_value * np.ones(data_shape)

    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)
    return td.FieldDataset(Ex=scalar_field, Ey=scalar_field, Ez=scalar_field)


def create_adjoint_field_dataarray(
    field_value: float,
    shape: tuple[int, int, int, int] = (10, 10, 5, 1),
) -> td.ScalarFieldDataArray:
    """Create adjoint field DataArray for testing."""

    # Create grid coordinates
    x = np.linspace(-0.5, 0.5, shape[0])
    y = np.linspace(-0.5, 0.5, shape[1])
    z = np.linspace(-0.05, 0.05, shape[2])
    f = [2e14]

    coords = {"x": x, "y": y, "z": z, "f": f}

    # Create uniform field data
    field_data = field_value * np.ones(shape)

    return td.ScalarFieldDataArray(field_data, coords=coords)


def create_gaussian_field_data(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    amplitude: float = 1.0,
    sigma: float = 0.1,
    num_points: int = 20,
) -> td.FieldDataset:
    """Create Gaussian field data for testing."""

    # Create grid coordinates
    x_min, x_max = center[0] - size[0] / 2, center[0] + size[0] / 2
    y_min, y_max = center[1] - size[1] / 2, center[1] + size[1] / 2
    z_min, z_max = center[2] - size[2] / 2, center[2] + size[2] / 2

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    z = np.linspace(z_min, z_max, max(1, num_points // 10))
    f = [2e14]

    # Create Gaussian field data
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Gaussian centered at source center
    r_sq = ((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2) / (2 * sigma**2)
    field_data = amplitude * np.exp(-r_sq)

    # Add frequency dimension to match coordinates
    field_data = field_data[..., np.newaxis]

    coords = {"x": x, "y": y, "z": z, "f": f}
    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords, dims=("x", "y", "z", "f"))
    return td.FieldDataset(Ex=scalar_field)


class TestSpatialWeights:
    """Unit tests for spatial weight helpers."""

    def test_compute_spatial_weights_cell_sizes(self):
        """Cell-size weights should match averaged coordinate spacing."""
        from tidy3d.components.autograd.derivative_utils import compute_spatial_weights

        coords = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 2.0]), "z": np.array([0.0])}
        values = np.zeros((3, 2, 1))
        arr = td.ScalarFieldDataArray(values, coords=coords, dims=("x", "y", "z"))

        weights = compute_spatial_weights(arr, dims=("x", "y", "z"))

        expected = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
        npt.assert_allclose(weights.values, expected)

    def test_transpose_interp_identity(self):
        """Adjoint interpolation should preserve weighted values on identical grids."""
        from tidy3d.components.autograd.derivative_utils import (
            compute_spatial_weights,
            transpose_interp_field_to_dataset,
        )

        coords = {
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        values = np.ones((2, 2, 1, 1), dtype=complex)
        adjoint_field = td.ScalarFieldDataArray(values, coords=coords, dims=("x", "y", "z", "f"))
        dataset_field = td.ScalarFieldDataArray(values, coords=coords, dims=("x", "y", "z", "f"))

        result = transpose_interp_field_to_dataset(
            adjoint_field, dataset_field, center=(0.0, 0.0, 0.0)
        )
        weights = compute_spatial_weights(adjoint_field, dims=("x", "y", "z"))
        expected = (adjoint_field * weights).transpose(*dataset_field.dims)

        npt.assert_allclose(result.values, expected.values)

    def test_transpose_interp_collapsed_axis(self):
        """Collapsed dataset axis should respect source-bounds cropping."""
        from tidy3d.components.autograd.derivative_utils import transpose_interp_field_to_dataset

        adjoint_coords = {
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        adjoint_values = np.ones((2, 1, 1, 1), dtype=complex)
        adjoint_field = td.ScalarFieldDataArray(
            adjoint_values, coords=adjoint_coords, dims=("x", "y", "z", "f")
        )

        dataset_coords = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        dataset_field = td.ScalarFieldDataArray(
            np.ones((1, 1, 1, 1)), coords=dataset_coords, dims=("x", "y", "z", "f")
        )

        result = transpose_interp_field_to_dataset(
            adjoint_field, dataset_field, center=(0.0, 0.0, 0.0)
        )
        npt.assert_allclose(result.values, 1.0)

    def test_transpose_interp_single_target_frequency_accumulates_all_inputs(self):
        """Single-frequency source datasets must accumulate all adjoint-frequency contributions."""
        from tidy3d.components.autograd.derivative_utils import transpose_interp_field_to_dataset

        adjoint_coords = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2.0e14, 3.0e14, 4.0e14]),
        }
        adjoint_field = td.ScalarFieldDataArray(
            np.array([[[[1.0, 2.0, 3.0]]]]),
            coords=adjoint_coords,
            dims=("x", "y", "z", "f"),
        )

        dataset_coords = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2.5e14]),
        }
        dataset_field = td.ScalarFieldDataArray(
            np.array([[[[1.0]]]]),
            coords=dataset_coords,
            dims=("x", "y", "z", "f"),
        )

        result = transpose_interp_field_to_dataset(
            adjoint_field, dataset_field, center=(0.0, 0.0, 0.0)
        )
        npt.assert_allclose(result.values, np.array([[[[6.0]]]]), rtol=1e-12, atol=1e-12)

    def test_transpose_interp_axis_requires_sorted_coordinates(self):
        """Axis-level transpose interpolation should enforce sorted source coordinates."""
        from tidy3d.components.autograd.derivative_utils import transpose_interp_axis

        field_values = np.ones((3, 1), dtype=float)
        field_coords = np.array([0.0, 0.5, 1.0])
        unsorted_param_coords = np.array([0.0, -0.5, 0.5])

        with pytest.raises(ValueError, match="must be sorted"):
            transpose_interp_axis(
                field_values=field_values,
                field_coords_1d=field_coords,
                param_coords_1d=unsorted_param_coords,
            )


class TestCustomCurrentSourceUniform:
    """Test CustomCurrentSource with uniform field distributions."""

    @pytest.fixture
    def source(self):
        """Create a CustomCurrentSource with uniform field data."""
        center = (0.0, 0.0, 0.0)
        size = (1.0, 1.0, 0.1)  # 2D source
        field_dataset = create_uniform_field_data(center, size, field_value=1.0)

        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

    @pytest.fixture
    def source_bounds(self, source):
        """Get the source bounds."""
        return source.geometry.bounds

    def test_uniform_adjoint_field(self, source, source_bounds):
        """Test with uniform adjoint field."""
        # Create uniform adjoint field
        adjoint_field_value = 2.0
        E_adj = {"Ex": create_adjoint_field_dataarray(adjoint_field_value)}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        field_data = source.current_dataset.Ex
        from tidy3d.components.autograd.derivative_utils import (
            source_scale_factor,
            transpose_interp_field_to_dataset,
        )

        adjoint_on_dataset = transpose_interp_field_to_dataset(
            E_adj["Ex"], field_data, center=source.center
        )
        source_scale = source_scale_factor(E_adj["Ex"], field_data)
        expected_gradient = np.sum(np.real(source_scale * adjoint_on_dataset).values)

        grad = results[("current_dataset", "Ex")]
        assert grad.shape == source.current_dataset.Ex.shape
        assert not np.isclose(expected_gradient, 0.0)
        assert not np.isclose(np.sum(grad), 0.0)
        npt.assert_allclose(np.sum(grad), expected_gradient, rtol=1e-2)

    def test_multi_frequency_adjoint_accumulates_for_single_frequency_dataset(self):
        """Current-source VJP must sum contributions from all adjoint frequencies."""
        coords = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2.5e14]),
        }
        source = td.CustomCurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            source_time=td.GaussianPulse(freq0=3.0e14, fwidth=1.0e14),
            current_dataset=td.FieldDataset(
                Ex=td.ScalarFieldDataArray(
                    np.array([[[[1.0]]]]), coords=coords, dims=("x", "y", "z", "f")
                )
            ),
        )

        adjoint_coords = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2.0e14, 3.0e14, 4.0e14]),
        }
        E_adj = {
            "Ex": td.ScalarFieldDataArray(
                np.array([[[[1.0, 2.0, 3.0]]]]),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=adjoint_coords["f"],
            bounds=source.geometry.bounds,
        )

        grad = source._compute_derivatives(di)[("current_dataset", "Ex")]
        npt.assert_allclose(grad, np.array([[[[12.0 * np.pi]]]]), rtol=1e-12, atol=1e-12)

    def test_interpolate_flag_changes_vjp_projection_on_zero_size_axis(self):
        """Current-source VJP should switch to nearest only along zero-size source axes."""
        coords = {
            "x": np.array([-0.4, -0.1, 0.2, 0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        dataset = td.FieldDataset(
            Ex=td.ScalarFieldDataArray(
                np.ones((4, 1, 1, 1)),
                coords=coords,
                dims=("x", "y", "z", "f"),
            )
        )
        source_linear = td.CustomCurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.5, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=dataset,
            interpolate=True,
        )
        source_nearest = source_linear.updated_copy(interpolate=False)

        adjoint_coords = {
            "x": np.array([-0.5, 0.0, 0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        E_adj = {
            "Ex": td.ScalarFieldDataArray(
                np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]]),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=adjoint_coords["f"],
            bounds=source_linear.geometry.bounds,
        )

        grad_linear = source_linear._compute_derivatives(di)[("current_dataset", "Ex")]
        grad_nearest = source_nearest._compute_derivatives(di)[("current_dataset", "Ex")]
        assert not np.allclose(grad_linear, grad_nearest)

    def test_interpolate_flag_keeps_linear_projection_on_nonzero_axis(self):
        """Current-source VJP should stay linear on nonzero-size source axes."""
        coords = {
            "x": np.array([-0.4, -0.1, 0.2, 0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        dataset = td.FieldDataset(
            Ex=td.ScalarFieldDataArray(
                np.ones((4, 1, 1, 1)),
                coords=coords,
                dims=("x", "y", "z", "f"),
            )
        )
        source_linear = td.CustomCurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(0.5, 0.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=dataset,
            interpolate=True,
        )
        source_nearest = source_linear.updated_copy(interpolate=False)

        adjoint_coords = {
            "x": np.array([-0.5, 0.0, 0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        E_adj = {
            "Ex": td.ScalarFieldDataArray(
                np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]]),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=adjoint_coords["f"],
            bounds=source_linear.geometry.bounds,
        )

        grad_linear = source_linear._compute_derivatives(di)[("current_dataset", "Ex")]
        grad_nearest = source_nearest._compute_derivatives(di)[("current_dataset", "Ex")]
        npt.assert_allclose(grad_nearest, grad_linear, rtol=1e-12, atol=1e-12)

    def test_confine_to_bounds_masks_out_of_bounds_dataset_points(self):
        """VJP should be zeroed at out-of-bounds dataset points when ``confine_to_bounds=True``."""
        coords = {
            "x": np.array([-1.0, 0.0, 1.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        dataset = td.FieldDataset(
            Ex=td.ScalarFieldDataArray(
                np.ones((3, 1, 1, 1)),
                coords=coords,
                dims=("x", "y", "z", "f"),
            )
        )
        source = td.CustomCurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=dataset,
            confine_to_bounds=False,
        )
        source_confined = source.updated_copy(confine_to_bounds=True)

        adjoint_coords = {
            "x": np.linspace(-1.0, 1.0, 9),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        E_adj = {
            "Ex": td.ScalarFieldDataArray(
                np.ones((9, 1, 1, 1)),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=adjoint_coords["f"],
            bounds=source.geometry.bounds,
        )

        grad_full = source._compute_derivatives(di)[("current_dataset", "Ex")]
        grad_confined = source_confined._compute_derivatives(di)[("current_dataset", "Ex")]

        assert not np.isclose(grad_full[0, 0, 0, 0], 0.0)
        assert not np.isclose(grad_full[2, 0, 0, 0], 0.0)
        npt.assert_allclose(grad_confined[0, 0, 0, 0], 0.0, atol=1e-12)
        npt.assert_allclose(grad_confined[2, 0, 0, 0], 0.0, atol=1e-12)
        assert np.isclose(grad_confined[1, 0, 0, 0], grad_full[1, 0, 0, 0])

    def test_unsorted_dataset_coords_supported(self):
        """Current-source VJP should support unsorted dataset coordinates by sorting internally."""
        coords = {
            "x": np.array([0.5, 0.0, -0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        source = td.CustomCurrentSource(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 0.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=td.FieldDataset(
                Ex=td.ScalarFieldDataArray(
                    np.ones((3, 1, 1, 1)),
                    coords=coords,
                    dims=("x", "y", "z", "f"),
                )
            ),
        )
        adjoint_coords = {
            "x": np.array([-0.5, 0.0, 0.5]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        E_adj = {
            "Ex": td.ScalarFieldDataArray(
                np.ones((3, 1, 1, 1)),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=adjoint_coords["f"],
            bounds=source.geometry.bounds,
        )

        grad = source._compute_derivatives(di)[("current_dataset", "Ex")]
        assert grad.shape == (3, 1, 1, 1)
        assert np.all(np.isfinite(grad))

    @pytest.mark.parametrize(
        ("shift_coords", "label"),
        (
            (False, "current_permittivity_scaling"),
            (True, "current_permittivity_shifted_coords"),
        ),
    )
    def test_uniform_adjoint_field_permittivity_invariance(
        self, source, source_bounds, shift_coords, label
    ):
        """Current-source gradients are invariant to supplied epsilon data and coordinate shifts."""
        adjoint_field_value = 2.0
        E_adj = {"Ex": create_adjoint_field_dataarray(adjoint_field_value)}
        field_data = source.current_dataset.Ex

        di_air = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        eps_rel = 2.25
        coord_offset = 8e-3 if shift_coords else 0.0
        z_offset = -8e-3 if shift_coords else 0.0
        freq_scale = 1 + 1e-9 if shift_coords else 1.0
        eps_coords = {
            "x": np.asarray(field_data.coords["x"].data) + source.center[0] + coord_offset,
            "y": np.asarray(field_data.coords["y"].data) + source.center[1] + coord_offset,
            "z": np.asarray(field_data.coords["z"].data) + source.center[2] + z_offset,
            "f": np.asarray(field_data.coords["f"].data) * freq_scale,
        }
        eps_values = eps_rel * np.ones(field_data.shape)
        eps_data = td.ScalarFieldDataArray(eps_values, coords=eps_coords, dims=field_data.dims)
        di_eps = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
            eps_data={"eps": eps_data},
        )

        grad_air = np.sum(source._compute_derivatives(di_air)[("current_dataset", "Ex")])
        grad_eps = np.sum(source._compute_derivatives(di_eps)[("current_dataset", "Ex")])
        ratio = grad_air / grad_eps

        assert not np.isclose(grad_air, 0.0)
        assert not np.isclose(grad_eps, 0.0)
        print(f"[{label}] ratio = {ratio}", file=sys.stderr)
        npt.assert_allclose(ratio, 1.0, rtol=1e-3)

    def test_zero_adjoint_field(self, source, source_bounds):
        """Test with zero adjoint field."""
        E_adj = {"Ex": create_adjoint_field_dataarray(0.0)}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Should be zero
        npt.assert_allclose(results[("current_dataset", "Ex")], 0.0, rtol=1e-10)

    def test_multiple_field_components(self, source, source_bounds):
        """Test with multiple field components."""
        adjoint_field_value = 1.5
        E_adj = {
            "Ex": create_adjoint_field_dataarray(adjoint_field_value),
            "Ey": create_adjoint_field_dataarray(0.5 * adjoint_field_value),
            "Ez": create_adjoint_field_dataarray(0.0),
        }
        di = DummySourceDI(
            paths=[("current_dataset", "Ex"), ("current_dataset", "Ey"), ("current_dataset", "Ez")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Check each component
        field_data = source.current_dataset.Ex
        from tidy3d.components.autograd.derivative_utils import (
            source_scale_factor,
            transpose_interp_field_to_dataset,
        )

        adjoint_on_dataset_ex = transpose_interp_field_to_dataset(
            E_adj["Ex"], field_data, center=source.center
        )
        adjoint_on_dataset_ey = transpose_interp_field_to_dataset(
            E_adj["Ey"], field_data, center=source.center
        )

        source_scale_ex = source_scale_factor(E_adj["Ex"], field_data)
        source_scale_ey = source_scale_factor(E_adj["Ey"], field_data)
        expected_ex = np.sum(np.real(source_scale_ex * adjoint_on_dataset_ex).values)
        expected_ey = np.sum(np.real(source_scale_ey * adjoint_on_dataset_ey).values)
        expected_ez = 0.0

        assert not np.isclose(expected_ex, 0.0)
        assert not np.isclose(expected_ey, 0.0)
        npt.assert_allclose(np.sum(results[("current_dataset", "Ex")]), expected_ex, rtol=1e-2)
        npt.assert_allclose(np.sum(results[("current_dataset", "Ey")]), expected_ey, rtol=1e-2)
        npt.assert_allclose(np.sum(results[("current_dataset", "Ez")]), expected_ez, rtol=1e-10)


class TestCustomFieldSourceUniform:
    """Test CustomFieldSource with uniform field distributions."""

    @pytest.fixture
    def source(self):
        """Create a CustomFieldSource with uniform field data."""
        center = (0.0, 0.0, 0.0)
        size = (1.0, 1.0, 0.0)  # Planar source (z=0)
        field_dataset = create_uniform_field_data(center, size, field_value=1.0)

        return td.CustomFieldSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            field_dataset=field_dataset,
        )

    @pytest.fixture
    def source_bounds(self, source):
        """Get the source bounds."""
        return source.geometry.bounds

    def test_uniform_adjoint_field(self, source, source_bounds):
        """Test with uniform adjoint field."""
        # Create uniform adjoint field
        adjoint_field_value = 2.0
        E_adj = {}
        H_adj = {"Hy": create_adjoint_field_dataarray(adjoint_field_value)}

        di = DummySourceDI(
            paths=[("field_dataset", "Ex")],
            E_adj=E_adj,
            H_adj=H_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Analytical solution
        from tidy3d.components.autograd.derivative_utils import (
            source_scale_factor,
            transpose_interp_field_to_dataset,
        )

        field_data = source.field_dataset.Ex
        adjoint_on_dataset = transpose_interp_field_to_dataset(
            H_adj["Hy"], field_data, center=source.center
        )
        source_scale = source_scale_factor(H_adj["Hy"], field_data)
        expected_gradient = np.sum(np.real(source_scale * adjoint_on_dataset).values)

        assert not np.isclose(expected_gradient, 0.0)
        assert not np.isclose(np.sum(results[("field_dataset", "Ex")]), 0.0)
        npt.assert_allclose(np.sum(results[("field_dataset", "Ex")]), expected_gradient, rtol=1e-2)

    def test_uniform_adjoint_field_invariant_to_dataset_spacing(self, source_bounds):
        """Summed field-source VJP should be resolution-invariant for same physical profile."""
        center = (0.0, 0.0, 0.0)
        size = (1.0, 1.0, 0.0)
        H_adj = {"Hy": create_adjoint_field_dataarray(2.0)}
        grad_sums = {}

        for num_points in (10, 20):
            x = np.linspace(-0.5, 0.5, num_points)
            y = np.linspace(-0.5, 0.5, num_points)
            z = np.array([0.0])
            f = [2e14]
            coords = {"x": x, "y": y, "z": z, "f": f}
            field_data = td.ScalarFieldDataArray(
                np.ones((num_points, num_points, 1, 1)), coords=coords
            )
            source = td.CustomFieldSource(
                center=center,
                size=size,
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
                field_dataset=td.FieldDataset(Ex=field_data),
            )
            di = DummySourceDI(
                paths=[("field_dataset", "Ex")],
                E_adj={},
                H_adj=H_adj,
                frequencies=np.array([2e14]),
                bounds=source.geometry.bounds,
            )
            grad_sums[num_points] = np.sum(source._compute_derivatives(di)[("field_dataset", "Ex")])

        assert not np.isclose(grad_sums[10], 0.0)
        spacing_ratio = grad_sums[20] / grad_sums[10]
        print(f"[field_spacing_invariance] ratio = {spacing_ratio}", file=sys.stderr)
        npt.assert_allclose(spacing_ratio, 1.0, rtol=2e-2)

    @pytest.mark.parametrize(
        ("shift_coords", "label"),
        (
            (False, "field_permittivity_scaling"),
            (True, "field_permittivity_shifted_coords"),
        ),
    )
    def test_uniform_adjoint_field_permittivity_invariance(
        self, source, source_bounds, shift_coords, label
    ):
        """Field-source gradients are invariant to supplied epsilon data and coordinate shifts."""
        adjoint_field_value = 2.0
        H_adj = {"Hy": create_adjoint_field_dataarray(adjoint_field_value)}
        field_data = source.field_dataset.Ex

        di_air = DummySourceDI(
            paths=[("field_dataset", "Ex")],
            E_adj={},
            H_adj=H_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        eps_rel = 2.25
        coord_offset = 8e-3 if shift_coords else 0.0
        z_offset = -8e-3 if shift_coords else 0.0
        freq_scale = 1 + 1e-9 if shift_coords else 1.0
        eps_coords = {
            "x": np.asarray(field_data.coords["x"].data) + source.center[0] + coord_offset,
            "y": np.asarray(field_data.coords["y"].data) + source.center[1] + coord_offset,
            "z": np.asarray(field_data.coords["z"].data) + source.center[2] + z_offset,
            "f": np.asarray(field_data.coords["f"].data) * freq_scale,
        }
        eps_values = eps_rel * np.ones(field_data.shape)
        eps_data = td.ScalarFieldDataArray(eps_values, coords=eps_coords, dims=field_data.dims)
        di_eps = DummySourceDI(
            paths=[("field_dataset", "Ex")],
            E_adj={},
            H_adj=H_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
            eps_data={"eps": eps_data},
        )

        grad_air = np.sum(source._compute_derivatives(di_air)[("field_dataset", "Ex")])
        grad_eps = np.sum(source._compute_derivatives(di_eps)[("field_dataset", "Ex")])
        ratio = grad_air / grad_eps

        assert not np.isclose(grad_air, 0.0)
        assert not np.isclose(grad_eps, 0.0)
        print(f"[{label}] ratio = {ratio}", file=sys.stderr)
        npt.assert_allclose(ratio, 1.0, rtol=1e-3)

    def test_unsorted_dataset_coords_supported(self):
        """Field-source VJP should support unsorted dataset coordinates by sorting internally."""
        coords = {
            "x": np.array([0.5, 0.0, -0.5]),
            "y": np.array([-0.5, 0.0, 0.5]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        source = td.CustomFieldSource(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            field_dataset=td.FieldDataset(
                Ex=td.ScalarFieldDataArray(
                    np.ones((3, 3, 1, 1)),
                    coords=coords,
                    dims=("x", "y", "z", "f"),
                )
            ),
        )
        adjoint_coords = {
            "x": np.array([-0.5, 0.0, 0.5]),
            "y": np.array([-0.5, 0.0, 0.5]),
            "z": np.array([0.0]),
            "f": np.array([2e14]),
        }
        H_adj = {
            "Hy": td.ScalarFieldDataArray(
                np.ones((3, 3, 1, 1)),
                coords=adjoint_coords,
                dims=("x", "y", "z", "f"),
            )
        }
        di = DummySourceDI(
            paths=[("field_dataset", "Ex")],
            E_adj={},
            H_adj=H_adj,
            frequencies=adjoint_coords["f"],
            bounds=source.geometry.bounds,
        )

        grad = source._compute_derivatives(di)[("field_dataset", "Ex")]
        assert grad.shape == (3, 3, 1, 1)
        assert np.all(np.isfinite(grad))


@pytest.mark.parametrize(
    (
        "source_ctor",
        "dataset_key",
        "source_size",
        "unsupported_path",
        "adjoint_component",
        "adjoint_value",
    ),
    (
        (td.CustomCurrentSource, "current_dataset", (1.0, 1.0, 0.1), ("center", 0), "Ex", -1j),
        (td.CustomFieldSource, "field_dataset", (1.0, 1.0, 0.0), ("size", 0), "Hy", 1j),
    ),
)
def test_unsupported_traced_paths_raise_error(
    source_ctor,
    dataset_key,
    source_size,
    unsupported_path,
    adjoint_component,
    adjoint_value,
):
    """Unsupported traced source parameters should raise an explicit error."""

    center = (0.0, 0.0, 0.0)
    field_dataset = create_uniform_field_data(center, source_size, field_value=1.0)

    source = source_ctor(
        center=center,
        size=source_size,
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        **{dataset_key: field_dataset},
    )

    E_adj = {}
    H_adj = {}
    if adjoint_component.startswith("E"):
        E_adj[adjoint_component] = create_adjoint_field_dataarray(adjoint_value)
    else:
        H_adj[adjoint_component] = create_adjoint_field_dataarray(adjoint_value)

    di = DummySourceDI(
        paths=[(dataset_key, "Ex"), unsupported_path, ("source_time", "freq0")],
        E_adj=E_adj,
        H_adj=H_adj,
        frequencies=np.array([2e14]),
        bounds=source.geometry.bounds,
    )

    with pytest.raises(ValueError, match="not supported"):
        source._compute_derivatives(di)


class TestCustomCurrentSourceGaussian:
    """Test CustomCurrentSource with Gaussian field distributions."""

    @pytest.fixture
    def source(self):
        """Create a CustomCurrentSource with Gaussian field data."""
        center = (0.0, 0.0, 0.0)
        size = (0.5, 0.5, 0.1)  # Smaller source for Gaussian test
        field_dataset = create_gaussian_field_data(center, size, amplitude=1.0, sigma=0.1)

        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

    @pytest.fixture
    def source_bounds(self, source):
        """Get the source bounds."""
        return source.geometry.bounds

    def test_gaussian_adjoint_field(self, source, source_bounds):
        """Test with Gaussian adjoint field."""
        # Create Gaussian adjoint field
        adjoint_amplitude = 1.0
        sigma = 0.1
        x = np.linspace(-0.25, 0.25, 10)
        y = np.linspace(-0.25, 0.25, 10)
        z = np.array([0.0])
        f = [2e14]

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        r_sq = (X**2 + Y**2 + Z**2) / (2 * sigma**2)
        adjoint_field_data = adjoint_amplitude * np.exp(-r_sq)

        # Add frequency dimension to match coordinates
        adjoint_field_data = adjoint_field_data[..., np.newaxis]
        coords = {"x": x, "y": y, "z": z, "f": f}
        adjoint_field = td.ScalarFieldDataArray(
            adjoint_field_data, coords=coords, dims=("x", "y", "z", "f")
        )

        E_adj = {"Ex": adjoint_field}
        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        from tidy3d.components.autograd.derivative_utils import (
            source_scale_factor,
            transpose_interp_field_to_dataset,
        )

        adjoint_on_dataset = transpose_interp_field_to_dataset(
            adjoint_field,
            source.current_dataset.Ex,
            center=source.center,
        )
        source_scale = source_scale_factor(adjoint_field, source.current_dataset.Ex)
        expected_gradient = np.sum(np.real(source_scale * adjoint_on_dataset).values)

        assert not np.isclose(expected_gradient, 0.0)
        assert not np.isclose(np.sum(results[("current_dataset", "Ex")]), 0.0)
        npt.assert_allclose(
            np.sum(results[("current_dataset", "Ex")]), expected_gradient, rtol=5e-1
        )


@pytest.mark.parametrize(
    ("source_ctor", "dataset_key"),
    (
        (td.CustomCurrentSource, "current_dataset"),
        (td.CustomFieldSource, "field_dataset"),
    ),
)
def test_source_adjoint_monitors(source_ctor, dataset_key):
    """Test that adjoint monitors are properly created for traced source datasets."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[],
        monitors=[
            td.FieldMonitor(
                size=(1.0, 1.0, 0.0), center=(0, 0, 0), freqs=[2e14], name="field_monitor"
            )
        ],
    )

    # Create traced field data
    data_shape = (10, 10, 1, 1)
    x = np.linspace(-0.5, 0.5, data_shape[0])
    y = np.linspace(-0.5, 0.5, data_shape[1])
    z = np.array([0])
    f = [2e14]
    coords = {"x": x, "y": y, "z": z, "f": f}

    field_data = 1.0 * np.ones(data_shape)
    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)
    field_dataset = td.FieldDataset(Ex=scalar_field)

    custom_source = source_ctor(
        center=(0, 0, 0),
        size=(1.0, 1.0, 0.0),
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        **{dataset_key: field_dataset},
    )
    sim = sim.updated_copy(sources=[custom_source])
    sim_fields_keys = [("sources", 0, dataset_key, "Ex")]

    # Test that adjoint monitors are created
    adjoint_monitors_fld, adjoint_monitors_eps = sim._make_adjoint_monitors(sim_fields_keys)

    # Check that field monitors were created for sources, but no for eps
    assert len(adjoint_monitors_fld) == 1
    assert len(adjoint_monitors_eps) == 0

    # Check that the field monitor covers the source region
    field_monitor = adjoint_monitors_fld[0]
    assert isinstance(field_monitor, td.FieldMonitor)
    assert field_monitor.center == custom_source.center
    assert field_monitor.size == custom_source.size
    assert len(field_monitor.freqs) == len(sim._freqs_adjoint)
    assert len(field_monitor.freqs) > 0


def test_mixed_structure_source_adjoint_monitors():
    """Test that adjoint monitors work correctly when both structures and sources are traced."""

    # Create a simulation with both structures and sources
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[],
        structures=[
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(0.5, 0.5, 0.5)),
                medium=td.Medium(permittivity=2.0),
            )
        ],
        monitors=[
            td.FieldMonitor(
                size=(1.0, 1.0, 0.0), center=(0, 0, 0), freqs=[2e14], name="field_monitor"
            )
        ],
    )

    # Create traced field data for source
    data_shape = (10, 10, 1, 1)
    x = np.linspace(-0.5, 0.5, data_shape[0])
    y = np.linspace(-0.5, 0.5, data_shape[1])
    z = np.array([0])
    f = [2e14]
    coords = {"x": x, "y": y, "z": z, "f": f}

    field_data = 1.0 * np.ones(data_shape)
    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)
    field_dataset = td.FieldDataset(Ex=scalar_field)

    # Create CustomCurrentSource with traced dataset
    custom_source = td.CustomCurrentSource(
        center=(-0.5, 0, 0),
        size=(0.5, 0.5, 0.0),
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        current_dataset=field_dataset,
    )

    # Add source to simulation
    sim = sim.updated_copy(sources=[custom_source])

    # Create sim_fields_keys for both structure and source
    sim_fields_keys = [
        ("structures", 0, "medium", "permittivity"),
        ("sources", 0, "current_dataset", "Ex"),
    ]

    # Test that adjoint monitors are created for both
    adjoint_monitors_fld, adjoint_monitors_eps = sim._make_adjoint_monitors(sim_fields_keys)

    # Should have monitors for both structure and source
    # Note: The structure might not create monitors if it doesn't have the right field keys
    # Let's be more flexible about the expected number
    assert len(adjoint_monitors_fld) == 2  # two field monitors (one for structure, one for source)
    assert len(adjoint_monitors_eps) == 1  # only one eps monitor for structure

    # Check that we have at least one source monitor
    source_monitor_found = False
    for _i, field_monitor_item in enumerate(adjoint_monitors_fld):
        # Handle both direct FieldMonitor and list of FieldMonitor
        if isinstance(field_monitor_item, td.FieldMonitor):
            # Direct FieldMonitor (could be structure or source)
            field_monitor = field_monitor_item
            # Check if this is our source monitor
            if (
                field_monitor.center == custom_source.center
                and field_monitor.size == custom_source.size
            ):
                assert len(field_monitor.freqs) > 0
                source_monitor_found = True
                break
        elif isinstance(field_monitor_item, list):
            # List of FieldMonitor (source monitors are wrapped in lists)
            for field_monitor in field_monitor_item:
                if isinstance(field_monitor, td.FieldMonitor):
                    # Check if this is our source monitor
                    if (
                        field_monitor.center == custom_source.center
                        and field_monitor.size == custom_source.size
                    ):
                        assert len(field_monitor.freqs) > 0
                        source_monitor_found = True
                        break
            if source_monitor_found:
                break

    assert source_monitor_found, "No source monitor found in adjoint monitors"


def test_split_adjoint_data_logs_mixed_source_structure_counts(monkeypatch):
    """Adjoint split log should report field/eps counts without assuming 1:1 pairing."""
    from tidy3d.components.data import sim_data as sim_data_module

    monitors = [
        td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="orig"),
        td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="adjoint_fld_0"),
        td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="source_adjoint_0"),
        td.PermittivityMonitor(
            center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="adjoint_eps_0"
        ),
    ]
    dummy_sim_data = SimpleNamespace(
        data=["orig_data", "adj_fld_data", "source_adj_data", "adj_eps_data"],
        simulation=SimpleNamespace(monitors=monitors),
    )

    messages = []
    monkeypatch.setattr(sim_data_module.log, "info", lambda msg: messages.append(msg))

    data_original, data_adjoint = sim_data_module.SimulationData._split_adjoint_data(
        dummy_sim_data, num_mnts_original=1
    )

    assert data_original == ["orig_data"]
    assert data_adjoint == ["adj_fld_data", "source_adj_data", "adj_eps_data"]
    assert any(
        "1 monitors, 2 adjoint field monitors, 1 adjoint eps monitors." in msg for msg in messages
    )


def _make_uniform_field_dataset(val, data_shape=(10, 10, 1, 1), freq=2e14):
    x = np.linspace(-0.5, 0.5, data_shape[0])
    y = np.linspace(-0.5, 0.5, data_shape[1])
    z = np.array([0])
    f = [freq]
    coords = {"x": x, "y": y, "z": z, "f": f}

    field_data = val * np.ones(data_shape)
    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)
    return td.FieldDataset(Ex=scalar_field)


SOURCE_CASES = [
    pytest.param(
        "custom_current_source",
        lambda val, freq: td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=freq, fwidth=1e13),
            current_dataset=_make_uniform_field_dataset(val, freq=freq),
        ),
        id="CustomCurrentSource",
    ),
    pytest.param(
        "custom_field_source",
        lambda val, freq: td.CustomFieldSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=freq, fwidth=1e13),
            field_dataset=_make_uniform_field_dataset(val, freq=freq),
        ),
        id="CustomFieldSource",
    ),
]


@pytest.mark.parametrize("kind, make_source", SOURCE_CASES)
def test_traced_source_derivative_computation(use_emulated_run, kind, make_source):  # noqa: F811
    """Test that traced source derivative computation works for different source types."""
    freq = 2e14

    def make_sim(val):
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[],
            monitors=[
                td.FieldMonitor(
                    size=(1.0, 1.0, 0.0),
                    center=(0, 0, 0),
                    freqs=[freq],
                    name="field_monitor",
                )
            ],
        )
        src = make_source(val, freq)
        return sim.updated_copy(sources=[src])

    def objective(val):
        sim = make_sim(val)
        sim_data = run(sim, task_name=f"test_derivative_{kind}")
        field_data = sim_data.load_field_monitor("field_monitor")
        Ex_field = field_data.Ex
        return anp.abs(Ex_field.isel(x=5, y=5, z=0, f=0).values) ** 2

    grad = ag.grad(objective)(1.0)

    assert grad is not None
    assert isinstance(grad, (float, np.ndarray))
    assert np.all(np.asarray(grad) != 0.0), "some gradients are 0"
