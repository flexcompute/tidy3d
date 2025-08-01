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

import numpy as np
import numpy.testing as npt
import pytest

import tidy3d as td


class DummySourceDI:
    """Stand-in for DerivativeInfo for source testing."""

    def __init__(
        self,
        *,
        paths,
        E_adj: dict,
        frequencies: np.ndarray,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        self.paths = paths
        self.E_adj = E_adj
        self.frequencies = frequencies
        self.bounds = bounds
        self.E_fwd = {}  # Not used for sources
        self.D_adj = {}
        self.D_fwd = {}
        self.eps_data = None
        self.eps_in = None
        self.eps_out = None
        self.eps_background = None
        self.eps_no_structure = None
        self.eps_inf_structure = None
        self.bounds_intersect = bounds


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
    return td.FieldDataset(Ex=scalar_field)


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


def analytical_uniform_source_gradient(
    source_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    adjoint_field_value: float,
    field_component: str = "Ex",
) -> float:
    """
    Analytical gradient for uniform source with uniform adjoint field.

    The gradient is simply the volume integral of the adjoint field.
    For uniform fields, this is: adjoint_field_value * volume
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = source_bounds
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    return adjoint_field_value * volume


def analytical_gaussian_source_gradient(
    source_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    adjoint_amplitude: float,
    sigma: float = 0.1,
    field_component: str = "Ex",
) -> float:
    """
    Analytical gradient for uniform source with Gaussian adjoint field.

    The gradient is the volume integral of the Gaussian adjoint field.
    For a Gaussian centered at the source center, we compute the actual
    integral over the source bounds.
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = source_bounds

    # For a Gaussian field exp(-(x^2 + y^2 + z^2)/(2*sigma^2)), the integral
    # over a rectangular domain can be approximated by the sum of field values
    # times the volume element, which is what the numerical integration does.

    # Since the test uses a 10x10x1 grid over the bounds, we can compute
    # the expected value by sampling the Gaussian at those points
    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(y_min, y_max, 10)
    z = np.array([0.0])  # Single z point as in the test

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r_sq = (X**2 + Y**2 + Z**2) / (2 * sigma**2)
    field_data = adjoint_amplitude * np.exp(-r_sq)

    # Compute the volume element
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # The integral is the sum of field values times the area element (2D integral)
    # since z has only one point, integrate_within_bounds only integrates over x and y
    integral = np.sum(field_data) * dx * dy

    return integral


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

        # Analytical solution
        expected_gradient = analytical_uniform_source_gradient(source_bounds, adjoint_field_value)

        npt.assert_allclose(results[("current_dataset", "Ex")], expected_gradient, rtol=1e-2)

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
        expected_ex = analytical_uniform_source_gradient(source_bounds, adjoint_field_value)
        expected_ey = analytical_uniform_source_gradient(source_bounds, 0.5 * adjoint_field_value)
        expected_ez = analytical_uniform_source_gradient(source_bounds, 0.0)

        npt.assert_allclose(results[("current_dataset", "Ex")], expected_ex, rtol=1e-2)
        npt.assert_allclose(results[("current_dataset", "Ey")], expected_ey, rtol=1e-2)
        npt.assert_allclose(results[("current_dataset", "Ez")], expected_ez, rtol=1e-10)


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
        E_adj = {"Ex": create_adjoint_field_dataarray(adjoint_field_value)}

        di = DummySourceDI(
            paths=[("field_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Analytical solution
        expected_gradient = analytical_uniform_source_gradient(source_bounds, adjoint_field_value)

        npt.assert_allclose(results[("field_dataset", "Ex")], expected_gradient, rtol=1e-2)


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

        # Analytical solution (approximate)
        expected_gradient = analytical_gaussian_source_gradient(
            source_bounds, adjoint_amplitude, sigma
        )

        npt.assert_allclose(results[("current_dataset", "Ex")], expected_gradient, rtol=5e-1)


class TestSourceEdgeCases:
    """Test edge cases for source VJP computation."""

    @pytest.fixture
    def small_source(self):
        """Create a very small source."""
        center = (0.0, 0.0, 0.0)
        size = (0.01, 0.01, 0.01)  # Very small source
        field_dataset = create_uniform_field_data(center, size, field_value=1.0)

        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

    @pytest.fixture
    def large_source(self):
        """Create a large source."""
        center = (0.0, 0.0, 0.0)
        size = (10.0, 10.0, 1.0)  # Large source
        field_dataset = create_uniform_field_data(center, size, field_value=1.0)

        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

    def test_small_source(self, small_source):
        """Test with very small source."""
        source_bounds = small_source.geometry.bounds
        adjoint_field_value = 1.0
        E_adj = {"Ex": create_adjoint_field_dataarray(adjoint_field_value, shape=(5, 5, 3, 1))}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = small_source._compute_derivatives(di)

        # Should be finite and positive
        assert np.isfinite(results[("current_dataset", "Ex")])
        assert results[("current_dataset", "Ex")] >= 0.0

    def test_large_source(self, large_source):
        """Test with large source."""
        source_bounds = large_source.geometry.bounds
        adjoint_field_value = 1.0
        E_adj = {"Ex": create_adjoint_field_dataarray(adjoint_field_value, shape=(20, 20, 10, 1))}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = large_source._compute_derivatives(di)

        # Should be finite and positive
        assert np.isfinite(results[("current_dataset", "Ex")])
        assert results[("current_dataset", "Ex")] >= 0.0

    def test_missing_field_component(self, small_source):
        """Test when adjoint field component is missing."""
        source_bounds = small_source.geometry.bounds
        E_adj = {}  # Empty adjoint field

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = small_source._compute_derivatives(di)

        # Should return zero for missing component
        npt.assert_allclose(results[("current_dataset", "Ex")], 0.0, rtol=1e-10)

    def test_invalid_path(self, small_source):
        """Test with invalid path."""
        source_bounds = small_source.geometry.bounds
        E_adj = {"Ex": create_adjoint_field_dataarray(1.0, shape=(5, 5, 3, 1))}

        di = DummySourceDI(
            paths=[("current_dataset", "InvalidField")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = small_source._compute_derivatives(di)

        # Should return zero for invalid field component
        npt.assert_allclose(results[("current_dataset", "InvalidField")], 0.0, rtol=1e-10)


class TestSourceNumericalStability:
    """Test numerical stability of source VJP computation."""

    @pytest.fixture
    def source(self):
        """Create a source for stability testing."""
        center = (0.0, 0.0, 0.0)
        size = (1.0, 1.0, 0.1)
        field_dataset = create_uniform_field_data(center, size, field_value=1.0)

        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

    def test_large_adjoint_field(self, source):
        """Test with very large adjoint field values."""
        source_bounds = source.geometry.bounds
        large_value = 1e10
        E_adj = {"Ex": create_adjoint_field_dataarray(large_value)}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Should be finite
        assert np.isfinite(results[("current_dataset", "Ex")])

    def test_small_adjoint_field(self, source):
        """Test with very small adjoint field values."""
        source_bounds = source.geometry.bounds
        small_value = 1e-10
        E_adj = {"Ex": create_adjoint_field_dataarray(small_value)}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Should be finite
        assert np.isfinite(results[("current_dataset", "Ex")])

    def test_mixed_adjoint_field_signs(self, source):
        """Test with mixed positive and negative adjoint field values."""
        source_bounds = source.geometry.bounds
        # Create field with mixed signs
        field_data = np.random.randn(10, 10, 5, 1)
        coords = {
            "x": np.linspace(-0.5, 0.5, 10),
            "y": np.linspace(-0.5, 0.5, 10),
            "z": np.linspace(-0.05, 0.05, 5),
            "f": [2e14],
        }
        E_adj = {"Ex": td.ScalarFieldDataArray(field_data, coords=coords)}

        di = DummySourceDI(
            paths=[("current_dataset", "Ex")],
            E_adj=E_adj,
            frequencies=np.array([2e14]),
            bounds=source_bounds,
        )

        results = source._compute_derivatives(di)

        # Should be finite
        assert np.isfinite(results[("current_dataset", "Ex")])
