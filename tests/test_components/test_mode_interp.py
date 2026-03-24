"""Tests for mode frequency interpolation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

# from tidy3d.plugins.smatrix.ports.wave import DEFAULT_WAVE_PORT_MODE_SPEC
from ..test_data.test_data_arrays import MODE_SPEC, SIZE_2D
from ..utils import AssertLogLevel, AssertLogStr

# Shared test constants
FREQS_DENSE = np.linspace(1e14, 2e14, 20)


# ============================================================================
# ModeInterpSpec Tests
# ============================================================================


def test_interp_spec_valid_linear():
    """Test creating valid ModeInterpSpec with linear interpolation."""
    spec = td.ModeInterpSpec.uniform(num_points=5, method="linear")
    assert spec.num_points == 5
    assert spec.method == "linear"
    assert not spec.reduce_data  # default value


def test_interp_spec_valid_cubic():
    """Test creating valid ModeInterpSpec with cubic interpolation."""
    spec = td.ModeInterpSpec.uniform(num_points=10, method="cubic")
    assert spec.num_points == 10
    assert spec.method == "cubic"
    assert not spec.reduce_data  # default value


def test_interp_spec_default_method():
    """Test that default method is 'linear' and reduce_data is False."""
    spec = td.ModeInterpSpec.uniform(num_points=5)
    assert spec.method == "linear"
    assert not spec.reduce_data


def test_interp_spec_cubic_needs_4_points():
    """Test that cubic interpolation requires at least 4 points."""
    with pytest.raises(ValidationError, match="Cubic interpolation requires at least 4"):
        td.ModeInterpSpec.uniform(num_points=3, method="cubic")


def test_interp_spec_valid_poly():
    """Test creating valid ModeInterpSpec with polynomial interpolation."""
    spec = td.ModeInterpSpec.uniform(num_points=10, method="poly")
    assert spec.num_points == 10
    assert spec.method == "poly"


def test_interp_spec_poly_needs_3_points():
    """Test that polynomial interpolation requires at least 3 points."""
    with pytest.raises(ValidationError, match="Polynomial interpolation requires at least 3"):
        td.ModeInterpSpec.uniform(num_points=2, method="poly")


def test_interp_spec_cheb_convenience():
    """Test the convenience method for Chebyshev sampling with polynomial interpolation."""
    spec = td.ModeInterpSpec.cheb(num_points=10)
    assert spec.num_points == 10
    assert spec.method == "poly"
    assert isinstance(spec.sampling_spec, td.ChebSampling)


def test_interp_spec_uniform_convenience():
    """Test the convenience method for uniform sampling."""
    spec = td.ModeInterpSpec.uniform(num_points=5, method="linear")
    assert spec.num_points == 5
    assert spec.method == "linear"
    assert isinstance(spec.sampling_spec, td.UniformSampling)


def test_interp_spec_custom_convenience():
    """Test the convenience method for custom sampling."""
    custom_freqs = [1e14, 1.5e14, 1.8e14, 2e14]
    spec = td.ModeInterpSpec.custom(freqs=custom_freqs, method="cubic")
    assert spec.num_points == 4
    assert spec.method == "cubic"
    assert isinstance(spec.sampling_spec, td.CustomSampling)


def test_interp_spec_sampling_points_uniform():
    """Test sampling_points for uniform sampling."""
    spec = td.ModeInterpSpec.uniform(num_points=5, method="linear")
    freqs = np.linspace(1e14, 2e14, 100)
    sampling = spec.sampling_points(freqs)

    assert len(sampling) == 5
    assert np.isclose(sampling[0], 1e14)
    assert np.isclose(sampling[-1], 2e14)
    # Check uniform spacing
    diffs = np.diff(sampling)
    assert np.allclose(diffs, diffs[0])
    # Check ascending order
    assert np.all(np.diff(sampling) > 0), "Uniform frequencies should be in ascending order"


def test_interp_spec_sampling_points_cheb():
    """Test sampling_points for Chebyshev sampling."""
    spec = td.ModeInterpSpec.cheb(num_points=5)
    freqs = np.linspace(1e14, 2e14, 100)
    sampling = spec.sampling_points(freqs)

    assert len(sampling) == 5
    # Chebyshev nodes should include endpoints
    assert np.isclose(sampling.min(), 1e14)
    assert np.isclose(sampling.max(), 2e14)

    # Verify they are Chebyshev nodes
    f_min, f_max = 1e14, 2e14
    k = np.arange(5)
    expected_normalized = np.cos(k * np.pi / 4)
    expected = 0.5 * (f_min + f_max) + 0.5 * (f_max - f_min) * expected_normalized
    assert np.allclose(np.sort(sampling), np.sort(expected))


def test_cheb_sampling_ascending_order():
    """Test that Chebyshev sampling returns frequencies in ascending order."""
    spec = td.ModeInterpSpec.cheb(num_points=10)
    freqs = np.linspace(1e14, 2e14, 100)
    sampling = spec.sampling_points(freqs)

    # Verify the frequencies are in strictly ascending order
    assert len(sampling) == 10
    assert np.all(np.diff(sampling) > 0), "Chebyshev frequencies should be in ascending order"

    # Verify endpoints are correct
    assert np.isclose(sampling[0], 1e14), "First frequency should be f_min"
    assert np.isclose(sampling[-1], 2e14), "Last frequency should be f_max"


def test_interp_spec_sampling_points_custom():
    """Test sampling_points for custom sampling."""
    custom_freqs = np.array([1e14, 1.3e14, 1.7e14, 2e14])
    spec = td.ModeInterpSpec.custom(freqs=custom_freqs, method="linear")
    freqs = np.linspace(1e14, 2e14, 100)
    sampling = spec.sampling_points(freqs)

    assert len(sampling) == 4
    assert np.allclose(sampling, custom_freqs)


def test_interp_spec_min_2_points():
    """Test that at least 2 points are required."""
    with pytest.raises(ValidationError):
        td.ModeInterpSpec.uniform(num_points=1, method="linear")


def test_interp_spec_positive_points():
    """Test that num_points must be positive."""
    with pytest.raises(ValidationError):
        td.ModeInterpSpec.uniform(num_points=0, method="linear")

    with pytest.raises(ValidationError):
        td.ModeInterpSpec.uniform(num_points=-5, method="linear")


def test_interp_spec_invalid_method():
    """Test that invalid interpolation method is rejected."""
    with pytest.raises(ValidationError):
        td.ModeInterpSpec.uniform(num_points=5, method="quadratic")


def test_interp_spec_reduce_data_true():
    """Test creating ModeInterpSpec with reduce_data=True."""
    spec = td.ModeInterpSpec.uniform(num_points=5, method="linear", reduce_data=True)
    assert spec.num_points == 5
    assert spec.method == "linear"
    assert spec.reduce_data


def test_interp_spec_reduce_data_false():
    """Test creating ModeInterpSpec with reduce_data=False (explicit)."""
    spec = td.ModeInterpSpec.uniform(num_points=5, method="linear", reduce_data=False)
    assert spec.num_points == 5
    assert spec.method == "linear"
    assert not spec.reduce_data


def test_interp_spec_requires_tracking():
    """Test that ModeMonitor with interp_spec requires track_freq."""

    with pytest.raises(ValidationError, match="tracking"):
        mode_spec_no_track = td.ModeSpec(
            num_modes=2,
            track_freq=None,
            sort_spec=td.ModeSortSpec(track_freq=None),
            interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
        )


# ============================================================================
# Monitor with interp_spec Tests
# ============================================================================


def test_mode_monitor_valid_with_tracking():
    """Test that ModeMonitor validates with tracking enabled."""
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
    )

    monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        name="test",
    )
    assert monitor.mode_spec.interp_spec.num_points == 5
    assert monitor.mode_spec.interp_spec.method == "linear"


def test_mode_solver_monitor_valid_with_tracking():
    """Test that ModeSolverMonitor validates with tracking enabled."""
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
    )

    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        name="test",
    )
    assert monitor.mode_spec.interp_spec.num_points == 5


def test_interp_num_points_less_than_freqs():
    """Test that num_points can be greater than total freqs."""
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=25, method="linear"),
    )

    with AssertLogLevel(None):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
            name="test",
        )


def test_interp_num_points_equal_to_freqs():
    """Test that num_points equal to freqs is not rejected."""
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=20, method="linear"),
    )

    with AssertLogLevel(None):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
            name="test",
        )


def test_interp_spec_none_allowed():
    """Test that interp_spec=None is allowed (no interpolation)."""
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=MODE_SPEC,
        name="test",
    )
    assert monitor.mode_spec.interp_spec is None


def test_interp_deprecated_track_freq_still_works():
    """Test that deprecated track_freq on ModeSpec still enables interpolation."""
    # Using deprecated track_freq instead of sort_spec.track_freq
    with AssertLogLevel("WARNING", contains_str="deprecated"):
        mode_spec = td.ModeSpec(
            num_modes=2,
            track_freq="central",
            interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
        )

    # Should still work since _track_freq property resolves it
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        name="test",
    )
    assert monitor.mode_spec.interp_spec.num_points == 5


# ============================================================================
# ModeSolver with interp_spec Tests
# ============================================================================


def get_simple_sim():
    """Create a simple simulation for ModeSolver tests."""
    return td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 10)),
                medium=td.Medium(permittivity=4.0),
            )
        ],
        run_time=1e-12,
    )


def test_mode_solver_valid_with_tracking():
    """Test that ModeSolver validates with tracking enabled."""
    sim = get_simple_sim()
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
    )
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
    )
    assert solver.mode_spec.interp_spec.num_points == 5
    assert solver.mode_spec.interp_spec.method == "linear"


@td.packaging.disable_local_subpixel
def test_mode_solver_warns_num_points():
    """Test that ModeSolver does not warn when num_points >= num_freqs."""
    sim = get_simple_sim()
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=25, method="linear"),
    )
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    with AssertLogStr(None, excludes_str=["has bounds that extend"]):
        ms = ModeSolver(
            simulation=sim,
            plane=plane,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
        )
    _ = ms.data_raw


def test_mode_solver_interp_spec_none():
    """Test that ModeSolver accepts interp_spec=None."""
    sim = get_simple_sim()
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=FREQS_DENSE,
        mode_spec=MODE_SPEC,
    )
    assert solver.mode_spec.interp_spec is None


# ============================================================================
# ModeSolverData.interp() Tests
# ============================================================================


def get_mode_solver_data():
    """Create a simple ModeSolverData object for testing."""
    from tidy3d.components.data.data_array import GroupIndexDataArray, ModeDispersionDataArray

    from ..test_data.test_data_arrays import SIM, make_scalar_mode_field_data_array
    from ..test_data.test_monitor_data import N_COMPLEX

    freqs = N_COMPLEX.f.data
    num_modes = len(N_COMPLEX.mode_index)
    mode_indices = np.arange(num_modes)
    mode_spec = td.ModeSpec(num_modes=num_modes, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test_monitor",
        colocate=False,
    )

    # Create n_group_raw and dispersion_raw with same shape as n_complex
    n_group_values = 1.5 + 0.1 * np.random.random((len(freqs), num_modes))
    n_group_raw = GroupIndexDataArray(
        n_group_values, coords={"f": freqs, "mode_index": mode_indices}
    )

    dispersion_values = 10.0 + 2.0 * np.random.random((len(freqs), num_modes))
    dispersion_raw = ModeDispersionDataArray(
        dispersion_values, coords={"f": freqs, "mode_index": mode_indices}
    )

    # Create mode data with the right frequencies
    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex", symmetry=False),
        Ey=make_scalar_mode_field_data_array("Ey", symmetry=False),
        Ez=make_scalar_mode_field_data_array("Ez", symmetry=False),
        Hx=make_scalar_mode_field_data_array("Hx", symmetry=False),
        Hy=make_scalar_mode_field_data_array("Hy", symmetry=False),
        Hz=make_scalar_mode_field_data_array("Hz", symmetry=False),
        n_complex=N_COMPLEX,
        n_group_raw=n_group_raw,
        dispersion_raw=dispersion_raw,
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=SIM.discretize_monitor(monitor),
    )
    return mode_data


def test_mode_solver_data_interp_linear():
    """Test linear interpolation on ModeSolverData."""
    mode_data = get_mode_solver_data()

    # Original has 5 frequencies
    assert len(mode_data.monitor.freqs) == 5
    original_num_modes = len(mode_data.n_complex.mode_index)

    # Interpolate to 20 frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp_in_freq(freqs=freqs_dense, method="linear")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 20
    assert data_interp.n_complex.shape[0] == 20

    # Check mode dimension is preserved
    assert len(data_interp.n_complex.mode_index) == original_num_modes

    # Check field components are interpolated
    for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        field_data = getattr(data_interp, field_name)
        assert field_data is not None
        assert field_data.coords["f"].size == 20


def test_mode_solver_data_interp_same_freqs_does_not_mutate_original():
    """Interpolating at stored frequencies should not renormalize the original data in-place."""
    mode_data = get_mode_solver_data()
    ex_before = mode_data.Ex.copy()
    hx_before = mode_data.Hx.copy()

    data_interp = mode_data.interp_in_freq(freqs=mode_data.monitor.freqs, method="linear")

    assert data_interp.Ex is not mode_data.Ex
    assert data_interp.Hx is not mode_data.Hx
    assert np.allclose(mode_data.Ex.values, ex_before.values)
    assert np.allclose(mode_data.Hx.values, hx_before.values)


def test_mode_solver_data_interp_cubic():
    """Test cubic interpolation on ModeSolverData."""
    mode_data = get_mode_solver_data()

    # Need at least 4 frequencies for cubic
    assert len(mode_data.monitor.freqs) >= 4

    # Interpolate to 20 frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp_in_freq(freqs=freqs_dense, method="cubic")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 20
    assert len(data_interp.n_complex.mode_index) == len(mode_data.n_complex.mode_index)


def test_mode_solver_data_interp_poly():
    """Test polynomial interpolation on ModeSolverData."""
    # Create data with frequencies at Chebyshev nodes for optimal polynomial interpolation
    interp_spec = td.ModeInterpSpec.cheb(num_points=5)
    freqs_all = np.linspace(1e14, 2e14, 50)
    freqs_cheb = interp_spec.sampling_points(freqs_all)

    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs_cheb,
        mode_spec=mode_spec,
        name="test_poly",
        colocate=False,
    )

    from ..test_data.test_data_arrays import SIM, make_scalar_mode_field_data_array
    from ..test_data.test_monitor_data import N_COMPLEX

    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex", symmetry=False),
        Ey=make_scalar_mode_field_data_array("Ey", symmetry=False),
        Ez=make_scalar_mode_field_data_array("Ez", symmetry=False),
        Hx=make_scalar_mode_field_data_array("Hx", symmetry=False),
        Hy=make_scalar_mode_field_data_array("Hy", symmetry=False),
        Hz=make_scalar_mode_field_data_array("Hz", symmetry=False),
        n_complex=N_COMPLEX.copy(),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=SIM.discretize_monitor(monitor),
    )

    # Interpolate to 50 frequencies using polynomial method
    data_interp = mode_data.interp_in_freq(freqs=freqs_all, method="poly")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 50
    assert len(data_interp.n_complex.mode_index) == len(N_COMPLEX.mode_index)


def test_mode_solver_data_interp_poly_needs_3_source():
    """Test that polynomial interpolation fails with too few source frequencies."""
    # Create data with only 2 frequencies
    freqs = np.linspace(1e14, 2e14, 2)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test",
    )

    from ..test_data.test_data_arrays import make_scalar_mode_field_data_array
    from ..test_data.test_monitor_data import N_COMPLEX

    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex"),
        n_complex=N_COMPLEX.copy(),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=td.Grid(boundaries=td.Coords(x=[0, 1], y=[0, 1], z=[0, 1])),
    )

    freqs_dense = np.linspace(1e14, 2e14, 10)
    with pytest.raises(td.exceptions.DataError, match="at least 3 source"):
        mode_data.interp_in_freq(freqs=freqs_dense, method="poly")


def test_mode_solver_data_interp_preserves_modes():
    """Test that interpolation preserves mode count."""
    mode_data = get_mode_solver_data()
    original_num_modes = len(mode_data.n_complex.mode_index)

    # Interpolate to different number of frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp_in_freq(freqs=freqs_dense, method="linear")

    # Mode count should be unchanged
    assert len(data_interp.n_complex.mode_index) == original_num_modes


def test_mode_solver_data_interp_includes_n_group_and_dispersion():
    """Test that interpolation includes n_group_raw and dispersion_raw."""
    mode_data = get_mode_solver_data()

    # Verify source data has n_group_raw and dispersion_raw
    assert mode_data.n_group_raw is not None
    assert mode_data.dispersion_raw is not None
    assert mode_data.n_group_raw.shape == (5, len(mode_data.n_complex.mode_index))
    assert mode_data.dispersion_raw.shape == (5, len(mode_data.n_complex.mode_index))

    # Interpolate to 20 frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp_in_freq(freqs=freqs_dense, method="linear")

    # Verify interpolated data has n_group_raw and dispersion_raw
    assert data_interp.n_group_raw is not None
    assert data_interp.dispersion_raw is not None

    # Check shapes are correct
    assert data_interp.n_group_raw.shape == (20, len(data_interp.n_complex.mode_index))
    assert data_interp.dispersion_raw.shape == (20, len(data_interp.n_complex.mode_index))

    # Check frequency coordinates are correct
    assert len(data_interp.n_group_raw.coords["f"]) == 20
    assert len(data_interp.dispersion_raw.coords["f"]) == 20


def test_mode_solver_data_interp_single_frequency():
    """Test that interpolation works with a single target frequency."""
    mode_data = get_mode_solver_data()

    # Original has 5 frequencies
    assert len(mode_data.monitor.freqs) == 5
    original_num_modes = len(mode_data.n_complex.mode_index)

    # Interpolate to a single frequency in the middle of the range
    single_freq = np.array([1.5e14])
    data_interp = mode_data.interp_in_freq(freqs=single_freq, method="linear")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 1
    assert data_interp.n_complex.shape[0] == 1

    # Check mode dimension is preserved
    assert len(data_interp.n_complex.mode_index) == original_num_modes

    # Check field components are interpolated
    for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        field_data = getattr(data_interp, field_name)
        assert field_data is not None
        assert field_data.coords["f"].size == 1
        assert float(field_data.coords["f"].item()) == 1.5e14

    # Check n_group_raw and dispersion_raw if present
    if data_interp.n_group_raw is not None:
        assert data_interp.n_group_raw.shape == (1, original_num_modes)
    if data_interp.dispersion_raw is not None:
        assert data_interp.dispersion_raw.shape == (1, original_num_modes)


def test_mode_solver_data_interp_cubic_needs_4_source():
    """Test that cubic interpolation fails with too few source frequencies."""
    # Create data with only 3 frequencies
    freqs = np.linspace(1e14, 2e14, 3)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test",
    )

    from ..test_data.test_data_arrays import make_scalar_mode_field_data_array
    from ..test_data.test_monitor_data import N_COMPLEX

    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex"),
        n_complex=N_COMPLEX.copy(),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=td.Grid(boundaries=td.Coords(x=[0, 1], y=[0, 1], z=[0, 1])),
    )

    freqs_dense = np.linspace(1e14, 2e14, 10)
    with pytest.raises(td.exceptions.DataError, match="at least 4 source"):
        mode_data.interp_in_freq(freqs=freqs_dense, method="cubic")


def test_mode_solver_data_interp_invalid_method():
    """Test that invalid interpolation method raises error."""
    mode_data = get_mode_solver_data()
    freqs_dense = np.linspace(1e14, 2e14, 10)

    with pytest.raises(td.exceptions.DataError, match="Invalid interpolation method"):
        mode_data.interp_in_freq(freqs=freqs_dense, method="quadratic")


def test_mode_solver_data_interp_extrapolation_warning():
    """Test that extrapolation triggers a warning."""
    mode_data = get_mode_solver_data()

    # Interpolate beyond original range
    freqs_extrap = np.linspace(0.5e14, 2.5e14, 10)

    with AssertLogLevel("WARNING", contains_str="outside original range"):
        mode_data.interp_in_freq(freqs=freqs_extrap, method="linear")


# ============================================================================
# assume_sorted Tests (for source frequencies)
# ============================================================================


@pytest.mark.parametrize("method", ["linear", "cubic", "poly"])
def test_interp_assume_sorted_source_frequencies(method):
    """Test that assume_sorted correctly handles sorted vs unsorted source frequencies.

    This test verifies that:
    - assume_sorted=True works correctly when source frequencies are sorted
    - assume_sorted=False correctly handles unsorted source frequencies
    - Both approaches produce identical interpolation results

    Parameters
    ----------
    method : str
        Interpolation method to test: "linear", "cubic", or "poly"
    """
    from tidy3d.components.data.data_array import ModeIndexDataArray
    from tidy3d.components.data.dataset import FreqDataset

    # Define number of source frequencies based on method requirements
    num_source_points = {"linear": 5, "cubic": 5, "poly": 5}[method]

    # Generate sorted source frequencies using ModeInterpSpec.sampling_points
    # Use Chebyshev sampling for polynomial interpolation for best accuracy
    if method == "poly":
        interp_spec = td.ModeInterpSpec.cheb(num_points=num_source_points)
    else:
        interp_spec = td.ModeInterpSpec.uniform(num_points=num_source_points, method=method)
    freqs_full_range = np.linspace(1e14, 2e14, 100)  # Full frequency range
    freqs_source_sorted = interp_spec.sampling_points(freqs_full_range)

    # Create unsorted version by shuffling
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    shuffle_indices = np.arange(len(freqs_source_sorted))
    rng.shuffle(shuffle_indices)
    freqs_source_unsorted = freqs_source_sorted[shuffle_indices]

    # Create test data values
    mode_indices = np.arange(2)
    data_values = (1.5 + 0.1j) * np.random.random((num_source_points, 2))

    # Create dataset with SORTED source frequencies
    data_sorted = ModeIndexDataArray(
        data_values.copy(), coords={"f": freqs_source_sorted, "mode_index": mode_indices}
    )

    # Create dataset with UNSORTED source frequencies
    # Need to reorder data values to match the shuffled frequencies
    data_unsorted = ModeIndexDataArray(
        data_values[shuffle_indices],
        coords={"f": freqs_source_unsorted, "mode_index": mode_indices},
    )

    # Define target frequencies for interpolation
    freqs_target = np.linspace(1e14, 2e14, 50)

    # Interpolate with sorted source frequencies and assume_sorted=True
    result_sorted = FreqDataset._interp_dataarray_in_freq(
        data_sorted, freqs_target, method=method, assume_sorted=True
    )

    # Interpolate with unsorted source frequencies and assume_sorted=False
    # This should internally sort the data before interpolating
    result_unsorted = FreqDataset._interp_dataarray_in_freq(
        data_unsorted, freqs_target, method=method, assume_sorted=False
    )

    # Both approaches should produce identical results
    assert np.allclose(result_sorted.values, result_unsorted.values, rtol=1e-10), (
        f"{method} interpolation: sorted and unsorted source frequencies "
        "should produce identical results"
    )

    # Verify the interpolation actually happened correctly
    assert result_sorted.shape == (50, 2)
    assert len(result_sorted.coords["f"]) == 50
    assert np.allclose(result_sorted.coords["f"].values, freqs_target)


# ============================================================================
# ModeSolver Integration Tests (Phase 5)
# ============================================================================


@pytest.mark.parametrize("method", ["linear", "cubic", "poly"])
def test_mode_solver_with_interp(method):
    """Test that ModeSolver uses interpolation when interp_spec is provided.

    With reduce_data=False (default), data_raw returns automatically interpolated data
    with all requested frequencies.
    """
    sim = get_simple_sim()

    # Create solver with 10 frequencies, reduce_data=False (default)
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method=method, reduce_data=False),
    )

    solver_with_interp = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
    )

    # The solver should have the original 10 frequencies
    assert len(solver_with_interp.freqs) == 10

    # With reduce_data=False, data_raw automatically interpolates and returns all 10 frequencies
    data = solver_with_interp.data_raw
    assert len(data.monitor.freqs) == 10
    assert data.n_complex.shape == (10, 2)
    assert data.monitor.mode_spec.interp_spec is None


@pytest.mark.parametrize("method", ["linear", "cubic", "poly"])
def test_mode_solver_creates_reduced_freqs(method):
    """Test that solver creates correct reduced frequency set internally."""
    sim = get_simple_sim()

    # Create solver with 20 frequencies
    freqs = np.linspace(1e14, 2e14, 20)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method=method, reduce_data=True),
    )

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
    )

    # The returned data should have 5 frequencies
    data = solver.data_raw
    assert len(data.monitor.freqs) == 20
    assert len(data.monitor._stored_freqs) == 5
    assert data.n_complex.shape == (5, 2)

    interpolated_data = data.interpolated_copy
    assert len(interpolated_data.monitor.freqs) == 20
    assert len(interpolated_data.monitor._stored_freqs) == 20
    assert interpolated_data.n_complex.shape == (20, 2)


def test_mode_solver_without_interp_returns_full_data():
    """Test that solver without interp_spec computes at all frequencies."""
    sim = get_simple_sim()

    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        # No interpolation
    )

    data = solver.data_raw
    assert len(data.monitor.freqs) == 10
    assert data.n_complex.shape == (10, 2)
    assert data.monitor.mode_spec.interp_spec is None


def test_mode_solver_with_reduce_data_but_small_num_freqs():
    """Test that ModeSolver with reduce_data=True but small number of frequencies
    returns data at the original frequencies instead of the reduced frequencies.
    """
    sim = get_simple_sim()

    # Create solver with 2 fsrequencies but reduce_data=True
    freqs = np.linspace(1e14, 2e14, 2)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear", reduce_data=True),
    )

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
    )

    data = solver.data_raw
    assert len(data.monitor._stored_freqs) == 2
    assert len(data.monitor.freqs) == 2
    assert data.n_complex.shape == (2, 2)
    assert data.monitor.mode_spec.interp_spec is None

    assert data.interpolated_copy == data


@pytest.mark.parametrize(
    "num_freqs,num_points,reduce_data,expected_stored_len",
    [
        # No interp_spec (num_points=None)
        (10, None, None, 10),
        # interp_spec with reduce_data=False
        (10, 5, False, 10),
        (10, 8, False, 10),
        # interp_spec with reduce_data=True and num_points < len(monitor.freqs)
        (20, 5, True, 5),
        (20, 10, True, 10),
        # interp_spec with reduce_data=True and num_points >= len(monitor.freqs)
        (10, 10, True, 10),
        (10, 15, True, 10),
        (5, 10, True, 5),
    ],
)
@pytest.mark.parametrize("rf", [False, True])
def test_mode_solver_data_stored_freqs(num_freqs, num_points, reduce_data, expected_stored_len, rf):
    """Test that _stored_freqs in ModeSolverData is correct based on interp_spec.

    Cases tested:
    1. No interp_spec: _stored_freqs matches monitor.freqs
    2. interp_spec with reduce_data=False: _stored_freqs matches monitor.freqs
    3. interp_spec with reduce_data=True and num_points < len(monitor.freqs):
       len(_stored_freqs) == num_points
    4. interp_spec with reduce_data=True and num_points >= len(monitor.freqs):
       _stored_freqs matches monitor.freqs
    """
    from tidy3d.components.data.data_array import ModeIndexDataArray

    from ..test_data.test_data_arrays import SIM, make_scalar_mode_field_data_array

    freqs = np.linspace(1e14, 2e14, num_freqs)

    # Create mode_spec based on parameters
    if num_points is None:
        # No interp_spec
        mode_spec = td.ModeSpec(
            num_modes=2,
            sort_spec=td.ModeSortSpec(track_freq="central"),
        )
    else:
        # With interp_spec
        mode_spec = td.ModeSpec(
            num_modes=2,
            sort_spec=td.ModeSortSpec(track_freq="central"),
            interp_spec=td.ModeInterpSpec.uniform(
                num_points=num_points, method="linear", reduce_data=reduce_data
            ),
        )

    # Create monitor
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test_monitor",
        colocate=False,
    )

    # Create n_complex with the expected stored frequencies
    mode_indices = np.arange(2)
    n_complex_values = (1.5 + 0.1j) * np.ones((expected_stored_len, 2))
    n_complex = ModeIndexDataArray(
        n_complex_values,
        coords={"f": np.linspace(1e14, 2e14, expected_stored_len), "mode_index": mode_indices},
    )

    # Create ModeSolverData
    data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex", symmetry=False),
        Ey=make_scalar_mode_field_data_array("Ey", symmetry=False),
        Ez=make_scalar_mode_field_data_array("Ez", symmetry=False),
        Hx=make_scalar_mode_field_data_array("Hx", symmetry=False),
        Hy=make_scalar_mode_field_data_array("Hy", symmetry=False),
        Hz=make_scalar_mode_field_data_array("Hz", symmetry=False),
        n_complex=n_complex,
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=SIM.discretize_monitor(monitor),
    )

    if rf:
        mode_spec = td.MicrowaveModeSpec(**mode_spec.model_dump(exclude={"type"}))
        monitor = td.MicrowaveModeSolverMonitor(
            **monitor.model_dump(exclude={"type", "mode_spec"}), mode_spec=mode_spec
        )
        data = td.ModeSolverData(**data.model_dump(exclude={"type", "monitor"}), monitor=monitor)

    # Check _stored_freqs length
    assert len(data.monitor._stored_freqs) == expected_stored_len, (
        f"Expected _stored_freqs length {expected_stored_len}, "
        f"got {len(data.monitor._stored_freqs)}"
    )

    # Check that monitor.freqs always matches original freqs
    assert len(data.monitor.freqs) == num_freqs
    assert np.allclose(data.monitor.freqs, freqs)

    # Check data shape matches _stored_freqs
    assert data.n_complex.shape[0] == expected_stored_len


# ============================================================================
# Monitor Integration Tests (Phase 6)
# ============================================================================


def test_mode_monitor_with_interp_spec():
    """Test that ModeMonitor can be created with interp_spec."""
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
    )

    monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_monitor",
    )

    assert monitor.mode_spec.interp_spec is not None
    assert monitor.mode_spec.interp_spec.num_points == 3
    assert monitor.mode_spec.interp_spec.method == "linear"


def test_mode_solver_monitor_with_interp_spec():
    """Test that ModeSolverMonitor can be created with interp_spec."""
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=4, method="cubic"),
    )

    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_solver_monitor",
    )

    assert monitor.mode_spec.interp_spec is not None
    assert monitor.mode_spec.interp_spec.num_points == 4
    assert monitor.mode_spec.interp_spec.method == "cubic"


def test_mode_monitor_warns_redundant_num_points():
    """Test no warning when num_points >= number of frequencies in ModeMonitor."""
    freqs = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=5, method="linear"),
    )

    with AssertLogLevel(None):
        td.ModeMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec,
            name="test",
        )


def test_mode_solver_monitor_warns_redundant_num_points():
    """Test no warning when num_points >= number of frequencies in ModeSolverMonitor."""
    freqs = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=6, method="linear"),
    )

    with AssertLogLevel(None):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec,
            name="test",
        )


def test_mode_monitor_interp_spec_none():
    """Test that ModeMonitor works without interp_spec."""
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test",
    )

    assert monitor.mode_spec.interp_spec is None


def test_mode_solver_sampling_freqs():
    """Test that ModeSolver computes modes at the correct frequencies."""
    sim = get_simple_sim()
    freqs = np.linspace(1e14, 2e14, 10)
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    # With interp_spec.num_points < len(freqs)
    mode_spec_true = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
    )
    solver_true = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=freqs,
        mode_spec=mode_spec_true,
    )
    assert len(solver_true._sampling_freqs) == 3

    # With interp_spec.num_points < len(freqs) + group_index_step
    mode_spec_true = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
        group_index_step=True,
    )
    solver_true = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=freqs,
        mode_spec=mode_spec_true,
    )
    assert len(solver_true._sampling_freqs) == 3 * 3

    # With interp_spec.num_points > len(freqs)
    mode_spec_true = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
    )
    solver_true = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=[1e14],
        mode_spec=mode_spec_true,
    )
    assert len(solver_true._sampling_freqs) == 1

    # With interp_spec.num_points > len(freqs) + group_index_step
    mode_spec_true = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
        group_index_step=True,
    )
    solver_true = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=[1e14, 2e14],
        mode_spec=mode_spec_true,
    )
    assert len(solver_true._sampling_freqs) == 2 * 3

    # Without interp_spec
    mode_spec_none = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    solver_none = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=freqs,
        mode_spec=mode_spec_none,
    )
    assert len(solver_none._sampling_freqs) == 10

    # Without interp_spec, with group_index_step
    mode_spec_none = td.ModeSpec(
        num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"), group_index_step=True
    )
    solver_none = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=freqs,
        mode_spec=mode_spec_none,
    )
    assert len(solver_none._sampling_freqs) == 10 * 3


# ============================================================================
# WavePort interp_spec Tests
# ============================================================================


def make_wave_port(num_interp_points=3, method="linear"):
    """Make a WavePort."""
    from tidy3d.plugins.smatrix.ports.wave import WavePort

    return WavePort(
        center=(0, 0, 0),
        size=(1, 1, 0),
        direction="+",
        name="port1",
        mode_spec=td.MicrowaveModeSpec(
            num_modes=1,
            sort_spec=td.ModeSortSpec(track_freq="central"),
            interp_spec=td.ModeInterpSpec.uniform(num_points=num_interp_points, method=method),
        ),
    )


def test_wave_port_to_monitors_propagates_default_interp_spec():
    """Test that WavePort.to_monitors() propagates default interp_spec to ModeMonitor."""

    port = make_wave_port(num_interp_points=3, method="linear")

    freqs = np.linspace(1e14, 2e14, 20)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.mode_spec.interp_spec is not None
    assert monitor.mode_spec.interp_spec.num_points == 3
    assert monitor.mode_spec.interp_spec.method == "linear"


def test_wave_port_to_monitors_propagates_custom_interp_spec():
    """Test that WavePort.to_monitors() propagates custom interp_spec to ModeMonitor."""
    custom_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.cheb(num_points=8),
    )
    port = make_wave_port().updated_copy(mode_spec=custom_mode_spec)

    freqs = np.linspace(1e14, 2e14, 50)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.mode_spec.interp_spec is not None
    assert monitor.mode_spec.interp_spec.num_points == 8
    assert monitor.mode_spec.interp_spec.method == "poly"


def test_wave_port_to_monitors_propagates_none_interp_spec():
    """Test that WavePort.to_monitors() propagates interp_spec=None to ModeMonitor."""
    mode_spec_no_interp = td.MicrowaveModeSpec(
        num_modes=1, sort_spec=td.ModeSortSpec(track_freq="central"), interp_spec=None
    )
    port = make_wave_port().updated_copy(mode_spec=mode_spec_no_interp)

    freqs = np.linspace(1e14, 2e14, 20)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.mode_spec.interp_spec is None


# ============================================================================
# Placeholder tests for future phases
# ============================================================================
