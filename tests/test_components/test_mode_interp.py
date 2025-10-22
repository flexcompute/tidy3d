"""Tests for mode frequency interpolation."""

from __future__ import annotations

import numpy as np
import pydantic.v1 as pydantic
import pytest

import tidy3d as td

from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.smatrix.ports.wave import DEFAULT_WAVE_PORT_INTERP_SPEC

from ..test_data.test_data_arrays import MODE_SPEC, SIZE_2D
from ..utils import AssertLogLevel

# Shared test constants
FREQS_DENSE = np.linspace(1e14, 2e14, 20)


# ============================================================================
# ModeInterpSpec Tests
# ============================================================================


def test_interp_spec_valid_linear():
    """Test creating valid ModeInterpSpec with linear interpolation."""
    spec = td.ModeInterpSpec(num_points=5, method="linear")
    assert spec.num_points == 5
    assert spec.method == "linear"


def test_interp_spec_valid_cubic():
    """Test creating valid ModeInterpSpec with cubic interpolation."""
    spec = td.ModeInterpSpec(num_points=10, method="cubic")
    assert spec.num_points == 10
    assert spec.method == "cubic"


def test_interp_spec_default_method():
    """Test that default method is 'linear'."""
    spec = td.ModeInterpSpec(num_points=5)
    assert spec.method == "linear"


def test_interp_spec_cubic_needs_4_points():
    """Test that cubic interpolation requires at least 4 points."""
    with pytest.raises(pydantic.ValidationError, match="Cubic interpolation requires at least 4"):
        td.ModeInterpSpec(num_points=3, method="cubic")


def test_interp_spec_valid_cheb():
    """Test creating valid ModeInterpSpec with Chebyshev interpolation."""
    spec = td.ModeInterpSpec(num_points=10, method="cheb")
    assert spec.num_points == 10
    assert spec.method == "cheb"


def test_interp_spec_cheb_needs_3_points():
    """Test that Chebyshev interpolation requires at least 3 points."""
    with pytest.raises(
        pydantic.ValidationError, match="Chebyshev interpolation requires at least 3"
    ):
        td.ModeInterpSpec(num_points=2, method="cheb")


def test_interp_spec_sampling_points_linear():
    """Test sampling_points for linear interpolation."""
    spec = td.ModeInterpSpec(num_points=5, method="linear")
    freqs = np.linspace(1e14, 2e14, 100)
    sampling = spec.sampling_points(freqs)

    assert len(sampling) == 5
    assert np.isclose(sampling[0], 1e14)
    assert np.isclose(sampling[-1], 2e14)
    # Check uniform spacing
    diffs = np.diff(sampling)
    assert np.allclose(diffs, diffs[0])


def test_interp_spec_sampling_points_cheb():
    """Test sampling_points for Chebyshev interpolation."""
    spec = td.ModeInterpSpec(num_points=5, method="cheb")
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


def test_interp_spec_min_2_points():
    """Test that at least 2 points are required."""
    with pytest.raises(pydantic.ValidationError):
        td.ModeInterpSpec(num_points=1, method="linear")


def test_interp_spec_positive_points():
    """Test that num_points must be positive."""
    with pytest.raises(pydantic.ValidationError):
        td.ModeInterpSpec(num_points=0, method="linear")

    with pytest.raises(pydantic.ValidationError):
        td.ModeInterpSpec(num_points=-5, method="linear")


def test_interp_spec_invalid_method():
    """Test that invalid interpolation method is rejected."""
    with pytest.raises(pydantic.ValidationError):
        td.ModeInterpSpec(num_points=5, method="quadratic")


# ============================================================================
# Monitor with interp_spec Tests
# ============================================================================


def test_mode_monitor_requires_tracking():
    """Test that ModeMonitor with interp_spec requires track_freq."""
    mode_spec_no_track = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq=None))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    with pytest.raises(pydantic.ValidationError, match="tracking"):
        td.ModeMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec_no_track,
            interp_spec=interp_spec,
            name="test",
        )


def test_mode_monitor_valid_with_tracking():
    """Test that ModeMonitor validates with tracking enabled."""
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
        name="test",
    )
    assert monitor.interp_spec.num_points == 5
    assert monitor.interp_spec.method == "linear"


def test_mode_solver_monitor_requires_tracking():
    """Test that ModeSolverMonitor with interp_spec requires track_freq."""
    mode_spec_no_track = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq=None))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    with pytest.raises(pydantic.ValidationError, match="tracking"):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec_no_track,
            interp_spec=interp_spec,
            name="test",
        )


def test_mode_solver_monitor_valid_with_tracking():
    """Test that ModeSolverMonitor validates with tracking enabled."""
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
        name="test",
    )
    assert monitor.interp_spec.num_points == 5


def test_interp_num_points_less_than_freqs():
    """Test that num_points must be less than total freqs."""
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=25, method="linear")

    with AssertLogLevel("WARNING", contains_str="num_points"):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
            interp_spec=interp_spec,
            name="test",
        )


def test_interp_num_points_equal_to_freqs():
    """Test that num_points equal to freqs is rejected."""
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=20, method="linear")

    with AssertLogLevel("WARNING", contains_str="num_points"):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
            interp_spec=interp_spec,
            name="test",
        )


def test_interp_spec_none_allowed():
    """Test that interp_spec=None is allowed (no interpolation)."""
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=MODE_SPEC,
        interp_spec=None,
        name="test",
    )
    assert monitor.interp_spec is None


def test_interp_deprecated_track_freq_still_works():
    """Test that deprecated track_freq on ModeSpec still enables interpolation."""
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    # Using deprecated track_freq instead of sort_spec.track_freq
    with AssertLogLevel("WARNING", contains_str="deprecated"):
        mode_spec = td.ModeSpec(num_modes=2, track_freq="central")

    # Should still work since _track_freq property resolves it
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
        name="test",
    )
    assert monitor.interp_spec.num_points == 5


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


def test_mode_solver_requires_tracking():
    """Test that ModeSolver with interp_spec requires track_freq."""
    sim = get_simple_sim()
    mode_spec_no_track = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq=None))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    with pytest.raises(pydantic.ValidationError, match="tracking"):
        ModeSolver(
            simulation=sim,
            plane=plane,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec_no_track,
            interp_spec=interp_spec,
        )


def test_mode_solver_valid_with_tracking():
    """Test that ModeSolver validates with tracking enabled."""
    sim = get_simple_sim()
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=FREQS_DENSE,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )
    assert solver.interp_spec.num_points == 5
    assert solver.interp_spec.method == "linear"


def test_mode_solver_warns_num_points():
    """Test that ModeSolver warns when num_points >= num_freqs."""
    sim = get_simple_sim()
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=25, method="linear")
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    with AssertLogLevel("WARNING", contains_str="num_points"):
        ModeSolver(
            simulation=sim,
            plane=plane,
            freqs=FREQS_DENSE,
            mode_spec=mode_spec,
            interp_spec=interp_spec,
        )


def test_mode_solver_interp_spec_none():
    """Test that ModeSolver accepts interp_spec=None."""
    sim = get_simple_sim()
    plane = td.Box(center=(0, 0, 0), size=SIZE_2D)

    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        freqs=FREQS_DENSE,
        mode_spec=MODE_SPEC,
        interp_spec=None,
    )
    assert solver.interp_spec is None


# ============================================================================
# ModeSolverData.interp() Tests
# ============================================================================


def get_mode_solver_data():
    """Create a simple ModeSolverData object for testing."""
    from ..test_data.test_data_arrays import (
        make_scalar_mode_field_data_array,
    )
    from ..test_data.test_monitor_data import N_COMPLEX

    freqs = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        name="test_monitor",
    )

    # Create mode data with the right frequencies
    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex"),
        Ey=make_scalar_mode_field_data_array("Ey"),
        Ez=make_scalar_mode_field_data_array("Ez"),
        Hx=make_scalar_mode_field_data_array("Hx"),
        Hy=make_scalar_mode_field_data_array("Hy"),
        Hz=make_scalar_mode_field_data_array("Hz"),
        n_complex=N_COMPLEX.copy(),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=td.Grid(boundaries=td.Coords(x=[0, 1], y=[0, 1], z=[0, 1])),
    )
    return mode_data


def test_mode_solver_data_interp_linear():
    """Test linear interpolation on ModeSolverData."""
    mode_data = get_mode_solver_data()

    # Original has 5 frequencies
    assert len(mode_data.monitor.freqs) == 5
    original_num_modes = mode_data.n_complex.shape[1]

    # Interpolate to 20 frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp(freqs=freqs_dense, method="linear")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 20
    assert data_interp.n_complex.shape[0] == 20

    # Check mode dimension is preserved
    assert data_interp.n_complex.shape[1] == original_num_modes

    # Check field components are interpolated
    for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        field_data = getattr(data_interp, field_name)
        assert field_data is not None
        assert field_data.coords["f"].size == 20


def test_mode_solver_data_interp_cubic():
    """Test cubic interpolation on ModeSolverData."""
    mode_data = get_mode_solver_data()

    # Need at least 4 frequencies for cubic
    assert len(mode_data.monitor.freqs) >= 4

    # Interpolate to 20 frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp(freqs=freqs_dense, method="cubic")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 20
    assert data_interp.n_complex.shape[0] == 20


def test_mode_solver_data_interp_cheb():
    """Test Chebyshev interpolation on ModeSolverData."""
    # Create data with frequencies at Chebyshev nodes
    interp_spec = td.ModeInterpSpec(num_points=5, method="cheb")
    freqs_all = np.linspace(1e14, 2e14, 50)
    freqs_cheb = interp_spec.sampling_points(freqs_all)

    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs_cheb,
        mode_spec=mode_spec,
        name="test_cheb",
    )

    from ..test_data.test_data_arrays import make_scalar_mode_field_data_array
    from ..test_data.test_monitor_data import N_COMPLEX

    mode_data = td.ModeSolverData(
        monitor=monitor,
        Ex=make_scalar_mode_field_data_array("Ex"),
        Ey=make_scalar_mode_field_data_array("Ey"),
        Ez=make_scalar_mode_field_data_array("Ez"),
        Hx=make_scalar_mode_field_data_array("Hx"),
        Hy=make_scalar_mode_field_data_array("Hy"),
        Hz=make_scalar_mode_field_data_array("Hz"),
        n_complex=N_COMPLEX.copy(),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=td.Grid(boundaries=td.Coords(x=[0, 1], y=[0, 1], z=[0, 1])),
    )

    # Interpolate to 50 frequencies
    data_interp = mode_data.interp(freqs=freqs_all, method="cheb")

    # Check frequency dimension
    assert len(data_interp.monitor.freqs) == 50
    assert data_interp.n_complex.shape[0] == 50


def test_mode_solver_data_interp_cheb_needs_3_source():
    """Test that Chebyshev interpolation fails with too few source frequencies."""
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
        mode_data.interp(freqs=freqs_dense, method="cheb")


def test_mode_solver_data_interp_cheb_validates_nodes():
    """Test that Chebyshev interpolation validates source frequencies are Chebyshev nodes."""
    # Create data with uniform (not Chebyshev) nodes
    freqs_uniform = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs_uniform,
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
    with pytest.raises(td.exceptions.DataError, match="must be at Chebyshev nodes"):
        mode_data.interp(freqs=freqs_dense, method="cheb")


def test_mode_solver_data_interp_preserves_modes():
    """Test that interpolation preserves mode count."""
    mode_data = get_mode_solver_data()
    original_num_modes = mode_data.n_complex.shape[1]

    # Interpolate to different number of frequencies
    freqs_dense = np.linspace(mode_data.monitor.freqs[0], mode_data.monitor.freqs[-1], 20)
    data_interp = mode_data.interp(freqs=freqs_dense, method="linear")

    # Mode count should be unchanged
    assert data_interp.n_complex.shape[1] == original_num_modes


def test_mode_solver_data_interp_too_few_target_freqs():
    """Test that interpolation fails with too few target frequencies."""
    mode_data = get_mode_solver_data()

    with pytest.raises(td.exceptions.DataError, match="fewer than 2"):
        mode_data.interp(freqs=[1e14], method="linear")


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
        mode_data.interp(freqs=freqs_dense, method="cubic")


def test_mode_solver_data_interp_invalid_method():
    """Test that invalid interpolation method raises error."""
    mode_data = get_mode_solver_data()
    freqs_dense = np.linspace(1e14, 2e14, 10)

    with pytest.raises(td.exceptions.DataError, match="Invalid interpolation method"):
        mode_data.interp(freqs=freqs_dense, method="quadratic")


def test_mode_solver_data_interp_extrapolation_warning():
    """Test that extrapolation triggers a warning."""
    mode_data = get_mode_solver_data()

    # Interpolate beyond original range
    freqs_extrap = np.linspace(0.5e14, 2.5e14, 10)

    with AssertLogLevel("WARNING", contains_str="outside original range"):
        mode_data.interp(freqs=freqs_extrap, method="linear")


# ============================================================================
# ModeSolver Integration Tests (Phase 5)
# ============================================================================


def test_mode_solver_with_interp():
    """Test that ModeSolver uses interpolation when interp_spec is provided."""
    sim = get_simple_sim()

    # Create solver with 10 frequencies
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # Create solver with interpolation: compute at 3 frequencies, interpolate to 10
    interp_spec = td.ModeInterpSpec(num_points=3, method="linear")

    solver_with_interp = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )

    # The solver should have the original 10 frequencies
    assert len(solver_with_interp.freqs) == 10

    # The returned data should have 10 frequencies
    data = solver_with_interp.data_raw
    assert len(data.monitor.freqs) == 10
    assert data.n_complex.shape[0] == 10


def test_mode_solver_creates_reduced_freqs():
    """Test that solver creates correct reduced frequency set internally."""
    sim = get_simple_sim()

    # Create solver with 20 frequencies
    freqs = np.linspace(1e14, 2e14, 20)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # Compute at 5 frequencies, interpolate to 20
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )

    # The returned data should have all 20 frequencies
    data = solver.data_raw
    assert len(data.monitor.freqs) == 20

    # The effective indices should be properly interpolated
    assert data.n_complex.shape[0] == 20


def test_mode_solver_interp_preserves_num_modes():
    """Test that interpolation preserves the number of modes."""
    sim = get_simple_sim()

    freqs = np.linspace(1e14, 2e14, 15)
    mode_spec = td.ModeSpec(num_modes=3, sort_spec=td.ModeSortSpec(track_freq="central"))

    interp_spec = td.ModeInterpSpec(num_points=4, method="linear")

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )

    data = solver.data_raw

    # Should have 3 modes at each of 15 frequencies
    assert data.n_complex.shape == (15, 3)


def test_mode_solver_interp_cubic():
    """Test that ModeSolver works with cubic interpolation."""
    sim = get_simple_sim()

    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # Cubic interpolation requires at least 4 points
    interp_spec = td.ModeInterpSpec(num_points=4, method="cubic")

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )

    data = solver.data_raw
    assert len(data.monitor.freqs) == 10
    assert data.n_complex.shape[0] == 10


def test_mode_solver_interp_cheb():
    """Test that ModeSolver works with Chebyshev interpolation."""
    sim = get_simple_sim()

    freqs = np.linspace(1e14, 2e14, 20)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # Chebyshev interpolation requires at least 3 points
    interp_spec = td.ModeInterpSpec(num_points=5, method="cheb")

    solver = ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=SIZE_2D),
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
    )

    data = solver.data_raw
    assert len(data.monitor.freqs) == 20
    assert data.n_complex.shape[0] == 20


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
        interp_spec=None,  # No interpolation
    )

    data = solver.data_raw
    assert len(data.monitor.freqs) == 10
    assert data.n_complex.shape[0] == 10


# ============================================================================
# Monitor Integration Tests (Phase 6)
# ============================================================================


def test_mode_monitor_with_interp_spec():
    """Test that ModeMonitor can be created with interp_spec."""
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=3, method="linear")

    monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
        name="mode_monitor",
    )

    assert monitor.interp_spec is not None
    assert monitor.interp_spec.num_points == 3
    assert monitor.interp_spec.method == "linear"


def test_mode_solver_monitor_with_interp_spec():
    """Test that ModeSolverMonitor can be created with interp_spec."""
    freqs = np.linspace(1e14, 2e14, 10)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))
    interp_spec = td.ModeInterpSpec(num_points=4, method="cubic")

    monitor = td.ModeSolverMonitor(
        center=(0, 0, 0),
        size=SIZE_2D,
        freqs=freqs,
        mode_spec=mode_spec,
        interp_spec=interp_spec,
        name="mode_solver_monitor",
    )

    assert monitor.interp_spec is not None
    assert monitor.interp_spec.num_points == 4
    assert monitor.interp_spec.method == "cubic"


def test_mode_monitor_interp_requires_tracking():
    """Test that ModeMonitor with interp_spec requires frequency tracking."""
    freqs = np.linspace(1e14, 2e14, 10)

    # Without tracking
    mode_spec_no_track = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq=None),  # No tracking
    )
    interp_spec = td.ModeInterpSpec(num_points=3, method="linear")

    with pytest.raises(pydantic.ValidationError, match="requires mode tracking to be enabled"):
        td.ModeMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec_no_track,
            interp_spec=interp_spec,
            name="test",
        )


def test_mode_solver_monitor_interp_requires_tracking():
    """Test that ModeSolverMonitor with interp_spec requires frequency tracking."""
    freqs = np.linspace(1e14, 2e14, 10)

    # Without tracking
    mode_spec_no_track = td.ModeSpec(
        num_modes=2,
        sort_spec=td.ModeSortSpec(track_freq=None),  # No tracking
    )
    interp_spec = td.ModeInterpSpec(num_points=3, method="linear")

    with pytest.raises(pydantic.ValidationError, match="requires mode tracking to be enabled"):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec_no_track,
            interp_spec=interp_spec,
            name="test",
        )


def test_mode_monitor_warns_redundant_num_points():
    """Test warning when num_points >= number of frequencies in ModeMonitor."""
    freqs = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # num_points >= len(freqs) should trigger warning
    interp_spec = td.ModeInterpSpec(num_points=5, method="linear")

    with AssertLogLevel("WARNING", contains_str="greater than or equal"):
        td.ModeMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec,
            interp_spec=interp_spec,
            name="test",
        )


def test_mode_solver_monitor_warns_redundant_num_points():
    """Test warning when num_points >= number of frequencies in ModeSolverMonitor."""
    freqs = np.linspace(1e14, 2e14, 5)
    mode_spec = td.ModeSpec(num_modes=2, sort_spec=td.ModeSortSpec(track_freq="central"))

    # num_points >= len(freqs) should trigger warning
    interp_spec = td.ModeInterpSpec(num_points=6, method="linear")

    with AssertLogLevel("WARNING", contains_str="greater than or equal"):
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=SIZE_2D,
            freqs=freqs,
            mode_spec=mode_spec,
            interp_spec=interp_spec,
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
        interp_spec=None,
        name="test",
    )

    assert monitor.interp_spec is None


# ============================================================================
# WavePort interp_spec Tests
# ============================================================================


def make_wave_port():
    """Make a WavePort."""
    from tidy3d.components.microwave.path_integrals.integrals.current import (
        AxisAlignedCurrentIntegral,
    )
    from tidy3d.plugins.smatrix.ports.wave import WavePort

    return WavePort(
        center=(0, 0, 0),
        size=(1, 1, 0),
        direction="+",
        name="port1",
        current_integral=AxisAlignedCurrentIntegral(
            center=(0, 0, 0),
            size=(1, 1, 0),
            sign="+",
            extrapolate_to_endpoints=True,
            snap_contour_to_grid=True,
        ),
    )


def test_wave_port_to_monitors_propagates_default_interp_spec():
    """Test that WavePort.to_monitors() propagates default interp_spec to ModeMonitor."""

    port = make_wave_port()

    freqs = np.linspace(1e14, 2e14, 20)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.interp_spec is not None
    assert monitor.interp_spec.num_points == DEFAULT_WAVE_PORT_INTERP_SPEC.num_points
    assert monitor.interp_spec.method == DEFAULT_WAVE_PORT_INTERP_SPEC.method


def test_wave_port_to_monitors_propagates_custom_interp_spec():
    """Test that WavePort.to_monitors() propagates custom interp_spec to ModeMonitor."""
    custom_interp = td.ModeInterpSpec(num_points=8, method="cheb")
    port = make_wave_port().updated_copy(interp_spec=custom_interp)

    freqs = np.linspace(1e14, 2e14, 50)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.interp_spec is not None
    assert monitor.interp_spec.num_points == 8
    assert monitor.interp_spec.method == "cheb"


def test_wave_port_to_monitors_propagates_none_interp_spec():
    """Test that WavePort.to_monitors() propagates interp_spec=None to ModeMonitor."""
    port = make_wave_port().updated_copy(interp_spec=None)

    freqs = np.linspace(1e14, 2e14, 20)
    monitors = port.to_monitors(freqs=freqs)

    assert len(monitors) == 1
    monitor = monitors[0]
    assert isinstance(monitor, td.ModeMonitor)
    assert monitor.interp_spec is None


# ============================================================================
# Placeholder tests for future phases
# ============================================================================
