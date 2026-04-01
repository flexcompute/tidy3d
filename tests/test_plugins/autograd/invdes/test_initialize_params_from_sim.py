from __future__ import annotations

import autograd.numpy as np
import pytest
from scipy.ndimage import zoom

import tidy3d as td
from tidy3d.plugins.autograd.invdes.parametrizations import initialize_params_from_simulation

from ....utils import AssertLogLevel


def _loss_like_impl(sim: td.Simulation, design_box: td.Box, p: np.ndarray, param_to_structure):
    """Compute a similar loss to the initializer's, for verification only.

    Uses simple zoom-based interpolation of the base permittivity to the parameter grid.
    """
    eps_base = sim.epsilon(box=design_box).data
    eps_param = param_to_structure(p).medium.permittivity.data
    factors = np.array(eps_param.shape) / np.array(eps_base.shape)
    eps_base_interp = zoom(eps_base, factors, order=1)

    denom = np.sqrt(np.sum(eps_base_interp.real**2 + eps_base_interp.imag**2))
    denom = 1.0 if denom == 0 else denom

    res = eps_base_interp - eps_param
    return 0.5 * np.sum(res.real**2 + res.imag**2) / denom


def _make_centered_linear_structure_case(shape: tuple[int, int]):
    """Construct a centered 2D initialization case with linear epsilon mapping."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps3d = (1.0 + 3.0 * p).reshape((*p.shape, 1))
        return td.Structure.from_permittivity_array(geometry=box, eps_data=eps3d)

    params0 = np.full(shape, 0.5)
    return sim, param_to_structure, params0


def test_initialize_params_from_simulation_basic_known_optimum():
    """2D, uniform base epsilon: converges near p*=0 with shape+bounds respected."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    def param_to_structure(p: np.ndarray) -> td.Structure:
        # Linear map: eps = 1 + 3 p  => optimum at p* = 0 for base eps=1
        eps = 1.0 + 3.0 * p
        eps3d = eps.reshape((*p.shape, 1))
        return td.Structure.from_permittivity_array(geometry=box, eps_data=eps3d)

    params0 = np.full((5, 5), 0.7)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        maxiter=25,
    )

    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)
    # known optimum is p ~= 0 everywhere
    assert np.allclose(params, 0.0, atol=1e-3)


def test_outside_handling_extrapolate_runs_and_bounds():
    """Extreme design coords: 'extrapolate' runs, shape and bounds respected."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    # design coords far outside in x to exercise extrapolation path
    coords_x = np.array([-1e6, 0.0, 1e6])
    coords_y = np.linspace(box.center[1] - 0.5, box.center[1] + 0.5, 3)
    coords_z = np.array([0.0])

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps = 1.0 + 3.0 * p
        eps3d = eps.reshape((*p.shape, 1))
        eps_da = td.SpatialDataArray(eps3d, coords={"x": coords_x, "y": coords_y, "z": coords_z})
        med = td.CustomMedium(permittivity=eps_da)
        return td.Structure(geometry=box, medium=med)

    params0 = np.full((len(coords_x), len(coords_y)), 0.7)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        outside_handling="extrapolate",
        maxiter=10,
    )
    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)


def test_outside_handling_mask_runs_and_bounds():
    """Extreme design coords: 'mask' runs, shape and bounds respected (partial coverage)."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    coords_x = np.array([-1e6, 0.0, 1e6])
    coords_y = np.linspace(box.center[1] - 0.5, box.center[1] + 0.5, 3)
    coords_z = np.array([0.0])

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps = 1.0 + 3.0 * p
        eps3d = eps.reshape((*p.shape, 1))
        eps_da = td.SpatialDataArray(eps3d, coords={"x": coords_x, "y": coords_y, "z": coords_z})
        med = td.CustomMedium(permittivity=eps_da)
        return td.Structure(geometry=box, medium=med)

    params0 = np.full((len(coords_x), len(coords_y)), 0.7)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        outside_handling="mask",
        maxiter=10,
    )
    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)


def test_outside_handling_nan_runs_and_bounds():
    """Extreme design coords: 'nan' runs (non-extended), shape and bounds respected."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    coords_x = np.array([-1e6, 0.0, 1e6])
    coords_y = np.linspace(box.center[1] - 0.5, box.center[1] + 0.5, 3)
    coords_z = np.array([0.0])

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps = 1.0 + 3.0 * p
        eps3d = eps.reshape((*p.shape, 1))
        eps_da = td.SpatialDataArray(eps3d, coords={"x": coords_x, "y": coords_y, "z": coords_z})
        med = td.CustomMedium(permittivity=eps_da)
        return td.Structure(geometry=box, medium=med)

    params0 = np.full((len(coords_x), len(coords_y)), 0.7)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        outside_handling="nan",
        maxiter=10,
    )
    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)


def test_invalid_outside_handling_raises_value_error():
    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps3d = (1.0 + 3.0 * p).reshape((*p.shape, 1))
        return td.Structure.from_permittivity_array(geometry=box, eps_data=eps3d)

    params0 = np.zeros((3, 3))

    with pytest.raises(ValueError):
        _ = initialize_params_from_simulation(
            sim=sim,
            param_to_structure=param_to_structure,
            params0=params0,
            outside_handling="invalid",
        )


def test_nan_all_outside_returns_params0():
    """If all design points are outside (NaN mask), params should remain unchanged."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    # put all design coords far away so interpolation (non-extended) yields all-NaN
    coords = {
        "x": np.array([1e6, 1e6 + 1.0, 1e6 + 2.0]),
        "y": np.array([1e6, 1e6 + 1.0, 1e6 + 2.0]),
        "z": np.array([0.0]),
    }

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps = 1.0 + 3.0 * p
        eps3d = eps.reshape((*p.shape, 1))
        eps_da = td.SpatialDataArray(eps3d, coords=coords)
        med = td.CustomMedium(permittivity=eps_da)
        return td.Structure(geometry=box, medium=med)

    params0 = np.full((3, 3), 0.42)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        outside_handling="nan",
        maxiter=5,
    )
    # no effective data in the loss, best-so-far remains the initial guess
    assert np.allclose(params, params0)


def test_3d_param_grid_supported_and_converges_to_zero():
    """3D parameter array (nx, ny, nz) converges to p~=0 for uniform base eps."""

    sim = td.Simulation(
        size=(2.0, 2.0, 1.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0))

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps3d = 1.0 + 3.0 * p
        return td.Structure.from_permittivity_array(geometry=box, eps_data=eps3d)

    params0 = np.full((3, 3, 2), 0.5)
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        freq=td.C_0 / 1.55,
        maxiter=25,
    )

    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)
    assert np.allclose(params, 0.0, atol=1e-3)


def test_param_to_structure_kwargs_forwarded():
    """Extra kwargs (e.g., beta) are forwarded into param_to_structure."""

    sim = td.Simulation(
        size=(2.0, 2.0, 0.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.0))

    def param_to_structure(p: np.ndarray, beta: float = 1.0) -> td.Structure:
        # Include beta to verify kwargs are accepted and used.
        eps = 1.0 + 3.0 * beta * p
        eps3d = eps.reshape((*p.shape, 1))
        return td.Structure.from_permittivity_array(geometry=box, eps_data=eps3d)

    params0 = np.full((4, 3), 0.6)
    # Pass beta via kwargs without functools.partial
    params = initialize_params_from_simulation(
        sim=sim,
        param_to_structure=param_to_structure,
        params0=params0,
        maxiter=10,
        beta=2.0,
    )
    assert params.shape == params0.shape
    assert np.all(params >= 0.0) and np.all(params <= 1.0)


def test_initialize_params_from_simulation_warns_when_center_is_between_design_pixels():
    """Centered even grids warn because the geometry center lies between design pixels."""

    sim, param_to_structure, params0 = _make_centered_linear_structure_case((4, 4))
    with AssertLogLevel(
        "WARNING", contains_str="Design coordinates do not include the geometry center"
    ):
        initialize_params_from_simulation(
            sim=sim,
            param_to_structure=param_to_structure,
            params0=params0,
            maxiter=2,
        )


def test_initialize_params_from_simulation_no_warning_when_center_is_sampled():
    """Centered odd grids include the geometry center and should not warn."""

    sim, param_to_structure, params0 = _make_centered_linear_structure_case((5, 5))
    with AssertLogLevel(None):
        initialize_params_from_simulation(
            sim=sim,
            param_to_structure=param_to_structure,
            params0=params0,
            maxiter=2,
        )


def test_initialize_params_from_simulation_no_warning_for_singleton_axis_off_center():
    """A singleton axis is invariant and should not warn if its lone sample is off-center."""

    sim = td.Simulation(
        size=(2.0, 2.0, 1.0),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        run_time=1e-12,
        medium=td.Medium(permittivity=1.0),
    )
    box = td.Box(center=(0, 0, 0), size=(1.0, 1.0, 0.2))
    coords_x = np.linspace(-0.4, 0.4, 5)
    coords_y = np.linspace(-0.4, 0.4, 5)
    coords_z = np.array([0.03])

    def param_to_structure(p: np.ndarray) -> td.Structure:
        eps3d = (1.0 + 3.0 * p).reshape((*p.shape, 1))
        eps_da = td.SpatialDataArray(eps3d, coords={"x": coords_x, "y": coords_y, "z": coords_z})
        med = td.CustomMedium(permittivity=eps_da)
        return td.Structure(geometry=box, medium=med)

    params0 = np.full((len(coords_x), len(coords_y)), 0.5)
    with AssertLogLevel(None):
        initialize_params_from_simulation(
            sim=sim,
            param_to_structure=param_to_structure,
            params0=params0,
            maxiter=2,
        )
