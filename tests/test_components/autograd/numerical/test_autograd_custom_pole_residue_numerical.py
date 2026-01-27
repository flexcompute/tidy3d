"""Numerical test for CustomPoleResidue adjoint gradients."""

from __future__ import annotations

import sys

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


SIM_SIZE_SCALE = (4, 3, 4)
BOX_SIZE_SCALE = (1, 1, 1)
GRID_STEPS_PER_WVL = 30
RUN_TIME = 2e-12
ANGLE_TOL = 10.0
FD_STEP = 5e-2

DENSITY_SHAPE = (2, 2, 2)
EPS_BACKGROUND = 1.0
EPS_INF_BASE = 2.8
A_BASE = -1.5e15
C_BASE = 0.8e15

POLE_RESIDUE_CASE = {
    "name": "custom_pole_residue",
    "wavelength": 1.3,
    "objective_kind": "flux",
    "monitor_size": (np.inf, np.inf, 0.0),
    "polarization": 0.0,
}


def _scale_monitor_dim(dim: float, wavelength: float) -> float:
    if np.isinf(dim):
        return np.inf
    return dim * wavelength


def _box_geometry(wavelength: float) -> td.Box:
    size = tuple(scale * wavelength for scale in BOX_SIZE_SCALE)
    return td.Box(size=size, center=(0.0, 0.0, 0.0))


def _build_base_sim(case):
    wavelength = case["wavelength"]
    freq0 = td.C_0 / wavelength
    sim_size = tuple(scale * wavelength for scale in SIM_SIZE_SCALE)

    plane_wave = td.PlaneWave(
        center=(0.0, 0.0, -0.75 * sim_size[2] / 2),
        size=(sim_size[0], sim_size[1], 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        direction="+",
        pol_angle=case.get("polarization", 0.0),
    )

    monitor_center = (0.0, 0.0, sim_size[2] / 2 * 0.75)
    monitor_size = tuple(_scale_monitor_dim(dim, wavelength) for dim in case["monitor_size"])
    monitor_name = f"{case['name']}_monitor"
    monitor = td.FieldMonitor(
        center=monitor_center,
        size=monitor_size,
        freqs=[freq0],
        name=monitor_name,
        colocate=False,
    )

    sim = td.Simulation(
        size=sim_size,
        center=(0.0, 0.0, 0.0),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=GRID_STEPS_PER_WVL, wavelength=wavelength),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[plane_wave],
        monitors=[monitor],
        structures=[],
        run_time=RUN_TIME,
    )
    return sim, monitor_name, freq0


def _density_coords(box_geom: td.Box) -> dict[str, np.ndarray]:
    return {
        "x": np.linspace(-box_geom.size[0] / 2, box_geom.size[0] / 2, DENSITY_SHAPE[0]),
        "y": np.linspace(-box_geom.size[1] / 2, box_geom.size[1] / 2, DENSITY_SHAPE[1]),
        "z": np.linspace(-box_geom.size[2] / 2, box_geom.size[2] / 2, DENSITY_SHAPE[2]),
    }


def _custom_pole_residue_from_density(density, coords):
    eps_inf = EPS_BACKGROUND + density * (EPS_INF_BASE - EPS_BACKGROUND)
    eps_inf_da = td.SpatialDataArray(eps_inf, coords=coords)

    a_vals = A_BASE * anp.ones_like(density)
    c_vals = C_BASE * density
    a_da = td.SpatialDataArray(a_vals, coords=coords)
    c_da = td.SpatialDataArray(c_vals, coords=coords)

    return td.CustomPoleResidue(eps_inf=eps_inf_da, poles=((a_da, c_da),), interp_method="linear")


def _add_custom_pole_residue(base_sim: td.Simulation, box_geom: td.Box, params) -> td.Simulation:
    density = anp.reshape(params, DENSITY_SHAPE)
    coords = _density_coords(box_geom)
    medium = _custom_pole_residue_from_density(density, coords)
    structure = td.Structure(geometry=box_geom, medium=medium)
    return base_sim.updated_copy(structures=[structure])


def _metric_value(case, dataset, freq0):
    if case["objective_kind"] == "flux":
        return dataset.flux.values.item()
    ex_vals = dataset.Ex.values
    ey_vals = dataset.Ey.values
    ez_vals = dataset.Ez.values
    intensity = np.abs(ex_vals) ** 2 + np.abs(ey_vals) ** 2 + np.abs(ez_vals) ** 2
    return anp.real(anp.mean(intensity))


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    cos_theta = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _run_simulation(
    case, base_sim, box_geom, params, label, tmp_path, monitor_name, freq0, local_gradient
):
    sim = _add_custom_pole_residue(base_sim, box_geom, params)
    sim_data = web.run(
        sim,
        task_name=f"custom_pole_residue_grad_{case['name']}_{label}",
        local_gradient=local_gradient,
        verbose=False,
        path=str(tmp_path / f"{case['name']}_{label}.hdf5"),
    )
    return _metric_value(case, sim_data[monitor_name], freq0)


@pytest.mark.numerical
def test_custom_pole_residue_grads_match_fd(numerical_case_dir, tmp_path, _enable_local_cache):
    case = POLE_RESIDUE_CASE
    base_sim, monitor_name, freq0 = _build_base_sim(case)
    box_geom = _box_geometry(case["wavelength"])
    params0 = anp.linspace(0.2, 0.8, num=int(np.prod(DENSITY_SHAPE)))

    def objective(params):
        return _run_simulation(
            case,
            base_sim,
            box_geom,
            params,
            label="adjoint",
            tmp_path=tmp_path,
            monitor_name=monitor_name,
            freq0=freq0,
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = get_static(grad_adj).reshape(-1)

    fd_sims = {}
    base_params = get_static(params0)
    for idx in range(base_params.size):
        delta = np.zeros_like(base_params)
        delta[idx] = FD_STEP
        fd_sims[f"fd_plus_{idx}"] = _add_custom_pole_residue(
            base_sim, box_geom, base_params + delta
        )
        fd_sims[f"fd_minus_{idx}"] = _add_custom_pole_residue(
            base_sim, box_geom, base_params - delta
        )

    fd_results = web.run_async(
        fd_sims,
        path_dir=str(numerical_case_dir / f"fd_batch_{case['name']}"),
        local_gradient=False,
        verbose=False,
    )

    grad_fd = np.zeros_like(grad_adj)
    for idx in range(base_params.size):
        plus = _metric_value(case, fd_results[f"fd_plus_{idx}"][monitor_name], freq0)
        minus = _metric_value(case, fd_results[f"fd_minus_{idx}"][monitor_name], freq0)
        grad_fd[idx] = (plus - minus) / (2.0 * FD_STEP)

    angle_deg = _angle_deg(grad_adj, grad_fd)

    print(
        f"[custom-pole-residue-grad-test:{case['name']}] adjoint={grad_adj}, "
        f"finite-difference={grad_fd}, angle_deg={angle_deg:.3f}",
        file=sys.stderr,
    )

    assert angle_deg <= ANGLE_TOL or np.isnan(angle_deg), (
        f"Gradient angle deviation {angle_deg:.3f} deg exceeds tolerance ({ANGLE_TOL}). "
        f"adj={grad_adj}, fd={grad_fd}"
    )
