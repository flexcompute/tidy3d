"""Numerical validation for multi-frequency custom dispersive medium gradients."""

from __future__ import annotations

import sys

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from autograd import value_and_grad

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch):
    monkeypatch.setattr(td.config.local_cache, "enabled", True)


SIM_SIZE_SCALE = (3.0, 2.5, 3.0)
BOX_SIZE_SCALE = (0.8, 0.8, 0.8)
GRID_STEPS_PER_WVL = 40
RUN_TIME = 2e-13
FD_STEP = 5e-3
ANGLE_TOL = 5.0

FREQS = np.array([1.7e14, 2.4e14])
FREQ_WEIGHTS = np.array([1.0, 0.6])

PARAM_SHAPE_2D = (2, 2)
PARAM_SHAPE = (2, 2, 2)
FD_SWEEP_STEPS = np.logspace(-3, -1, num=7)

SELLMEIER_C_VAL = 0.6 * (td.C_0 / np.max(FREQS)) ** 2

TEST_CASES = [
    {
        "name": "lo1",  # keep names short, filenames get too long otherwise
        "kind": "lorentz",
        "eps_inf": 1.6,
        "param0": 0.5,
        "f0": 2.6e14,
        "delta": 0.2e14,
    },
    {
        "name": "lo2",
        "kind": "lorentz",
        "eps_inf": 2.3,
        "param0": 0.7,
        "f0": 2.3e14,
        "delta": 0.2e14,
    },
    {
        "name": "lo3",
        "kind": "lorentz",
        "eps_inf": 1.9,
        "param0": 0.35,
        "f0": 3.0e14,
        "delta": 0.2e14,
    },
    {
        "name": "sl",
        "kind": "sellmeier",
        "param0": 0.6,
        "c_val": SELLMEIER_C_VAL,
    },
    {
        "name": "dd",
        "kind": "drude",
        "eps_inf": 1.6,
        "param0": 0.5,
        "param_scale": 2.0e14,
        "delta": 0.3e14,
    },
    {
        "name": "db",
        "kind": "debye",
        "eps_inf": 2.5,
        "param0": 0.5,
        "tau": 0.4e-14,
    },
    {
        "name": "pr",
        "kind": "pole_residue",
        "eps_inf": 1.6,
        "param0": 0.5,
        "param_scale": 1.0e14,
        "a_val": -1.2e14,
    },
]


def _build_base_sim(freqs: np.ndarray) -> tuple[td.Simulation, str, float]:
    wavelength_min = td.C_0 / np.max(freqs)
    sim_size = tuple(scale * wavelength_min for scale in SIM_SIZE_SCALE)

    freq0 = float(np.mean(freqs))
    fwidth = float(max(freqs.max() - freqs.min(), 0.4 * freq0))

    src = td.PlaneWave(
        center=(0.0, 0.0, -0.75 * sim_size[2] / 2),
        size=(sim_size[0], sim_size[1], 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        pol_angle=0.0,
    )

    monitor_name = "field_monitor"
    monitor = td.FieldMonitor(
        center=(0.0, 0.0, sim_size[2] / 2 * 0.6),
        size=(sim_size[0], sim_size[1], 0.0),
        freqs=list(freqs),
        name=monitor_name,
        colocate=False,
    )

    sim = td.Simulation(
        size=sim_size,
        center=(0.0, 0.0, 0.0),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=GRID_STEPS_PER_WVL,
            wavelength=wavelength_min,
        ),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[src],
        monitors=[monitor],
        structures=[],
        run_time=RUN_TIME,
    )
    return sim, monitor_name, wavelength_min


def _box_geometry(wavelength_min: float) -> td.Box:
    size = tuple(scale * wavelength_min for scale in BOX_SIZE_SCALE)
    return td.Box(size=size, center=(0.0, 0.0, 0.0))


def _coords_for_bounds(bounds, shape):
    return {
        "x": np.linspace(bounds[0][0], bounds[1][0], shape[0]),
        "y": np.linspace(bounds[0][1], bounds[1][1], shape[1]),
        "z": np.linspace(bounds[0][2], bounds[1][2], shape[2]),
    }


def _custom_medium(case, param_vals: anp.ndarray, box_geom: td.Box):
    bounds = box_geom.bounds
    coords = _coords_for_bounds(bounds, param_vals.shape)
    kind = case["kind"]
    param_scale = case.get("param_scale", 1.0)
    scaled = param_scale * param_vals

    if kind == "lorentz":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        de = td.SpatialDataArray(scaled, coords=coords)
        f0 = td.SpatialDataArray(np.full(param_vals.shape, case["f0"]), coords=coords)
        delta = td.SpatialDataArray(np.full(param_vals.shape, case["delta"]), coords=coords)
        return td.CustomLorentz(eps_inf=eps_inf, coeffs=[(de, f0, delta)])
    if kind == "sellmeier":
        b = td.SpatialDataArray(scaled, coords=coords)
        c = td.SpatialDataArray(np.full(param_vals.shape, case["c_val"]), coords=coords)
        return td.CustomSellmeier(coeffs=[(b, c)])
    if kind == "drude":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        fp = td.SpatialDataArray(scaled, coords=coords)
        delta = td.SpatialDataArray(np.full(param_vals.shape, case["delta"]), coords=coords)
        return td.CustomDrude(eps_inf=eps_inf, coeffs=[(fp, delta)])
    if kind == "debye":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        de = td.SpatialDataArray(scaled, coords=coords)
        tau = td.SpatialDataArray(np.full(param_vals.shape, case["tau"]), coords=coords)
        return td.CustomDebye(eps_inf=eps_inf, coeffs=[(de, tau)])
    if kind == "pole_residue":
        eps_inf = td.SpatialDataArray(np.full(param_vals.shape, case["eps_inf"]), coords=coords)
        a_val = td.SpatialDataArray(np.full(param_vals.shape, case["a_val"]), coords=coords)
        c_val = td.SpatialDataArray(scaled, coords=coords)
        return td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a_val, c_val)])
    raise ValueError(f"Unsupported medium kind: {kind}")


def _add_medium(
    sim: td.Simulation, box_geom: td.Box, case, param_vals: anp.ndarray
) -> td.Simulation:
    medium = _custom_medium(case, param_vals, box_geom)
    structure = td.Structure(geometry=box_geom, medium=medium)
    return sim.updated_copy(structures=[structure])


def _metric_value(dataset) -> float:
    ex_vals = dataset.Ex.values
    ey_vals = dataset.Ey.values
    ez_vals = dataset.Ez.values
    intensity = anp.abs(ex_vals) ** 2 + anp.abs(ey_vals) ** 2 + anp.abs(ez_vals) ** 2
    weighted = intensity * anp.asarray(FREQ_WEIGHTS)
    return anp.real(anp.mean(weighted))


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    cos_theta = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _expand_params(params: anp.ndarray) -> anp.ndarray:
    vals_2d = anp.reshape(params, PARAM_SHAPE_2D)
    return anp.repeat(vals_2d[..., None], PARAM_SHAPE[2], axis=2)


def _run_simulation(
    sim: td.Simulation,
    monitor_name: str,
    tmp_path,
    label: str,
    local_gradient: bool,
) -> float:
    sim_data = web.run(
        sim,
        task_name=f"custom_disp_{label}",
        local_gradient=local_gradient,
        verbose=False,
        path=str(tmp_path / f"custom_disp_{label}.hdf5"),
    )
    return _metric_value(sim_data[monitor_name])


@pytest.mark.numerical
@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["name"])
def test_custom_dispersive_multifreq_grad_matches_fd(
    case, numerical_case_dir, tmp_path, _enable_local_cache
):
    base_sim, monitor_name, wavelength_min = _build_base_sim(FREQS)
    box_geom = _box_geometry(wavelength_min)

    params0 = anp.full(PARAM_SHAPE_2D, case["param0"]).reshape(-1)

    def objective(param_vec):
        param_vals = _expand_params(param_vec)
        sim = _add_medium(base_sim, box_geom, case, param_vals)
        return _run_simulation(
            sim=sim,
            monitor_name=monitor_name,
            tmp_path=tmp_path,
            label="adjoint",
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = np.asarray(get_static(grad_adj), dtype=float).reshape(-1)

    fd_sims: dict[str, td.Simulation] = {}
    for idx in range(params0.size):
        delta = np.zeros_like(params0)
        delta[idx] = FD_STEP
        plus_vals = _expand_params(params0 + delta)
        minus_vals = _expand_params(params0 - delta)
        fd_sims[f"plus_{idx}"] = _add_medium(base_sim, box_geom, case, plus_vals)
        fd_sims[f"minus_{idx}"] = _add_medium(base_sim, box_geom, case, minus_vals)

    fd_results = web.run_async(
        fd_sims,
        path_dir=str(numerical_case_dir / f"{case['name']}"),
        local_gradient=False,
        verbose=False,
    )

    grad_fd = np.zeros_like(grad_adj)
    for idx in range(params0.size):
        val_plus = _metric_value(fd_results[f"plus_{idx}"][monitor_name])
        val_minus = _metric_value(fd_results[f"minus_{idx}"][monitor_name])
        grad_fd[idx] = (val_plus - val_minus) / (2.0 * FD_STEP)

    angle_deg = _angle_deg(grad_adj, grad_fd)
    print(
        (
            f"[custom-dispersive-multifreq:{case['name']}] adjoint={grad_adj}, "
            f"finite-difference={grad_fd}, angle_deg={angle_deg:.3f}"
        ),
        file=sys.stderr,
    )

    assert angle_deg <= ANGLE_TOL or np.isnan(angle_deg), (
        f"Multi-frequency CustomDispersive gradient mismatch for {case['name']}. "
        f"angle_deg={angle_deg:.3f}, adj={grad_adj}, fd={grad_fd}"
    )


@pytest.mark.numerical
def test_custom_lorentz_fd_step_sweep(numerical_case_dir, tmp_path, _enable_local_cache):
    base_sim, monitor_name, wavelength_min = _build_base_sim(FREQS)
    box_geom = _box_geometry(wavelength_min)

    case = TEST_CASES[0]
    params0 = anp.full(PARAM_SHAPE_2D, case["param0"]).reshape(-1)

    def objective(de_params):
        de_vals = _expand_params(de_params)
        sim = _add_medium(base_sim, box_geom, case, de_vals)
        return _run_simulation(
            sim=sim,
            monitor_name=monitor_name,
            tmp_path=tmp_path,
            label="adjoint_sweep",
            local_gradient=True,
        )

    _, grad_adj = value_and_grad(objective)(params0)
    grad_adj = np.asarray(get_static(grad_adj), dtype=float).reshape(-1)

    sweep_runs: dict[str, td.Simulation] = {}
    step_labels = [f"{step:.3e}" for step in FD_SWEEP_STEPS]
    for step_label, step in zip(step_labels, FD_SWEEP_STEPS):
        plus_vals = _expand_params(params0 + step)
        minus_vals = _expand_params(params0 - step)
        sweep_runs[f"step_{step_label}_plus"] = _add_medium(base_sim, box_geom, case, plus_vals)
        sweep_runs[f"step_{step_label}_minus"] = _add_medium(base_sim, box_geom, case, minus_vals)

    sweep_results = web.run_async(
        sweep_runs,
        path_dir=str(numerical_case_dir / f"fd_sweep_{case['name']}"),
        local_gradient=False,
        verbose=False,
    )

    fd_sweep = []
    for step_label, step in zip(step_labels, FD_SWEEP_STEPS):
        plus_key = f"step_{step_label}_plus"
        minus_key = f"step_{step_label}_minus"
        plus_val = _metric_value(sweep_results[plus_key][monitor_name])
        minus_val = _metric_value(sweep_results[minus_key][monitor_name])
        fd_sweep.append((plus_val - minus_val) / (2.0 * step))

    fd_sweep = np.array(fd_sweep, dtype=float)
    fd_min = float(np.min(fd_sweep))
    fd_max = float(np.max(fd_sweep))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(FD_SWEEP_STEPS, fd_sweep, marker="o", label="FD")
    ax.axhline(
        np.mean(grad_adj),
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        alpha=0.7,
        label="Adjoint (mean)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step")
    ax.set_ylabel("Gradient value")
    ax.set_title("CustomLorentz FD sweep")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = numerical_case_dir / "custom_lorentz_fd_step_sweep.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(
        (
            "[custom-dispersive-fd-sweep] "
            f"grad_adj={grad_adj} "
            f"fd_grad[min,max]=({fd_min:.6e},{fd_max:.6e})"
        ),
        file=sys.stderr,
    )
