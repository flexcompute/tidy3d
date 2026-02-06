"""Numerical finite-difference validation for ClipOperation gradients with real simulations."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from matplotlib import pyplot as plt

import tidy3d as td
import tidy3d.web as web
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    angled_overlap_deg,
)
from tidy3d import config

WL_UM = 0.8
FREQ0 = td.C_0 / WL_UM
SRC_OFFSET = -2.2
MONITOR_OFFSET = 2.2
SIM_SIZE = (4.0, 4.0, 6.0)
RUN_TIME = 2e-11
PERMITTIVITY = 1.4**2
GRID_STEPS_PER_WVL = 30
FINITE_DIFF_STEP = 0.05
BASE_OFFSET = (0.05, -0.035, 0.02)
BASE_CENTER_A = (-0.25, 0.12, 0.0)
BASE_CENTER_B = (0.15, -0.08, 0.0)
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 11.0
BASE_CENTER_A_VEC = anp.array(BASE_CENTER_A, dtype=float)

SIZE_MAP = {
    "sphere": {"a": (0.9, 0.9, 0.9), "b": (0.7, 0.7, 0.7)},
    "box": {"a": (1.0, 0.9, 0.8), "b": (0.9, 0.75, 0.7)},
    "polyslab": {"a": (0.95, 0.85, 0.8), "b": (0.8, 0.7, 0.6)},
    "mesh": {"a": (0.85, 0.95, 0.95), "b": (0.7, 0.95, 0.8)},
}

STRUCTURE_MEDIUM = td.Medium(permittivity=PERMITTIVITY)


@pytest.fixture(autouse=True)
def _enable_local_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.local_cache, "enabled", True)


def _make_base_simulation() -> tuple[td.Simulation, Callable[[td.SimulationData], float]]:
    """Shared ClipOperation simulation (plane wave excitation and Ex monitor)."""
    source = td.PointDipole(
        center=(0.0, 0.0, SRC_OFFSET),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=0.2 * FREQ0),
        polarization="Ex",
    )
    field_monitor = td.FieldMonitor(
        center=(0.0, 0.0, MONITOR_OFFSET),
        size=(0, 0, 0.0),
        freqs=[FREQ0],
        name="field",
    )
    base_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=SIM_SIZE,
        sources=[source],
        monitors=[field_monitor],
        structures=[],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml()
        ),
        run_time=RUN_TIME,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=GRID_STEPS_PER_WVL),
    )

    def fom(sim_data: td.SimulationData) -> float:
        dataset = sim_data["field"]
        ex_vals = dataset.Ex.values
        ey_vals = dataset.Ey.values
        ez_vals = dataset.Ez.values
        intensity = np.abs(ex_vals) ** 2 + np.abs(ey_vals) ** 2 + np.abs(ez_vals) ** 2
        return anp.real(anp.mean(intensity))

    return base_sim, fom


def _triangle_prism(
    center: tuple[float, float, float], size: tuple[float, float, float]
) -> td.PolySlab:
    half_x, half_y, half_z = 0.5 * size[0], 0.5 * size[1], 0.5 * size[2]
    cx, cy, cz = center
    vertices = (
        (cx - half_x, cy - half_y),
        (cx + half_x, cy - half_y),
        (cx, cy + half_y),
    )
    slab_bounds = (cz - half_z, cz + half_z)
    return td.PolySlab(vertices=vertices, axis=2, slab_bounds=slab_bounds)


def _tetra_mesh(
    center: tuple[float, float, float], size: tuple[float, float, float]
) -> td.TriangleMesh:
    center_arr = _normalize_array(center)
    half = 0.5 * _normalize_array(size)
    cx, cy, cz = center_arr
    vertices = anp.array(
        [
            (cx - half[0], cy - half[1], cz - half[2]),
            (cx + half[0], cy - half[1], cz - half[2]),
            (cx, cy + half[1], cz - half[2]),
            (cx, cy, cz + half[2]),
        ],
        dtype=float,
    )
    faces = anp.array(
        [
            (0, 2, 1),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ],
        dtype=int,
    )
    mesh = td.TriangleMesh.from_triangles(vertices[faces])
    return mesh


def _make_geometry(
    geometry_type: str, center: tuple[float, float, float], size: tuple[float, float, float]
) -> td.Geometry:
    if geometry_type == "sphere":
        radius = 0.5 * min(size)
        return td.Sphere(center=center, radius=radius)
    if geometry_type == "box":
        return td.Box(center=center, size=size)
    if geometry_type == "polyslab":
        return _triangle_prism(center, size)
    if geometry_type == "mesh":
        return _tetra_mesh(center, size)
    raise ValueError(f"Unsupported geometry_type '{geometry_type}'.")


def _normalize_array(offset_vec):
    """Return the offset as an autograd array of length 3."""

    offset = anp.array(offset_vec, dtype=float)
    if offset.shape != (3,):
        raise ValueError("ClipOperation offset vector must have length 3.")
    return offset


def _clip_structure(offset_vec, geometry_type: str, operation: str) -> td.Structure:
    offset = _normalize_array(offset_vec)
    size_spec = SIZE_MAP[geometry_type]
    center_arr = BASE_CENTER_A_VEC + offset
    center_a = tuple(center_arr)
    geometry_a = _make_geometry(geometry_type, center=center_a, size=size_spec["a"])
    geometry_b = _make_geometry(geometry_type, center=BASE_CENTER_B, size=size_spec["b"])
    clip_geom = td.ClipOperation(
        operation=operation,
        geometry_a=geometry_a,
        geometry_b=geometry_b,
    )
    return td.Structure(geometry=clip_geom, medium=STRUCTURE_MEDIUM)


def _run_clip_simulation(
    base_sim: td.Simulation,
    fom: Callable[[td.SimulationData], float],
    offset_vec,
    geometry_type: str,
    operation: str,
    result_dir: Path,
    *,
    local_gradient: bool,
    tag: str,
    structure_builder: Callable = _clip_structure,
) -> float:
    return _run_clip_simulation_batch(
        base_sim,
        fom,
        offset_list=[offset_vec],
        geometry_type=geometry_type,
        operation=operation,
        result_dir=result_dir,
        local_gradient=local_gradient,
        tag=tag,
        structure_builder=structure_builder,
    )[0]


def _run_clip_simulation_batch(
    base_sim: td.Simulation,
    fom: Callable[[td.SimulationData], float],
    offset_list,
    geometry_type: str,
    operation: str,
    result_dir: Path,
    *,
    local_gradient: bool,
    tag: str,
    structure_builder: Callable = _clip_structure,
) -> list:
    """Run a batch of simulations (in parallel if possible) for different offsets."""

    offsets = [_normalize_array(off) for off in offset_list]
    simulations: dict[str, td.Simulation] = {}
    key_order: list[str] = []

    for idx, offset in enumerate(offsets):
        structure = structure_builder(offset, geometry_type, operation)
        grid_spec = td.GridSpec.auto(
            min_steps_per_wvl=GRID_STEPS_PER_WVL, override_structures=[structure]
        )
        sim = base_sim.updated_copy(structures=[structure], grid_spec=grid_spec, validate=True)
        task_name = f"clip_{geometry_type}_{operation}_{tag}_{idx}_{uuid.uuid4().hex[:6]}"
        simulations[task_name] = sim
        key_order.append(task_name)

    result_dir.mkdir(parents=True, exist_ok=True)

    if len(simulations) == 1:
        key = key_order[0]
        sim = simulations[key]
        result_path = result_dir / f"{key}.hdf5"
        sim_data = web.run(
            sim,
            task_name=key,
            path=str(result_path),
            local_gradient=local_gradient,
            verbose=False,
        )
        return [fom(sim_data)]

    sim_data_map = web.run_async(
        simulations,
        path_dir=str(result_dir),
        local_gradient=local_gradient,
        verbose=False,
    )

    return [fom(sim_data_map[key]) for key in key_order]


def _make_clip_objective(
    base_sim: td.Simulation,
    fom,
    geometry_type: str,
    operation: str,
    case_dir: Path,
    *,
    local_gradient: bool,
    structure_builder: Callable = _clip_structure,
):
    run_dir = case_dir / ("adjoint" if local_gradient else "finite_difference")

    def objective(params, *, batched: bool = False):
        tag = "adj" if local_gradient else "fd"
        if batched:
            offsets = [_normalize_array(p) for p in params]
            return _run_clip_simulation_batch(
                base_sim,
                fom,
                offset_list=offsets,
                geometry_type=geometry_type,
                operation=operation,
                result_dir=run_dir,
                local_gradient=local_gradient,
                tag=tag,
                structure_builder=structure_builder,
            )
        offset = _normalize_array(params)
        return _run_clip_simulation(
            base_sim,
            fom,
            offset_vec=offset,
            geometry_type=geometry_type,
            operation=operation,
            result_dir=run_dir,
            local_gradient=local_gradient,
            tag=tag,
            structure_builder=structure_builder,
        )

    return objective


def _finite_difference_gradient(objective, params: anp.ndarray, step: float) -> np.ndarray:
    base = _normalize_array(params)
    offsets = []
    axis_indices = []

    for idx in range(3):
        delta = anp.array(
            [step if dim == idx else 0.0 for dim in range(3)],
            dtype=float,
        )
        offsets.append(base + delta)
        offsets.append(base - delta)
        axis_indices.append(idx)

    results = objective(offsets, batched=True)
    grads = np.zeros(3, dtype=float)
    for pair_idx, axis in enumerate(axis_indices):
        obj_up = float(np.asarray(results[2 * pair_idx], dtype=float))
        obj_down = float(np.asarray(results[2 * pair_idx + 1], dtype=float))
        grads[axis] = (obj_up - obj_down) / (2.0 * step)

    return grads


@pytest.mark.numerical
@pytest.mark.parametrize("geometry_type", ["sphere", "box", "polyslab", "mesh"])
@pytest.mark.parametrize(
    "operation", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_clip_operation_vs_fd(geometry_type: str, operation: str, tmp_path: Path, monkeypatch):
    """Compare ClipOperation adjoint gradients against finite differences (3D offsets)."""
    if operation == "symmetric_difference" and geometry_type in ("mesh", "box"):
        # we need more sampling density in these cases
        monkeypatch.setattr(td.config.adjoint, "default_wavelength_fraction", 0.01)
    base_sim, fom = _make_base_simulation()
    adjoint_objective = _make_clip_objective(
        base_sim,
        fom,
        geometry_type,
        operation,
        tmp_path,
        local_gradient=True,
    )
    params0 = anp.array(BASE_OFFSET, dtype=float)
    _, grad = value_and_grad(adjoint_objective)(params0)
    fd_objective = _make_clip_objective(
        base_sim,
        fom,
        geometry_type,
        operation,
        tmp_path,
        local_gradient=False,
    )

    autograd_grad = np.asarray(grad, dtype=float).ravel()
    fd_grad = _finite_difference_gradient(fd_objective, params0, FINITE_DIFF_STEP)

    angle = angled_overlap_deg(autograd_grad, fd_grad)
    assert angle < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
        f"FD–adjoint angle overlap too large ({angle:.2f}°) for {geometry_type}/{operation}"
    )


@pytest.mark.numerical
@pytest.mark.parametrize("geometry_type", ["box", "polyslab", "mesh"])
@pytest.mark.parametrize(
    "operation", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_clip_operation_fd_sweep(geometry_type: str, operation: str, numerical_case_dir: Path):
    """Sweep FD step sizes and save comparison plots for diagnostic use."""

    base_sim, fom = _make_base_simulation()
    case_dir = numerical_case_dir / f"{geometry_type}_{operation}_fd_sweep"
    case_dir.mkdir(parents=True, exist_ok=True)
    adjoint_objective = _make_clip_objective(
        base_sim,
        fom,
        geometry_type,
        operation,
        case_dir,
        local_gradient=True,
    )
    fd_objective = _make_clip_objective(
        base_sim,
        fom,
        geometry_type,
        operation,
        case_dir,
        local_gradient=False,
    )

    params0 = anp.array(BASE_OFFSET, dtype=float)
    _, grad = value_and_grad(adjoint_objective)(params0)
    autograd_grad = np.asarray(grad, dtype=float).ravel()

    steps = np.logspace(-4, -1, num=7)
    fd_grads = np.array(
        [_finite_difference_gradient(fd_objective, params0, float(step)) for step in steps]
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ("dx", "dy", "dz")
    for idx, label in enumerate(labels):
        ax.plot(steps, fd_grads[:, idx], marker="o", label=f"{label} FD")
        ax.axhline(
            autograd_grad[idx],
            color=ax.get_lines()[-1].get_color(),
            linestyle="--",
            alpha=0.7,
            label=f"{label} adjoint",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Finite-difference step (µm)")
    ax.set_ylabel("Gradient value")
    ax.set_title(f"FD sweep for {geometry_type} / {operation}")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="best", fontsize="small")

    fig_path = case_dir / f"fd_sweep_{geometry_type}_{operation}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        case_dir / f"fd_sweep_{geometry_type}_{operation}.npz",
        steps=steps,
        gradients=fd_grads,
        autograd_grad=autograd_grad,
    )
