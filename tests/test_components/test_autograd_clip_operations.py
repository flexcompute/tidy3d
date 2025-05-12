"""
Finite‑difference (FD) vs Autograd (AD) checks for td.ClipOperation
covering all four Boolean operations: union, intersection, difference,
symmetric_difference.

The test contains *two distinct PolySlabs* that are combined by the
chosen Boolean op.
"""

from __future__ import annotations

import atexit
from collections import defaultdict
from pathlib import Path

import autograd
import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tidy3d as td
import tidy3d.web as web

# simulation constants
wavelength = 1.5
freq0 = td.C_0 / wavelength
L = 10 * wavelength
buffer = 1.0 * wavelength
run_time = 120 / freq0

source = td.PointDipole(
    center=(-L / 2 + buffer, 0.0, 0.0),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
    polarization="Ez",
)
monitor = td.FieldMonitor(
    center=(+L / 2 - buffer, 0.5 * buffer, 0.5 * buffer),
    size=(0, 0, 0),
    freqs=[freq0],
    name="point_out",
)


SCENARIOS: dict[str, list[dict[str, tuple | str | float]]] = {
    # ------------------------------------------------------------------  intersection
    "intersection": [
        {
            "name": " separated in z",
            "box_eps": 2.0,
            "center1": (0.0, 0.00, -0.25),
            "size1": (1.8, 1.5, 2.5),
            "center2": (0.0, 0.00, 0.25),
            "size2": (1.8, 1.5, 2.5),
        },
    ],
    # ------------------------------------------------------------------  difference
    "difference": [
        {
            "name": " separated in z",
            "box_eps": 2.0,
            "center1": (0.0, 0.00, -0.25),
            "size1": (1.8, 1.5, 2.5),
            "center2": (0.0, 0.00, 0.25),
            "size2": (1.8, 1.5, 2.5),
        },
    ],
    # ------------------------------------------------------------------  union
    "union": [
        {
            "name": " separated in z",
            "box_eps": 2.0,
            "center1": (0.0, 0.00, -0.25),
            "size1": (2.0, 1.5, 1.5),
            "center2": (0.0, 0.00, 0.25),
            "size2": (2.0, 1.5, 1.5),
        },
    ],
    # ------------------------------------------------------------------  symmetric_difference
    "symmetric_difference": [
        {
            "name": " separated in z",
            "box_eps": 2.5,
            "center1": (0.0, 0.45, -0.3),
            "size1": (1.9, 1.5, 2.5),
            "center2": (0.0, 0.45, 0.4),
            "size2": (1.9, 1.5, 2.5),
        },
    ],
}

# all Boolean ops we want to check
OPERATIONS = ["union", "intersection", "difference", "symmetric_difference"]

# geometric parameter labels --------------------------------------------------
PARAM_LABELS = [
    "p1_center_x",
    "p1_center_y",
    "p1_center_z",
    "p1_size_x",
    "p1_size_y",
    "p1_size_z",
    "p2_center_x",
    "p2_center_y",
    "p2_center_z",
    "p2_size_x",
    "p2_size_y",
    "p2_size_z",
]
_PARAM_MAP = {
    # poly‑1
    "p1_center_x": (0, "center1"),
    "p1_center_y": (1, "center1"),
    "p1_center_z": (2, "center1"),
    "p1_size_x": (0, "size1"),
    "p1_size_y": (1, "size1"),
    "p1_size_z": (2, "size1"),
    # poly‑2
    "p2_center_x": (0, "center2"),
    "p2_center_y": (1, "center2"),
    "p2_center_z": (2, "center2"),
    "p2_size_x": (0, "size2"),
    "p2_size_y": (1, "size2"),
    "p2_size_z": (2, "size2"),
}

# bookkeeping for optional plots / raw data ----------------------------------
SAVE_RESULTS = False
PLOT_RESULTS = True
RESULTS_DIR = Path("./clipop_fd_ad_results")
results_collector = defaultdict(list)


# helper: build simulation for a given op & geometry
def make_simulation(
    center1: tuple[float, float, float],
    size1: tuple[float, float, float],
    center2: tuple[float, float, float],
    size2: tuple[float, float, float],
    eps_box: float,
    operation: str,
) -> td.Simulation:
    """Scene with two PolySlabs combined by *operation*."""

    # PolySlab‑1 -----------------------------------------------------------
    cx1, cy1, cz1 = center1
    sx1, sy1, sz1 = size1
    hx1, hy1, hz1 = sx1 / 2, sy1 / 2, sz1 / 2
    verts1 = anp.array(
        [
            [cx1 - hx1, cz1 - hz1],
            [cx1 + hx1, cz1 - hz1],
            [cx1 + hx1, cz1 + hz1],
            [cx1 - hx1, cz1 + hz1],
        ]
    )
    slab_bounds1 = (cy1 - hy1, cy1 + hy1)
    poly1 = td.PolySlab(axis=1, vertices=verts1, slab_bounds=slab_bounds1)

    # PolySlab‑2 -----------------------------------------------------------
    cx2, cy2, cz2 = center2
    sx2, sy2, sz2 = size2
    hx2, hy2, hz2 = sx2 / 2, sy2 / 2, sz2 / 2
    verts2 = anp.array(
        [
            [cx2 - hx2, cz2 - hz2],
            [cx2 + hx2, cz2 - hz2],
            [cx2 + hx2, cz2 + hz2],
            [cx2 - hx2, cz2 + hz2],
        ]
    )
    slab_bounds2 = (cy2 - hy2, cy2 + hy2)
    poly2 = td.PolySlab(axis=1, vertices=verts2, slab_bounds=slab_bounds2)

    # Boolean operation ----------------------------------------------------
    geometry_union = td.ClipOperation(operation=operation, geometry_a=poly1, geometry_b=poly2)

    scatter = td.Structure(
        geometry=geometry_union,
        medium=td.Medium(permittivity=eps_box),
    )
    min_steps_per_wvl = 55
    if operation == "symmetric_difference":
        min_steps_per_wvl = 60

    return td.Simulation(
        size=(L, L, L),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        sources=[source],
        monitors=[monitor],
        structures=[scatter],
        run_time=run_time,
    )


# objective fn
def objective_fn(params: dict, scenario: dict, operation: str) -> anp.ndarray:
    """Intensity at the output monitor."""
    sim = make_simulation(
        params["center1"],
        params["size1"],
        params["center2"],
        params["size2"],
        eps_box=scenario["box_eps"],
        operation=operation,
    )
    sim_data = web.run(sim, task_name="clipop_fd_vs_ad", local_gradient=True, verbose=False)
    return anp.sum(sim_data.get_intensity("point_out").values)


# FD vs AD for a single parameter
def fd_vs_ad_single_param(
    params: dict, scenario: dict, operation: str, param_label: str, delta: float = 3.0e-2
):
    def pack(p: dict) -> anp.ndarray:
        return anp.concatenate([p["center1"], p["size1"], p["center2"], p["size2"]])

    def unpack(vec: anp.ndarray) -> dict:
        return dict(
            center1=tuple(vec[0:3]),
            size1=tuple(vec[3:6]),
            center2=tuple(vec[6:9]),
            size2=tuple(vec[9:12]),
        )

    x0 = pack(params)

    val_grad = autograd.value_and_grad(lambda v: objective_fn(unpack(v), scenario, operation))
    _, grad_vec = val_grad(x0)

    # index of the parameter being tested
    idx, which_key = _PARAM_MAP[param_label]
    flat_offset = {"center1": 0, "size1": 3, "center2": 6, "size2": 9}[which_key]
    flat_idx = flat_offset + idx
    ad_val = grad_vec[flat_idx]

    # centred FD
    x_plus = x0.copy()
    x_plus[flat_idx] += delta
    x_minus = x0.copy()
    x_minus[flat_idx] -= delta
    p_plus = objective_fn(unpack(x_plus), scenario, operation)
    p_minus = objective_fn(unpack(x_minus), scenario, operation)
    fd_val = (p_plus - p_minus) / (2.0 * delta)

    return fd_val, ad_val, p_plus, p_minus


# parametric pytest
@pytest.mark.numerical
@pytest.mark.parametrize("operation", OPERATIONS, ids=OPERATIONS)
@pytest.mark.parametrize("param_label", PARAM_LABELS)
def test_clipop_fd_vs_ad(operation: str, param_label: str):
    # initial geometry
    for scenario in SCENARIOS[operation]:
        params0 = dict(
            center1=scenario["center1"],
            size1=scenario["size1"],
            center2=scenario["center2"],
            size2=scenario["size2"],
        )
        fd_val, ad_val, p_plus, p_minus = fd_vs_ad_single_param(
            params0, scenario, operation, param_label, delta=0.06
        )

        # basic sanity
        assert np.isfinite(fd_val), f"FD derivative NaN/Inf ({param_label}, {operation})"
        assert np.isfinite(ad_val), f"AD derivative NaN/Inf ({param_label}, {operation})"

        # relative agreement
        denom = max(abs(fd_val), 1e-12)
        rel_diff = abs(fd_val - ad_val) / denom
        assert rel_diff < 0.4, (
            f"{operation} – {param_label} – {scenario['name']}: "
            f"FD–AD mismatch (rel diff {rel_diff:.2%}), (fd_val {fd_val:.2}), (ad_val {ad_val:.2})"
        )

        results_collector[(param_label, operation)].append((scenario["name"], rel_diff))

        # optional saving
        if SAVE_RESULTS:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(
                RESULTS_DIR / f"fd_ad_{scenario['name'].replace(' ', '_')}"
                f"_{operation}_{param_label}.npy",
                dict(
                    scenario_name=scenario["name"],
                    operation=operation,
                    param_label=param_label,
                    fd_val=float(fd_val),
                    ad_val=float(ad_val),
                    p_plus=float(p_plus),
                    p_minus=float(p_minus),
                    rel_diff=float(rel_diff),
                ),
            )


# optional bar‑plot
def _finalize_plotting():
    if not PLOT_RESULTS:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for param in PARAM_LABELS:
        # collect labels and errors across ALL operations for this param
        labels, rel_diffs = [], []
        for op in OPERATIONS:
            for scenario_name, rel_diff in results_collector.get((param, op), []):
                labels.append(f"{op}\n{scenario_name}")
                rel_diffs.append(rel_diff)

        if not rel_diffs:
            continue

        plt.figure(figsize=(8, 4))
        plt.bar(labels, rel_diffs)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("relative |FD–AD| / max(|FD|)")
        plt.title(f"FD vs AD rel error for {param}")
        plt.tight_layout()
        fname = RESULTS_DIR / f"rel_error_{param}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")


atexit.register(_finalize_plotting)
