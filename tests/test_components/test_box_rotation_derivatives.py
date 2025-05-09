"""
Finite‑difference (FD) vs Autograd (AD) gradients with respect to the
rotation angle θ of a dielectric box, tested for rotations about all
three coordinate axes (x, y, z).  A separate line plot is produced for
each axis.
"""

import atexit
import os
from collections import defaultdict

import autograd
import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tidy3d as td
import tidy3d.web as web

# ───── global switches ────────────────────────────────────────────────────
SAVE_RESULTS = False
PLOT_RESULTS = False
RESULTS_DIR = "./fd_ad_theta_results"

# ───── simulation constants ───────────────────────────────────────────────
wavelength = 1.5
freq0 = td.C_0 / wavelength
L = 10 * wavelength
buffer = 1.0 * wavelength
run_time = 120 / freq0

# baseline geometry / material
center0 = (0.0, 0.0, 0.0)
size0 = (2.0, 2.0, 2.0)
eps0 = 2.0

# sweep parameters
delta = 0.016  # FD step
theta_sweep = np.array(
    [0.0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 3 * np.pi / 2.0, 2 * np.pi]
)
AXES = [0, 1, 2]  # x, y, z
AXIS_LABEL = {0: "x", 1: "y", 2: "z"}

plots = {a: defaultdict(list) for a in AXES}


# ───── helper: build a simulation ─────────────────────────────────────────
def make_simulation(center, size, eps, theta, axis):
    """Tidy3D simulation with the box rotated by θ about the given axis."""
    source = td.PointDipole(
        center=(-L / 2 + 0.5 * buffer, 0.0, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        polarization="Ez",
    )
    monitor = td.FieldMonitor(
        center=(+L / 2 - 0.5 * buffer, 2 * buffer, 2 * buffer),
        size=(0, 0, 0),
        freqs=[freq0],
        name="m",
    )

    box = td.Box(center=center, size=size)
    geom = box.rotated(theta, axis)
    struct = td.Structure(geometry=geom, medium=td.Medium(permittivity=eps))

    return td.Simulation(
        size=(L, L, L),
        run_time=run_time,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=50),
        sources=[source],
        monitors=[monitor],
        structures=[struct],
    )


# ───── objective & FD helper ──────────────────────────────────────────────
def objective_fn(center, size, eps, theta, axis):
    sim = make_simulation(center, size, eps, theta, axis)
    data = web.run(sim, task_name="fd_ad_theta", verbose=False, local_gradient=True)
    return anp.sum(data.get_intensity("m").values)


def finite_diff_theta(center, size, eps, theta, axis, delta=1e-3):
    p_plus = objective_fn(center, size, eps, theta + delta, axis)
    p_minus = objective_fn(center, size, eps, theta - delta, axis)
    return (p_plus - p_minus) / (2.0 * delta)


# ───── pytest parametrised check ─────────────────────────────────────────
@pytest.mark.numerical
@pytest.mark.parametrize("axis", AXES, ids=[f"axis_{AXIS_LABEL[a]}" for a in AXES])
@pytest.mark.parametrize("theta", theta_sweep)
def test_fd_vs_ad_theta(theta, axis):
    grad_theta_fn = autograd.grad(lambda th: objective_fn(center0, size0, eps0, th, axis))

    fd_val = finite_diff_theta(center0, size0, eps0, theta, axis, delta)
    ad_val = grad_theta_fn(theta)

    # keep for plotting
    plots[axis]["θ"].append(theta)
    plots[axis]["fd"].append(fd_val)
    plots[axis]["ad"].append(ad_val)

    # sanity & agreement checks
    assert np.isfinite(fd_val) and np.isfinite(ad_val)
    rel_diff = abs(fd_val - ad_val) / max(abs(fd_val), 1e-12)
    assert rel_diff < 0.3, f"axis={axis} θ={theta:.3f}: FD={fd_val:.4e}, AD={ad_val:.4e}"

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(
            os.path.join(RESULTS_DIR, f"axis_{AXIS_LABEL[axis]}_θ_{theta:.4f}.npz"),
            theta=float(theta),
            axis=int(axis),
            fd=float(fd_val),
            ad=float(ad_val),
        )


# ───── plot after pytest run ──────────────────────────────────────────────
def finalize_plotting():
    if not PLOT_RESULTS:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for axis in AXES:
        if not plots[axis]["θ"]:
            continue

        idx = np.argsort(plots[axis]["θ"])
        θ = np.array(plots[axis]["θ"])[idx]
        fd = np.array(plots[axis]["fd"])[idx]
        ad = np.array(plots[axis]["ad"])[idx]

        plt.figure(figsize=(6, 4))
        plt.plot(θ, fd, label="Finite diff", marker="o")
        plt.plot(θ, ad, label="Autograd", marker="s", linestyle="--")
        plt.xlabel("θ  [rad]")
        plt.ylabel("∂(intensity)/∂θ")
        plt.title(f"FD vs AD gradient (rotation about {AXIS_LABEL[axis]}‑axis)")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(RESULTS_DIR, f"grad_theta_fd_vs_ad_axis_{AXIS_LABEL[axis]}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[plot] saved ⇒ {fname}")


atexit.register(finalize_plotting)
