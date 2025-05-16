"""
FD vs AD checks for every affine parameter of a dielectric box:
      • centre      c = (cx,cy,cz)
      • size        a = (ax,ay,az)
      • rotation    θ about each axis
      • scale       s = (sx,sy,sz)
      • translation t = (tx,ty,tz)
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
from autograd import tuple

# ───────── switches ───────────────────────────────────────────────────────
SAVE = False  # save raw .npz
PLOT = True  # make png plots
OUT = "./fd_ad_all_results"

# ───────── physical / geometric constants ─────────────────────────────────
λ = 1.5
f0 = td.C_0 / λ
Lbox = 10 * λ
buffer = 1.0 * λ
Tsim = 120 / f0

# baseline shape parameters  ( **plain Python lists / tuples** )
center0 = [0.0, 0.0, 0.0]
size0 = [2.0, 2.0, 2.0]
eps_box = 2.0

theta0 = np.pi / 4  # baseline rotation (x-axis)
axis0 = 0
scale0 = [1.2, 1.3, 0.9]
trans0 = [0.45, 0.20, 0.30]

# finite-difference steps
Δθ = 0.015
Δxyz = 0.03

AXES = (0, 1, 2)
LAB = {0: "x", 1: "y", 2: "z"}

plots = defaultdict(list)


# ───────── simulation builder ─────────────────────────────────────────────
def make_simulation(center, size, scale, trans, theta, axis):
    """Return a Tidy3D Simulation with the requested affine parameters."""
    src = td.PointDipole(
        center=(-Lbox / 2 + 0.5 * buffer, 0, 0),
        source_time=td.GaussianPulse(freq0=f0, fwidth=f0 / 10),
        polarization="Ez",
    )

    mon = td.FieldMonitor(
        center=(+Lbox / 2 - 0.5 * buffer, 2.0 * buffer, 2.0 * buffer),
        size=(0, 0, 0),
        freqs=[f0],
        name="m",
    )

    geom = (
        td.Box(center=tuple(center), size=tuple(size))
        .rotated(theta, axis=axis)
        .scaled(x=scale[0], y=scale[1], z=scale[2])
        .translated(x=trans[0], y=trans[1], z=trans[2])
    )
    struct = td.Structure(geometry=geom, medium=td.Medium(permittivity=eps_box))

    return td.Simulation(
        size=(Lbox, Lbox, Lbox),
        run_time=Tsim,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=50),
        sources=[src],
        monitors=[mon],
        structures=[struct],
    )


def objective(c, a, s, t, θ, ax):
    sim = make_simulation(c, a, s, t, θ, ax)
    data = web.run(sim, task_name="fd_ad_all", verbose=False, local_gradient=True)
    return anp.sum(data.get_intensity("m").values)


def finite_diff(fun, x0, δ):
    return (fun(x0 + δ) - fun(x0 - δ)) / (2 * δ)


def _assert_close(fd_val, ad_val, tag, axis=None, extra="", tol=0.35):
    """
    Raise AssertionError if |FD-AD|/max(|FD|,1e-12) >= tol.
    """
    rel = abs(fd_val - ad_val) / max(abs(fd_val), 1e-12)
    if rel >= tol:
        ax_lbl = {0: "x", 1: "y", 2: "z"}.get(axis, "")
        axis_str = f"_{ax_lbl}" if ax_lbl else ""
        raise AssertionError(
            f"{tag}{axis_str}{extra}: FD–AD mismatch "
            f"(rel diff {rel:.2%}) | FD={fd_val:.4e}, AD={ad_val:.4e}"
        )


#  1.  ROTATION θ_k   (k = 0,1,2)
θ_vals = np.array([0, np.pi / 4, np.pi / 2])


@pytest.mark.numerical
@pytest.mark.parametrize("k", AXES, ids=[f"θ_{LAB[a]}" for a in AXES])
@pytest.mark.parametrize("θ", θ_vals)
def test_fd_vs_ad_rotation(k, θ):
    f = lambda th: objective(center0, size0, scale0, trans0, th, k)
    g_ad = autograd.grad(f)(θ)
    g_fd = finite_diff(f, θ, Δθ)

    plots[("θ", k, "fd")].append((θ, g_fd))
    plots[("θ", k, "ad")].append((θ, g_ad))

    assert np.isfinite(g_fd) and np.isfinite(g_ad)
    _assert_close(g_fd, g_ad, tag="θ", axis=k)


# 2. SCALE  s_k
@pytest.mark.numerical
@pytest.mark.parametrize("k", AXES, ids=[f"s_{LAB[a]}" for a in AXES])
def test_fd_vs_ad_scale(k):
    def f(sk):
        s = list(scale0)
        s[k] = sk
        return objective(center0, size0, s, trans0, theta0, axis0)

    g_ad = autograd.grad(f)(scale0[k])
    g_fd = finite_diff(f, scale0[k], Δxyz)

    plots[("s", k, "fd")].append(g_fd)
    plots[("s", k, "ad")].append(g_ad)

    assert np.isfinite(g_fd) and np.isfinite(g_ad), "NaN/Inf in FD or AD result"
    _assert_close(g_fd, g_ad, tag="scale", axis=k)


# 3. TRANSLATION  t_k
@pytest.mark.numerical
@pytest.mark.parametrize("k", AXES, ids=[f"t_{LAB[a]}" for a in AXES])
def test_fd_vs_ad_translation(k):
    def f(tk):
        t = list(trans0)
        t[k] = tk
        return objective(center0, size0, scale0, t, theta0, axis0)

    g_ad = autograd.grad(f)(trans0[k])
    g_fd = finite_diff(f, trans0[k], Δxyz)

    plots[("t", k, "fd")].append(g_fd)
    plots[("t", k, "ad")].append(g_ad)

    assert np.isfinite(g_fd) and np.isfinite(g_ad), "NaN/Inf in FD or AD result"
    _assert_close(g_fd, g_ad, tag="trans", axis=k)


# # 4. SIZE  a_k
@pytest.mark.numerical
@pytest.mark.parametrize("k", AXES, ids=[f"a_{LAB[a]}" for a in AXES])
def test_fd_vs_ad_size(k):
    def f(ak):
        a = list(size0)
        a[k] = ak
        return objective(center0, a, scale0, trans0, theta0, axis0)

    g_ad = autograd.grad(f)(size0[k])
    g_fd = finite_diff(f, size0[k], Δxyz)

    plots[("a", k, "fd")].append(g_fd)
    plots[("a", k, "ad")].append(g_ad)

    assert np.isfinite(g_fd) and np.isfinite(g_ad), "NaN/Inf in FD or AD result"
    _assert_close(g_fd, g_ad, tag="size", axis=k)


# # 5. CENTRE  c_k
@pytest.mark.numerical
@pytest.mark.parametrize("k", AXES, ids=[f"c_{LAB[a]}" for a in AXES])
def test_fd_vs_ad_center(k):
    def f(ck):
        c = list(center0)
        c[k] = ck
        return objective(c, size0, scale0, trans0, theta0, axis0)

    g_ad = autograd.grad(f)(center0[k])
    g_fd = finite_diff(f, center0[k], Δxyz)

    plots[("c", k, "fd")].append(g_fd)
    plots[("c", k, "ad")].append(g_ad)

    assert np.isfinite(g_fd) and np.isfinite(g_ad), "NaN/Inf in FD or AD result"
    _assert_close(g_fd, g_ad, tag="center", axis=k)


# ───────── save / plot after test run ────────────────────────────
def _save_and_plot():
    if not (SAVE or PLOT):
        return

    os.makedirs(OUT, exist_ok=True)

    if SAVE:
        np.savez_compressed(os.path.join(OUT, "gradients.npz"), **plots)

    labels, errs = [], []

    for k in AXES:
        fd_pairs = plots.get(("θ", k, "fd"), [])
        ad_pairs = plots.get(("θ", k, "ad"), [])
        for (theta, fd), (_, ad) in zip(sorted(fd_pairs), sorted(ad_pairs)):
            lbl = f"θ_{LAB[k]} {theta:.2f}"
            rel = abs(fd - ad) / max(abs(fd), 1e-12)
            labels.append(lbl)
            errs.append(rel)

    for tag, pretty in zip(("s", "t", "a", "c"), ("scale", "trans", "size", "center")):
        for k in AXES:
            fd_vals = plots.get((tag, k, "fd"), [])
            ad_vals = plots.get((tag, k, "ad"), [])
            if fd_vals:
                rel = abs(fd_vals[0] - ad_vals[0]) / max(abs(fd_vals[0]), 1e-12)
                labels.append(f"{pretty}_{LAB[k]}")
                errs.append(rel)

    if not (PLOT and errs):
        return

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(errs)), errs)
    plt.xticks(range(len(errs)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("relative |FD – AD|  /  max(|FD|)")
    plt.title("FD vs AD relative errors for all parameters")
    plt.tight_layout()
    fname = os.path.join(OUT, "bar_rel_error_all_params.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[plot] saved {fname}")


atexit.register(_save_and_plot)
