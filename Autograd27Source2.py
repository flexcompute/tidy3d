"""
Autograd27Source2: Minimal single-simulation demo where design parameters directly set
the amplitudes of a CustomFieldSource over an x–z plane. Geometry is fixed; only the
source field values are optimized. Uses autograd-enabled web.run with MCP worker group.
"""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np

import tidy3d as td
from tidy3d.web import run

# ---------------------------
# Configuration
# ---------------------------

LAMBDA0 = 1.55
FREQ0 = td.C_0 / LAMBDA0
FWIDTH = FREQ0 / 10.0

LY = 0.0  # 2D simulation along y (suppressed)
BAR_WIDTH = 1.7
BAR_HEIGHT = 1.4
EPS_SI = 4.0  # ~ (n=3.48)^2 constant-permittivity silicon
NUM_BARS = 8
BAR_SPACING = 2.3
LX = (BAR_SPACING + BAR_WIDTH) * NUM_BARS + 3 * BAR_SPACING
SILICON = td.Medium(permittivity=EPS_SI)
RUN_TIME = 100 / FWIDTH
MNT_SIZE_Z = LAMBDA0 / 4
SPC_ABOVE_GRATING = 5.0
LZ = SPC_ABOVE_GRATING + BAR_HEIGHT + SPC_ABOVE_GRATING + 2 * LAMBDA0
OBS_CENTER_Z = SPC_ABOVE_GRATING + BAR_HEIGHT / 2
SRC_CENTER_Z = -OBS_CENTER_Z
PML_X = True
STEPS_PER_WVL = 30

PLOT_SIMS = False

# Plotting helpers
PLOT_GAUSSIANS = True

# Dense sampling for CustomFieldSource along x and Gaussian basis width
SAMPLES_PER_LAMBDA_X = 4  # samples per wavelength along x for the source dataset
GAUSS_SIGMA = 5 * LAMBDA0  # Gaussian width (in length units)

# Use MCP worker group for API calls
# MCP_WORKER_GROUP = "flexagent-dev3"


def build_fixed_grating(z_center: float) -> list[td.Structure]:
    """Create a fixed 1D grating of Si bars (no dependence on params)."""
    base_centers_x = np.arange(NUM_BARS) * (BAR_SPACING + BAR_WIDTH)
    base_centers_x = base_centers_x - np.mean(base_centers_x)

    structures: list[td.Structure] = []
    for cx in base_centers_x:
        geom = td.Box(center=(cx, 0.0, z_center), size=(BAR_WIDTH, td.inf, BAR_HEIGHT))
        structures.append(td.Structure(geometry=geom, medium=SILICON))
    return structures


def make_source(params: anp.ndarray) -> td.CustomFieldSource:
    """Build a CustomFieldSource parameterized by Gaussian bases along x.

    - Keeps the number of design parameters small (len(params)).
    - Uses a denser sampling along x for the FieldDataset, targeting
      SAMPLES_PER_LAMBDA_X samples per wavelength.
    - Each parameter controls the amplitude of a Gaussian centered at one of
      the evenly spaced control points across the source width.
    """
    params = anp.atleast_1d(params)
    num_params = int(params.size)

    # Dense x sampling for the dataset
    nx_dense = max(2, int(np.ceil(LX / LAMBDA0 * SAMPLES_PER_LAMBDA_X)))
    xs = np.linspace(-LX / 2, LX / 2, nx_dense)
    ys = np.array([0.0])  # y is the injection axis (size[y]==0), coords relative to source center
    zs = np.array([0.0])
    fs = np.array([FREQ0])

    # Gaussian basis centers across x and amplitudes from params
    # Ensure a 1-sigma margin from both boundaries for the first and last centers
    margin = float(GAUSS_SIGMA)
    left = -LX / 2 + margin
    right = LX / 2 - margin
    if right <= left:
        # Fallback if sigma is too large: place centers across the full span
        left, right = -LX / 2, LX / 2
    x_centers = anp.linspace(left, right, num_params)
    x_arr = anp.asarray(xs)
    dx = x_arr[None, :] - x_centers[:, None]
    gaussians = anp.exp(-0.5 * (dx / GAUSS_SIGMA) ** 2)
    amp_x = anp.sum(params[:, None] * gaussians, axis=0)

    data = anp.reshape(amp_x, (nx_dense, 1, 1, 1))
    sfd = td.ScalarFieldDataArray(data, coords={"x": xs, "y": ys, "z": zs, "f": fs})

    # from autograd.tracer import getval
    # # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.plot(getval(data.squeeze().real))
    # plt.plot(getval(data.squeeze().imag))
    # plt.legend()
    # plt.show()

    field_dataset = td.FieldDataset(Ex=sfd, Ey=sfd, Ez=sfd)

    src = td.CustomFieldSource(
        center=(0.0, 0.0, SRC_CENTER_Z),
        size=(LX, 0.0, MNT_SIZE_Z),  # planar x–z source (y is injection axis)
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        field_dataset=field_dataset,
    )
    return src


def make_sim(params: anp.ndarray) -> td.Simulation:
    """Single simulation with fixed grating and a CustomFieldSource parameterized by params."""
    structures = build_fixed_grating(z_center=0.0)
    src = make_source(params)

    # Observation point (intensity objective)
    fld = td.FieldMonitor(
        center=(0.0, 0.0, OBS_CENTER_Z),
        size=(0.0, 0.0, 0.0),
        freqs=[FREQ0],
        name="fld",
        colocate=True,
        fields=["Ex", "Ey", "Ez"],
    )

    sim = td.Simulation(
        size=(LX, LY, LZ),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=STEPS_PER_WVL),
        structures=structures,
        sources=[src],
        monitors=[fld],
        run_time=RUN_TIME,
        boundary_spec=td.BoundarySpec.pml(x=PML_X, y=False, z=True),
    )
    return sim


def objective(params: anp.ndarray) -> anp.ndarray:
    """Simple intensity objective at the observation point."""
    sim = make_sim(params)
    data = run(
        sim,
        task_name="autograd27_single_sim",
        local_gradient=True,
    )
    intensity = data.get_intensity("fld")
    return intensity.values.item()


if __name__ == "__main__":
    # Design: set number of x samples for the source plane
    NX = 8
    p0 = anp.ones((NX,))

    if PLOT_SIMS:
        import matplotlib.pyplot as plt

        sim = make_sim(p0)
        sim.plot(y=0)
        plt.show()

    # Optional: visualize Gaussian basis functions on the x-plane
    if PLOT_GAUSSIANS:
        import matplotlib.pyplot as plt

        # Dense sampling used by the source dataset
        nx_dense = max(2, int(np.ceil(LX / LAMBDA0 * SAMPLES_PER_LAMBDA_X)))
        xs = np.linspace(-LX / 2, LX / 2, nx_dense)
        # Ensure a 1-sigma margin from both boundaries for plotting centers
        margin = float(GAUSS_SIGMA)
        left = -LX / 2 + margin
        right = LX / 2 - margin
        if right <= left:
            left, right = -LX / 2, LX / 2
        x_centers = np.linspace(left, right, NX)
        dx = xs[None, :] - x_centers[:, None]
        gaussians = np.exp(-0.5 * (dx / GAUSS_SIGMA) ** 2)

        fig_g, ax_g = plt.subplots(1, 1, figsize=(7, 3.5))
        # plot each Gaussian scaled by its parameter
        for i in range(NX):
            ax_g.plot(xs, np.asarray(p0[i]) * gaussians[i], label=f"g{i}")
        amp_x = (np.asarray(p0)[:, None] * gaussians).sum(axis=0)
        ax_g.plot(xs, amp_x, "k--", lw=2, label="sum")
        ax_g.set_title("Gaussian basis on source plane (scaled by params)")
        ax_g.set_xlabel("x")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(ncol=2, fontsize=8)

    J, dJ = ag.value_and_grad(objective)(p0)
    print("Objective:", float(J))
    print("Adjoint grad (first 8):", np.asarray(dJ)[:8])

    if np.all(dJ == 0) or any(np.isnan(dJ)):
        print("Adjoint grad is all zeros - this suggests an architectural issue!")
        print(
            "   The issue is likely that no field components are being found in derivative_info.E_adj"
        )
        print("   Please check the CustomFieldSource implementation.")
        exit()

    # Finite-difference gradient across all parameters
    delta = 1e-3
    grad_fd = np.zeros_like(np.asarray(p0), dtype=float)
    base = np.asarray(p0, dtype=float)
    for i in range(grad_fd.size):
        p_plus = base.copy()
        p_minus = base.copy()
        p_plus[i] += delta
        p_minus[i] -= delta
        Jp = objective(p_plus)
        Jm = objective(p_minus)
        grad_fd[i] = (float(Jp) - float(Jm)) / (2.0 * delta)

    # Normalize and compare
    g_adj = np.asarray(dJ, dtype=float)
    n_adj = np.linalg.norm(g_adj) or 1.0
    n_fd = np.linalg.norm(grad_fd) or 1.0
    g_adj_n = g_adj / n_adj
    g_fd_n = grad_fd / n_fd

    rms_error = np.linalg.norm(g_adj_n - g_fd_n)
    print("FD grad:", grad_fd)
    print("Adj grad:", g_adj)
    print(f"RMS error (normalized): {rms_error:.3e}")

    # Plot FD vs Adjoint on the same axes
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    ax.plot(g_fd_n, "o-", label="FD (normalized)")
    ax.plot(g_adj_n, "o-", label="Adjoint (normalized)")
    ax.set_title("Gradient comparison")
    ax.set_xlabel("parameter index")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
