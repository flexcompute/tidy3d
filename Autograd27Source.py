# # #!/usr/bin/env python3
"""
Autograd27Source: Two-simulation demo wiring FieldMonitor → CustomFieldSource.

Workflow:
  f(p):
    1) sim1(p): grating of Si boxes; PlaneWave; FieldMonitor → fld1
    2) sim2(fld1, p): second grating; CustomFieldSource(field_dataset=fld1); FieldMonitor → fld2
    3) objective(fld2): simple |Ex|^2 at a point (autograd-friendly)

Notes:
  - Assumes autograd VJP rules implemented for CustomFieldSource and geometry.
  - Uses tidy3d.web.run(..., local_gradient=True) to enable local gradients.
  - Provides geometry plots for both simulations.
"""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp

# import matplotlib.pyplot as plt
import numpy as np

import tidy3d as td
from tidy3d.web import run

# ---------------------------
# Configuration (feel free to tweak)
# ---------------------------

LAMBDA0 = 1.55
FREQ0 = td.C_0 / LAMBDA0
FWIDTH = FREQ0 / 10.0
DL = 0.02
LY = 0.0  # 2D simulation along y (suppressed)
BAR_WIDTH = 1.4
BAR_HEIGHT = 0.8
EPS_SI = 4.0  # ~ (n=3.48)^2 constant-permittivity silicon
NUM_BARS = 8
BAR_SPACING = 2.3
LX = (BAR_SPACING + BAR_WIDTH) * NUM_BARS + 3 * BAR_SPACING
SILICON = td.Medium(permittivity=EPS_SI)
RUN_TIME = 100 / FWIDTH
MNT_SIZE_Z = 2 * DL
SPC_ABOVE_GRATING = 2.0
LZ = SPC_ABOVE_GRATING + BAR_HEIGHT + SPC_ABOVE_GRATING + 2 * LAMBDA0
FLD1_CENTER_Z = SPC_ABOVE_GRATING + BAR_HEIGHT / 2
SRC2_CENTER_Z = -FLD1_CENTER_Z
PML_X = True

PLOT_SIMS = False


def build_grating_structures_from_params(
    p: anp.ndarray,
    z_center: float,
) -> list[td.Structure]:
    """Create a 1D grating of Si bars whose x-centers are controlled by p.

    Parameters
    ----------
    p : anp.ndarray
        Offsets applied to base equally spaced x-centers (shape: [N]).
    z_center : float
        Common z center of all bars.
    """

    # spc_left = -LX / 2 + BAR_SPACING + BAR_WIDTH / 2
    base_centers_x = np.arange(NUM_BARS) * (BAR_SPACING + BAR_WIDTH)
    base_centers_x = base_centers_x - np.mean(base_centers_x)

    centers_x = base_centers_x + p

    structures: list[td.Structure] = []
    for cx in centers_x:
        geom = td.Box(center=(cx, 0.0, z_center), size=(BAR_WIDTH, td.inf, BAR_HEIGHT))
        structures.append(td.Structure(geometry=geom, medium=SILICON))
    return structures


def make_sim1(p: anp.ndarray) -> td.Simulation:
    """Simulation 1: Grating with PlaneWave and FieldMonitor (fld1)."""

    # Geometry: bars centered near z=0, spanning y fully
    structures = build_grating_structures_from_params(p=p, z_center=0.0)

    # Source: plane wave from -z to +z
    src = td.PlaneWave(
        center=(0.0, 0.0, -0.5 * LZ + 0.25),
        size=(LX, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        direction="+",
        pol_angle=np.pi / 2,
    )

    # Monitor just past the grating
    fld1 = td.FieldMonitor(
        center=(0.0, 0.0, FLD1_CENTER_Z),
        size=(LX, 0.0, MNT_SIZE_Z),
        freqs=[FREQ0],
        name="fld1",
        colocate=False,
        fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
    )

    sim = td.Simulation(
        size=(LX, LY, LZ),
        # grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
        grid_spec=td.GridSpec.uniform(dl=DL),
        structures=structures,
        sources=[src],
        monitors=[fld1],
        run_time=RUN_TIME,
        boundary_spec=td.BoundarySpec.pml(x=PML_X, y=False, z=True),
    )
    return sim


def make_sim2(fld1_dataset: td.FieldDataset, p: anp.ndarray) -> td.Simulation:
    """Simulation 2: Second grating and CustomFieldSource made from sim1's fld1."""
    # Build second grating (keep geometry independent of p for now)
    structures = build_grating_structures_from_params(
        p=0 * p,
        z_center=0.0,
    )

    # Convert FieldData from sim1 to a proper CustomFieldSource using helper
    # Ensure dataset coords are relative to source center
    cfs = fld1_dataset.to_source(
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        center=(0, 0, SRC2_CENTER_Z),
        size=tuple(fld1_dataset.monitor.size),
    )

    # Downstream planar xz monitor after grating 2 (for objective)
    fld2 = td.FieldMonitor(
        center=(0.0, 0.0, FLD1_CENTER_Z),
        size=(0.0, 0.0, 0.0),
        freqs=[FREQ0],
        name="fld2",
        colocate=True,
        fields=["Ex", "Ey", "Ez"],
    )

    sim = td.Simulation(
        size=(LX, LY, LZ),
        grid_spec=td.GridSpec.uniform(dl=DL),
        structures=structures,
        sources=[cfs],
        monitors=[fld2],
        run_time=RUN_TIME,
        boundary_spec=td.BoundarySpec.pml(x=PML_X, y=False, z=True),
    )
    return sim


def figure_of_merit_from_field(sim_data: td.SimulationData) -> anp.ndarray:
    """Compute planar intensity on fld2: sum(|Ex|^2 + |Ez|^2)."""
    intensity = sim_data.get_intensity("fld2")
    return intensity.values.item() * 1e5


def objective(p: anp.ndarray) -> anp.ndarray:
    """Objective wiring sim1 → CustomFieldSource in sim2 → FOM."""
    # Ensure array type for autograd
    import matplotlib.pyplot as plt

    # Sim 1
    sim1 = make_sim1(p)
    if PLOT_SIMS:
        sim1.plot(y=0)
        plt.show()

    data1 = run(sim1, task_name="autograd27_sim1", local_gradient=True)
    fld1 = data1["fld1"]

    # Sim 2
    sim2 = make_sim2(fld1_dataset=fld1, p=p)
    if PLOT_SIMS:
        sim2.plot(y=0)
        plt.show()

    data2 = run(sim2, task_name="autograd27_sim2", local_gradient=True)

    # FOM
    return figure_of_merit_from_field(data2)


# ---------------------------
# Plotting helpers
# ---------------------------
def plot_sim_overview(sim: td.Simulation, title: str, y: float = 0.0) -> None:
    """Plot xz-plane (y fixed) cross-section of the simulation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    sim.plot(y=y, ax=ax, monitor_alpha=0.7, source_alpha=0.7)
    ax.set_title(title)
    plt.tight_layout()


def _make_dummy_field_dataset(nx: int = 21, ny: int = 1) -> td.FieldDataset:
    """Small planar FieldDataset at f=FREQ0 to instantiate a CustomFieldSource for plotting."""
    xs = np.linspace(-LX / 2, LX / 2, nx)
    ys = [0]
    zs = np.array([0.0])
    fs = np.array([FREQ0])
    data = np.ones((nx, 1, 1, 1))
    sfd = td.ScalarFieldDataArray(data, coords={"x": xs, "y": ys, "z": zs, "f": fs})
    return td.FieldDataset(Ex=sfd)


def demo_plots(p: anp.ndarray) -> None:
    """Generate basic geometry plots for both simulations (no runs)."""
    sim1 = make_sim1(p)
    dummy_ds = _make_dummy_field_dataset()
    sim2 = make_sim2(fld1_dataset=dummy_ds, p=p)
    plot_sim_overview(sim1, title="Sim1: Grating + PlaneWave (xz)", y=0.0)
    plot_sim_overview(sim2, title="Sim2: Grating + CustomFieldSource (xz)", y=0.0)


def main() -> None:
    # Parameters: N bars with zero initial offsets
    import matplotlib.pyplot as plt

    p0 = anp.zeros((NUM_BARS,))
    demo_plots(p0)
    plt.show()


if __name__ == "__main__":
    p0 = anp.zeros((NUM_BARS,))
    J, grad_adj = ag.value_and_grad(objective)(p0)
    print(J, grad_adj)

    # Numerical finite-difference gradient (centered difference)
    delta = DL
    grad_fd = np.zeros_like(np.asarray(p0), dtype=float)
    base = np.asarray(p0, dtype=float)
    for i in range(grad_fd.size):
        p_plus = base.copy()
        p_minus = base.copy()
        p_plus[i] += delta
        p_minus[i] -= delta
        J_plus = objective(p_plus)
        J_minus = objective(p_minus)
        grad_fd[i] = (float(J_plus) - float(J_minus)) / (2.0 * delta)

    print(grad_fd)
    print(grad_adj)

    _grad_fd = grad_fd / np.linalg.norm(grad_fd)
    _grad_adj = grad_adj / np.linalg.norm(grad_adj)

    rms_error = np.linalg.norm(_grad_fd - _grad_adj)
    print(f"RMS error: {rms_error:.3e}")

    import matplotlib.pyplot as plt

    plt.plot(_grad_fd, label="finite-difference")
    plt.plot(_grad_adj, label="adjoint")
    plt.legend()
    plt.show()


"""
Adjoint gradient:
[-33.86037943,  15.75518254,  16.47161767, -12.1026643, -1.94251531, -5.80539923, -21.05272119, 7.48395687]
[-6.15052789, 2.68140578, -5.99104248, -3.37864877,  7.94693111, -0.25750337, -3.26601009,  1.38588731]
[7491.82491336, -5183.32123377,  2763.5461682,   3608.17932507, -3608.18056444, -2763.54612709,  5183.32494262, -7491.82452151]

Finite-difference gradient:
[-850.95214844  605.45349121 1184.17358398  271.27075195   54.67224121 -234.40551758 -685.66894531  260.52856445]
[-368.19458008,  -45.47119141,  608.06274414, 1076.81274414, -557.86132812, -675.20141602, -140.22827148,  223.23608398]

"""
