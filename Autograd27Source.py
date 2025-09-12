#
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
FREQ0 = 200e12
FWIDTH = 20e12
DL = 0.05
LY = 0.0  # 2D simulation along y (suppressed)
LZ = 4.0
BAR_WIDTH = 0.5
BAR_HEIGHT = 0.2
EPS_SI = 12.11  # ~ (n=3.48)^2 constant-permittivity silicon
NUM_BARS = 8
BAR_SPACING = 0.6
LX = (BAR_SPACING + BAR_WIDTH) * NUM_BARS + BAR_SPACING
SILICON = td.Medium(permittivity=EPS_SI)
RUN_TIME = 5e-12
MNT_SIZE_Z = 2 * DL


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

    spc_left = -LX / 2 + BAR_SPACING + BAR_WIDTH / 2
    base_centers_x = spc_left + np.arange(NUM_BARS) * (BAR_SPACING + BAR_WIDTH)

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
    )

    # Monitor just past the grating
    fld1 = td.FieldMonitor(
        center=(0.0, 0.0, +0.5 * BAR_HEIGHT + 0.2),
        size=(LX, 0.0, MNT_SIZE_Z),
        freqs=[FREQ0],
        name="fld1",
        colocate=False,
    )

    sim = td.Simulation(
        size=(LX, LY, LZ),
        grid_spec=td.GridSpec.uniform(dl=DL),
        structures=structures,
        sources=[src],
        monitors=[fld1],
        run_time=RUN_TIME,
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True),  # No PML in y (periodic)
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
        center=(0.0, 0.0, +0.5 * BAR_HEIGHT + 0.2),
        size=(LX, td.inf, 0),
    )

    # Downstream monitor after grating 2
    fld2 = td.FieldMonitor(
        center=(0.0, 0.0, +0.5 * BAR_HEIGHT + 0.4),
        size=(0.0, 0.0, 0.0),
        freqs=[FREQ0],
        name="fld2",
    )

    sim = td.Simulation(
        size=(LX, LY, LZ),
        grid_spec=td.GridSpec.uniform(dl=DL),
        structures=structures,
        sources=[cfs],
        monitors=[fld2],
        run_time=RUN_TIME,
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True),  # No PML in y (periodic)
    )
    return sim


def figure_of_merit_from_field(sim_data: td.SimulationData) -> anp.ndarray:
    """Compute |Ex|^2 at (x0,y0)≈center for freq index 0. Autograd-friendly."""
    # Indices at center of each axis

    intensity = sim_data.get_intensity("fld2")
    return anp.sum(intensity.values)


def objective(p: anp.ndarray) -> anp.ndarray:
    """Objective wiring sim1 → CustomFieldSource in sim2 → FOM."""
    # Ensure array type for autograd

    # Sim 1
    sim1 = make_sim1(p)
    data1 = run(sim1, task_name="autograd27_sim1", local_gradient=True)
    fld1 = data1.load_field_monitor("fld1")

    # Sim 2
    sim2 = make_sim2(fld1_dataset=fld1, p=p)
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

    p0 = anp.zeros((NUM_BARS,)) + np.random.randn(NUM_BARS) * 0.1
    demo_plots(p0)
    plt.show()


if __name__ == "__main__":
    p0 = anp.zeros((NUM_BARS,)) + np.random.randn(NUM_BARS) * 0.1
    J = objective(p0)
    print(J)
    grad = ag.grad(objective)(p0)
    print(grad)
