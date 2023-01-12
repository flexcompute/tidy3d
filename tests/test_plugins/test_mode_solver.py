import pytest
import numpy as np
import matplotlib.pyplot as plt
import pydantic

import tidy3d as td

from tidy3d.plugins import DispersionFitter
from tidy3d.plugins import ModeSolver
from tidy3d.plugins.mode.solver import compute_modes
from tidy3d import FieldData, ScalarFieldDataArray, FieldMonitor
from tidy3d.plugins.smatrix.smatrix import Port, ComponentModeler
from tidy3d.plugins.smatrix.smatrix import ComponentModeler
from ..utils import clear_tmp, run_emulated

WAVEGUIDE = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0))
PLANE = td.Box(center=(0, 0, 0), size=(5, 0, 5))
SIM_SIZE = (5, 5, 5)
SRC = td.PointDipole(
    center=(0, 0, 0), source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13), polarization="Ex"
)


def test_mode_solver_simple():
    """Simple mode solver run (with symmetry)"""

    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(1, 0, -1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        filter_pol="tm",
        precision="double",
        track_freq="lowest",
    )
    ms = ModeSolver(
        simulation=simulation,
        plane=PLANE,
        mode_spec=mode_spec,
        freqs=[td.constants.C_0 / 0.9, td.constants.C_0 / 1.0, td.constants.C_0 / 1.1],
    )
    modes = ms.solve()


def test_mode_solver_custom_medium():
    """Test mode solver can work with custom medium. Consider a waveguide with varying
    permittivity along x-direction. The value of n_eff at different x position should be
    different.
    """

    # waveguide made of custom medium
    x_custom = np.linspace(-0.6, 0.6, 2)
    y_custom = [0]
    z_custom = [0]
    freq0 = td.constants.C_0 / 1.0
    n = np.array([1.5, 5])
    n = n[:, None, None, None]
    n_data = ScalarFieldDataArray(n, coords=dict(x=x_custom, y=y_custom, z=z_custom, f=[freq0]))
    mat_custom = td.CustomMedium.from_nk(n_data, interp_method="nearest")

    waveguide = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=mat_custom)
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[waveguide],
        run_time=1e-12,
    )
    mode_spec = td.ModeSpec(
        num_modes=1,
        precision="double",
    )

    plane_left = td.Box(center=(-0.5, 0, 0), size=(1, 0, 1))
    plane_right = td.Box(center=(0.5, 0, 0), size=(1, 0, 1))

    n_eff = []
    for plane in [plane_left, plane_right]:
        ms = ModeSolver(
            simulation=simulation,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )
        modes = ms.solve()
        n_eff.append(modes.n_eff.values)

    assert n_eff[0] < 1.5
    assert n_eff[1] > 4
    assert n_eff[1] < 5


def test_mode_solver_angle_bend():
    """Run mode solver with angle and bend and symmetry"""
    simulation = td.Simulation(
        size=SIM_SIZE,
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        symmetry=(-1, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3,
        bend_axis=0,
        angle_theta=np.pi / 3,
        angle_phi=np.pi,
        track_freq="highest",
    )
    # put plane entirely in the symmetry quadrant rather than sitting on its center
    plane = td.Box(center=(0, 0.5, 0), size=(1, 0, 1))
    ms = ModeSolver(
        simulation=simulation, plane=plane, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()
    # Plot field
    _, ax = plt.subplots(1)
    ms.plot_field("Ex", ax=ax, mode_index=1)

    # Create source and monitor
    st = td.GaussianPulse(freq0=1.0, fwidth=1.0)
    msource = ms.to_source(source_time=st, direction="-")
    mmonitor = ms.to_monitor(freqs=[1.0, 2.0], name="mode_mnt")


def test_mode_solver_2D():
    """Run mode solver in 2D simulations."""
    mode_spec = td.ModeSpec(
        num_modes=3,
        filter_pol="te",
        precision="double",
        num_pml=(0, 10),
        track_freq="central",
    )
    simulation = td.Simulation(
        size=(0, SIM_SIZE[1], SIM_SIZE[2]),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()

    mode_spec = td.ModeSpec(num_modes=3, filter_pol="te", precision="double", num_pml=(10, 0))
    simulation = td.Simulation(
        size=(SIM_SIZE[0], SIM_SIZE[1], 0),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[WAVEGUIDE],
        run_time=1e-12,
        sources=[SRC],
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()

    # The simulation and the mode plane are both 0D along the same dimension
    simulation = td.Simulation(
        size=PLANE.size,
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[SRC],
    )
    ms = ModeSolver(
        simulation=simulation, plane=PLANE, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()


def test_compute_modes():
    """Test direct call to ``compute_modes`` (used in e.g. gdsfactory)."""
    eps_cross = np.random.rand(10, 10)
    coords = np.arange(11)
    mode_spec = td.ModeSpec(num_modes=3, target_neff=2.0)
    modes = compute_modes(
        eps_cross=[eps_cross] * 3,
        coords=[coords, coords],
        freq=td.constants.C_0 / 1.0,
        mode_spec=mode_spec,
    )
