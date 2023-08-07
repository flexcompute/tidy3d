import boto3

import tidy3d.plugins.mode.web as msweb
from tidy3d.plugins.mode import ModeSolver
from tidy3d.web.environment import Env



# tidy3d imports
import tidy3d as td
from tidy3d import web

Env.dev.active()
WAVEGUIDE = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0))
PLANE = td.Box(center=(0, 0, 0), size=(5, 0, 5))
SIM_SIZE = (5, 5, 5)
SRC = td.PointDipole(
    center=(0, 0, 0), source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13), polarization="Ex"
)

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
    precision="single",
    track_freq="lowest",
)
ms = ModeSolver(
    simulation=simulation,
    plane=PLANE,
    mode_spec=mode_spec,
    freqs=[td.C_0 / 0.9, td.C_0 / 1.0, td.C_0 / 1.1],
)
msweb.run(ms)

