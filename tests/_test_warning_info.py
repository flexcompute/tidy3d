# standard python imports
import numpy as np
import matplotlib.pyplot as plt

# tidy3d imports
import tidy3d as td
import tidy3d.web as web
import pydantic
from tidy3d import log

# Define material properties
medium = td.Medium(permittivity=3)

wavelength = 1
f0 = td.C_0 / wavelength / np.sqrt(medium.permittivity)

# Set the domain size in x, y, and z
domain_size = 12 * wavelength

# create the geometry
geometry = []

# construct simulation size array
sim_size = (domain_size, domain_size, domain_size)

# Bandwidth in Hz
fwidth = f0 / 10.0

source_time = td.GaussianPulse(freq0=f0, fwidth=fwidth)

run_time = 10 / fwidth

freqs = np.linspace(f0 - fwidth, f0 + fwidth, 11)

dipole = td.PointDipole(
    center=(0, 0, 0),
    source_time=source_time,
    polarization="Ex",
)

monitor_xz_time = td.FieldTimeMonitor(
    center=(0, 0, 0), size=(domain_size, 0, domain_size), interval=50, name="xz_time"
)

mode_mnt = td.ModeMonitor(
    center=(0, 0, 0),
    size=(domain_size, 0, domain_size),
    freqs=list(freqs),
    mode_spec=td.ModeSpec(num_modes=3, group_index_step=1),
    name="mode",
)

mode_source = td.ModeSource(
    size=(domain_size, 0, domain_size),
    source_time=source_time,
    mode_spec=td.ModeSpec(num_modes=2, group_index_step=1),
    mode_index=1,
    num_freqs=50,
    direction='-'
)

monitor_flux = td.FluxMonitor(
    center=(0, 0, 0),
    size=(8, 8, 8),
    freqs=freqs,
    name="flux",
    normal_dir="+",  # warning
)

box1 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(11.5, 11.5, 11.5)),  # warning
    medium=td.Medium(permittivity=4),
)

box2 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(5, 5, 5)),
    medium=td.Medium(permittivity=5),
)

box3 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
    medium=td.Medium(permittivity=6),
)


# define a basic boundary spec setting PML in all directions
bspec_pml = td.BoundarySpec.all_sides(boundary=td.PML())

source = td.GaussianBeam(
   center=(4, 0, 0),
   size=(0, 8, 9),
   waist_radius=2.0,
   waist_distance=1,
   source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth, amplitude=1),
   direction="+",
   angle_theta=0,
   angle_phi=0,
   pol_angle=0,
   num_freqs=30,  # warning
)

sim = td.Simulation(
   size=sim_size,
   sources=[source, mode_source, mode_source, mode_source],
   structures=[box1, box2, box3],
   monitors=[monitor_xz_time, monitor_flux, mode_mnt],
   run_time=run_time,
   boundary_spec=bspec_pml,
   grid_spec=td.GridSpec.uniform(dl=0.04),
   shutoff=1e-10,
)

# write and load to trigger whole hierarchy parsing
sim.to_file("test_sim.json")

log.set_capture(True)
sim = td.Simulation.from_file("test_sim.json")
warning_list = log.captured_warnings()
for w in warning_list:
    print(w)
log.set_capture(False)

