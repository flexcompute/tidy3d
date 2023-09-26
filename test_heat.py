import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import tidy3d as td

# basic parameters of simulation setups
r1 = 0.2
r0 = 0.5
r2 = 0.9
conductivity1 = 15
conductivity2 = 2
source1 = 1000
source2 = 300
temperature1=400
temperature2=300

# simulation domain parameters
scene_planar_center = (0.5, 0, 0)
scene_planar_size = (1, 0.1, 0.1)

medium1 = td.Medium(
    permittivity=2,
    heat_spec=td.SolidSpec(
        capacity=1,
        conductivity=conductivity1,
    ),
    name="solid1",
)

medium2 = td.Medium(
    permittivity=3,
    heat_spec=td.SolidSpec(
        capacity=1,
        conductivity=conductivity2,
    ),
    name="solid2",
)

background_medium = td.Medium(
    heat_spec=td.FluidSpec(),
    name="fluid",
)

planar_top_layer = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(2 * r2, 1, 1)),
    medium=medium2,
    name="top_layer",
)

planar_bottom_layer = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(2 * r0, 1, 1)),
    medium=medium1,
    name="bottom_layer",
)

planar_core = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(2 * r1, 1, 1)),
    medium=background_medium,
    name="core",
)

scene_planar = td.Scene(
    center=scene_planar_center,
    size=scene_planar_size,
    structures=[planar_top_layer, planar_bottom_layer, planar_core],
    medium=background_medium,
)


bc_side = td.HeatBoundarySpec(
    condition=td.HeatFluxBC(flux=0), 
    placement=td.SimulationBoundary(surfaces=["z+", "y-"]),
)

bc_top = td.HeatBoundarySpec(
    condition=td.TemperatureBC(temperature=temperature2), 
    placement=td.StructureBoundary(structure="top_layer"),
)

bc_bottom = td.HeatBoundarySpec(
    condition=td.TemperatureBC(temperature=temperature1), 
    placement=td.StructureStructureInterface(structures=["core", "bottom_layer"]),
)

source_bottom = td.UniformHeatSource(structures=["bottom_layer"], rate=source1)
source_top = td.UniformHeatSource(structures=["top_layer"], rate=source2)

temp_mnt = td.TemperatureMonitor(size=(td.inf, td.inf, 0), name="temperature")

heat_sim = td.HeatSimulation(
    scene=scene_planar,
    boundary_spec=[bc_top, bc_bottom, bc_side],
    sources=[source_bottom, source_top],
    grid_spec=td.UniformHeatGrid(dl=0.02),
    monitors=[temp_mnt],
)

from tidy3d.web.core.environment import Env
from tidy3d import web, HeatSimulation

Env.dev.active()
sim_data = web.run(heat_sim, "heat-test")
