#!/usr/bin/env -S poetry run python
# ruff: noqa

# # Grounded co-planar waveguide with via fence

# Co-planar waveguide (CPWs) transmission lines offer many advantages, including better EM shielding, ease of fabrication, and lower radiation loss when compared to microstrips. However, conventional CPWs can be susceptible to coupling to higher-order modes in nearby structures, thus introducing undesirable resonances that restrict the operating bandwidth of the CPW. In particular, at high frequencies, the side ground planes in a conventional CPW can support higher-order resonances that radiate similarly to a patch antenna.
#
# One solution is to introduce via fences on either side of the signal line. This reduces the transverse electrical size of the CPW and "pushes" the resonances out beyond the design bandwidth of the CPW. It also provides the added benefit of better shielding from EM interference.
#
# In this notebook, we will simulate some of the designs investigated in this paper:
#
# * Sain, Arghya, and Kathleen L. Melde, "Impact of ground via placement in grounded coplanar waveguide interconnects." IEEE Transactions on Components, Packaging and Manufacturing Technology 6.1 (2015): 136-144.
#
# We begin by simulating a conventional CPW with a bottom ground plane to demonstrate the undesirable resonant effect. Then, we will compare it to a design with via fences. Finally, we will test a setup fabricated by Sain et al. in order to validate the simulation results.

# <center><img src="./img/gcpw_via_fence_render.png" width=540 /></center>

# In[1]:
from __future__ import annotations

import numpy as np

import tidy3d as td
import tidy3d.plugins.microwave as mw
import tidy3d.plugins.smatrix as sm
from tidy3d.plugins.dispersion import FastDispersionFitter

td.config.logging_level = "ERROR"


# ## Building the simulation

# ### Key parameters and materials

# The key parameters for this model are the operating frequency range, the geometry, and the material properties. These are defined below.

# <center><img src="./img/gcpw_via_fence_schematic.png" width=800 /></center>

# In[2]:


# Frequency
f_min, f_max = (1e9, 100e9)  # Frequency range (Hz)
f0 = (f_max + f_min) / 2  # Center frequency
freqs = np.linspace(f_min, f_max, 301)  # Sample frequency points
field_mon_freqs = np.linspace(f_min, f_max, 21)  # Frequency points for field monitor data

# CPW Geometry
len_inf = 1e5  # Effective infinity
ws = 177.8  # Signal strip width
wg = 1600  # Ground strip width
t = 17.78  # Conductor thickness
h = 101.6  # Substrate thickness
gg = 101.6  # Gap between signal and side ground
L = 6000  # CPW line length

# Via geometry
vr = 76.2  # Via radius
vp = 287.3  # Via pitch (longitudinal spacing)
vl = 400  # Via transverse spacing (from center of signal strip)
es = 127  # Distance from first/last via to CPW start/end

# Overall simulation size and center
sim_size = (5600, 1000, 7500)
sim_center = (0, 500 - h - t, 0)


# Both the substrate and metal are assumed to have constant loss over the frequency range. The materials used for the simulation are defined below.

# In[3]:


# Material properties
cond = 58  # Metal conductivity in S/um
eps = 3.5  # Relative permittivity, substrate
losstan = 0.002  # Loss tangent, substrate


# In[4]:


med_sub = FastDispersionFitter.constant_loss_tangent_model(
    eps, losstan, (f_min, f_max), tolerance_rms=3e-4
)
med_metal = td.LossyMetalMedium(conductivity=cond, frequency_range=(f_min, f_max), name="Metal")
med_air = td.Medium(permittivity=1.0, name="Air")


# ### Geometry construction

# We start by creating all the planar structures (substrate and metal strips). The ground plane and substrate are assumed to be infinite in width and length.

# In[5]:


# Substrate
str_sub = td.Structure(
    geometry=td.Box(center=(0, -h / 2, 0), size=(len_inf, h, len_inf)), medium=med_sub
)

# Signal strip
str_sig = td.Structure(geometry=td.Box(center=(0, t / 2, 0), size=(ws, t, L)), medium=med_metal)

# Side and bottom grounds
str_gnd1 = td.Structure(
    geometry=td.Box(center=((ws + wg) / 2 + gg, t / 2, 0), size=(wg, t, L)), medium=med_metal
)
str_gnd2 = td.Structure(
    geometry=td.Box(center=(-(ws + wg) / 2 - gg, t / 2, 0), size=(wg, t, L)), medium=med_metal
)
str_gnd3 = td.Structure(
    geometry=td.Box(center=(0, -h - t / 2, 0), size=(len_inf, t, len_inf)), medium=med_metal
)


# We define a reusable function to create the via structure at a given (x, z) position.

# In[6]:


def create_via(xpos, zpos):
    """Create one via structure at given x and z positions"""
    via_structure = td.Structure(
        geometry=td.Cylinder(center=(xpos, -h / 2, zpos), radius=vr, length=h, axis=1),
        medium=med_metal,
    )

    return via_structure


# Then, we loop over all x- and z-positions to create the via fence.

# In[7]:


# create vias
str_via_fence = []
for zpos in np.arange(-L / 2 + es, L / 2 - es + 10, vp):
    for xpos in [-vl, vl]:
        str_via_fence += [create_via(xpos, zpos)]


# We consolidate the structures into two lists, with and without the via fence. We will use these in two separate simulations later.

# In[8]:


# List of all structures (with and without via fence)
str_list_1 = [str_sub, str_gnd1, str_gnd2, str_gnd3, str_sig]
str_list_2 = [str_sub, str_gnd1, str_gnd2, str_gnd3, str_sig] + str_via_fence


# ### Grid

# We use the `LayerRefinementSpec` feature to automatically apply refinement to metal edges and corners.

# In[9]:


def create_layer_refinement_spec(structure_list):
    """Creates a LayerRefinementSpec using the bounding box of the given list of structures"""
    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=structure_list,
        axis=1,
        min_steps_along_axis=1,
        bounds_snapping="bounds",  # snap grid to metal boundaries
        corner_refinement=td.GridRefinement(
            dl=t, num_cells=2
        ),  # snap to corners and apply added refinement
    )

    return lr_spec


# Apply layer refinement to top and bottom metal planes
lr_spec_1 = create_layer_refinement_spec([str_sig, str_gnd1, str_gnd2])
lr_spec_2 = create_layer_refinement_spec([str_gnd3])


# For the rest of the model, we define the grid size using the `min_steps_per_sim_size` parameter since the simulation is largely sub-wavelength.

# In[10]:


# Define overall grid specification
grid_spec = td.GridSpec.auto(
    min_steps_per_sim_size=100,
    wavelength=td.C_0 / f_max,
    layer_refinement_specs=[lr_spec_1, lr_spec_2],
)


# ### Boundaries

# We use the Perfectly Matched Layer (PML) boundary condition on all sides except the bottom. For the bottom boundary, we use the PEC condition for simplicity, since it is adjacent to the infinite bottom ground plane.

# In[11]:


boundary_spec = td.BoundarySpec(
    x=td.Boundary.pml(),
    y=td.Boundary(plus=td.PML(), minus=td.PECBoundary()),
    z=td.Boundary.pml(),
)


# ### Monitors

# We define two field monitors for visualization purposes. One is in the longitudinal substrate plane, while the other is along the transverse plane halfway down the CPW.

# In[12]:


field_mon_1 = td.FieldMonitor(
    center=(0, -h / 2, 0),
    size=(td.inf, 0, td.inf),
    freqs=field_mon_freqs,
    name="field (substrate plane)",
)

field_mon_2 = td.FieldMonitor(
    center=(0, 0, 0), size=(td.inf, td.inf, 0), freqs=field_mon_freqs, name="transverse field"
)


# ### Ports

# In general, we have a choice of using `WavePort` or `LumpedPort` to excite the structure. Because we are probing the resonances of this finite length CPW, it makes more sense to use the `LumpedPort`.
#
# The lumped port is positioned in the gap between the signal strip and side ground planes, with the port voltage axis aligned along the x-direction. The design impedance of the CPW is around 50 ohms, so each lumped port should be 100 ohms.

# In[13]:


LP_len = 50  # Lumped port length

LP1 = sm.LumpedPort(
    center=((ws + gg) / 2, t / 2, -L / 2 + LP_len / 2),
    size=(gg, 0, LP_len),
    name="LP1",
    impedance=100,  # in Ohms
    voltage_axis=0,  # 0 = x-axis
)

LP2 = LP1.updated_copy(name="LP2", center=((ws + gg) / 2, t / 2, L / 2 - LP_len / 2))


# ### Simulation and TerminalComponentModeler

# The `Simulation` object contains all of the previously defined specifications. The first simulation implements the conventional CPW layout without via fences.

# In[14]:


sim_without_via_fence = td.Simulation(
    size=sim_size,
    center=sim_center,
    medium=med_air,  # background medium is air/vacuum
    grid_spec=grid_spec,
    boundary_spec=boundary_spec,
    structures=str_list_1,
    monitors=[field_mon_1, field_mon_2],
    run_time=3.5e-9,  # simulation run time
    symmetry=(1, 0, 0),  # odd symmetry in x-direction
    plot_length_units="mm",  # set plot units to mm
)


# The `TerminalComponentModeler` is a wrapper class that conducts a port sweep on the simulation. The lumped ports are set here.

# In[15]:


tcm_without_via_fence = sm.TerminalComponentModeler(
    simulation=sim_without_via_fence,  # simulation, previously defined
    ports=[LP1, LP2],  # lumped ports, previously defined
    freqs=freqs,  # S-parameter frequency points
)


import os

import tidy3d.web as web
from tidy3d.web.core.environment import Env, dev

Env.set_current(dev)
web.configure(os.environ["TIDY3D_DEV_API_KEY"])
web.run(tcm_without_via_fence, task_name="test_rf", solver_version="dario-rf2-0.0.0")
