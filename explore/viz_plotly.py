import plotly.graph_objects as go
import numpy as np
from typing import List

from .tidy3d import GeometryObject, Box, Sphere, Cylinder, PolySlab
from .tidy3d import Simulation, Monitor, Source, Structure, PMLLayer

""" Handles plotting and visualzation 

Features
 - [x] plotting simple Boxes
 - [x] plotting all geometric objects in simulation
 - [x] customize viz options
 - [ ] add text, interactive info
 - [ ] viz functions for spheres, cylinders
 - [ ] viz functions for polygons / gds files

"""

DEFAULT_KWARGS = {
    # note: kwargs for go.Mesh3D()
    # colors: https://coolors.co/palettes/trending
    'Simulation': {
        'color': '#f4f1de',
        'opacity': 0.1
    },
    'Structure': {
        'color' : '#e07a5f',
        'opacity': 0.5        
    },
    'Source': {
        'color' : '#81b29a',
        'opacity': 0.4       
    },
    'Monitor': {
        'color' : '#f2cc8f',
        'opacity': 0.4      
    },
    'PML': {
        'color' : '#3d405b',
        'opacity': .1
    },
    'Mesh': {
        'color' : '#3d405b',
        'opacity': .05
    }
}


def viz_sim(sim: Simulation, mesh_lines=False):
    meshes = []

    # add simulation
    sim_mesh = viz_geo(sim, **DEFAULT_KWARGS['Simulation'])
    meshes.append(sim_mesh)

    # add structures
    for name, structure in sim.structures.items():
        struct_mesh = viz_geo(structure, **DEFAULT_KWARGS['Structure'])
        meshes.append(struct_mesh)

    # add sources
    for name, source in sim.sources.items():
        sources_mesh = viz_geo(source, **DEFAULT_KWARGS['Source'])
        meshes.append(sources_mesh)

    # add monitors
    for name, monitor in sim.monitors.items():
        monitor_mesh = viz_geo(monitor, **DEFAULT_KWARGS['Monitor'])
        meshes.append(monitor_mesh)

    # add pml
    pml_meshes = get_pml_meshes(sim, **DEFAULT_KWARGS['PML'])
    meshes += pml_meshes

    # add mesh_lines
    if mesh_lines and False:
        # not ready
        mesh_meshes = get_mesh_meshes(sim, **DEFAULT_KWARGS['Mesh'])
        meshes += mesh_meshes

    return meshes

""" ==== Helper functions === """

def _meshgrid_flatten(*args):
    """ meshgrids all args and flattens the results """    
    coords = np.meshgrid(*args, indexing='ij')
    return tuple(c.flatten() for c in coords)

def _rotate_points(dl1, dl2, dl3, axis=2):
    """ rotates order of three objects such that the 3rd is at axis """
    if axis == 0:
        return dl3, dl1, dl2
    elif axis == 1:
        return dl2, dl3, dl1
    elif axis == 2:
        return dl1, dl2, dl3

""" ==== Viualization of different GeometryObjects === """

def viz_geo(obj: GeometryObject, **kwargs):

    assert hasattr(obj, 'geometry'), f"object '{obj}' does not have a geometry"

    if isinstance(obj.geometry, Box):
        return viz_box(obj.geometry, **kwargs)
    elif isinstance(obj.geometry, Sphere):
        return viz_sphere(obj.geometry, **kwargs)
    elif isinstance(obj.geometry, Cylinder):
        return viz_cylinder(obj.geometry, **kwargs)
    elif isinstance(obj.geometry, PolySlab):
        return viz_polyslab(obj.geometry, **kwargs)
    else:
        raise ValueError(f"object '{obj}' not recognized")

def viz_box(box: Box, **kwargs):

    ((xm, ym, zm), (xp, yp, zp)) = box.bounds
    x, y, z = _meshgrid_flatten([xm, xp], [ym, yp], [zm, zp])

    mesh = go.Mesh3d(
        alphahull=0,
        x=x, y=y, z=z,
        **kwargs,
        flatshading=True)
    return mesh

def viz_sphere(sph: Sphere, num_pts=20, **kwargs):

    r = sph.radius
    center = x0, y0, z0 = sph.center
    phis = np.linspace(0, 2*np.pi, num_pts)
    thetas = np.linspace(0, np.pi, num_pts)    

    p, t = _meshgrid_flatten(phis, thetas)


    x = x0 + r * np.cos(p) * np.sin(t)
    y = y0 + r * np.sin(p) * np.sin(t)
    z = z0 + r * np.cos(t)

    mesh = go.Mesh3d(
        alphahull=1,
        x=x, y=y, z=z,
        **kwargs,
        flatshading=True)
    return mesh

def viz_cylinder(cyl: Cylinder, num_pts=20, **kwargs):

    r = cyl.radius
    l = cyl.length
    axis = cyl.axis
    center = cyl.center
    phis = np.linspace(0, 2*np.pi, num_pts)
    lengths = np.linspace(-l/2, l/2, num_pts)
    p, l = _meshgrid_flatten(phis, lengths)

    dl1 = r * np.cos(p)
    dl2 = r * np.sin(p)
    dl = l

    points = _rotate_points(dl1, dl2, dl, axis=axis)
    points += np.array([center]).T
    xyz = {k:v for (k,v) in zip('xyz', points)}

    mesh = go.Mesh3d(
        alphahull=0,
        **xyz,
        **kwargs,
        flatshading=True)

    return mesh

def viz_polyslab(psb: PolySlab, num_pts=15, **kwargs):
    
    num_vertices = len(psb.vertices)

    xy_plane = np.array(psb.vertices)
    xs, ys = xy_plane.T

    zs = np.array(psb.slab_bounds)
    x, z = _meshgrid_flatten(xs, zs)
    y, z = _meshgrid_flatten(ys, zs)

    points = _rotate_points(x, y, z, axis=psb.axis)
    xyz = {k:v for (k, v) in zip('xyz', points)}

    mesh = go.Mesh3d(
        alphahull=0,
        **xyz,
        **kwargs,
        flatshading=True)

    return mesh

""" ==== Helper functions for Simulation plotting === """

def get_pml_meshes(sim: Simulation, **kwargs) -> list:

    pml_layers = sim.pml_layers

    dls = np.array(sim.mesh.grid_step)
    Nls = np.array([pml_layer.num_layers for pml_layer in pml_layers])
    thicknesses = dls * Nls
    bounds_m, bounds_p = np.array(sim.geometry.bounds)

    sim_size = sim.geometry.size
    sim_center = sim.geometry.center

    sizes = [list(sim_size) for _ in range(6)]
    centers = [list(sim_center) for _ in range(6)]

    for axis in range(3):
        index_p = 2*axis
        index_m = 2*axis + 1
        sizes[index_p][axis] = thicknesses[axis]
        sizes[index_m][axis] = thicknesses[axis]
        centers[index_p][axis] += (sim_size[axis] + thicknesses[axis]) / 2
        centers[index_m][axis] -= (sim_size[axis] + thicknesses[axis]) / 2

    boxes = []
    for (s, c) in zip(sizes, centers):
        if 0.0 not in s:
            boxes.append(Box(size=s, center=c))

    meshes = [viz_box(box, **kwargs) for box in boxes]
    return meshes

def get_mesh_meshes(sim: Simulation, **kwargs) -> list:

    dls = np.array(sim.mesh.grid_step)
    sim_size = np.array(sim.geometry.size)
    sim_center = np.array(sim.geometry.center)
    sim_bounds = np.array(sim.geometry.bounds)

    meshes = []
    for axis in range(3):
        bmin, bmax = sim_bounds
        centers_axis = np.arange(bmin[axis], bmax[axis], dls[axis])
        for center_axis in centers_axis:
            center = sim_center.copy()
            center[axis] = center_axis
            size = sim_size.copy()
            size[axis] = 0.1
            print(center)
            box = Box(size=list(size), center=list(center))
            meshes.append(viz_box(box, **kwargs))
    return meshes


    return mesh
