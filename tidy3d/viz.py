import plotly.graph_objects as go
import numpy as np

from .tidy3d import GeometryObject, Box
from .tidy3d import Simulation, Monitor, Source, Structure

DEFAULT_KWARGS = {
    'Simulation': {
        'color': '#f4f1de',
        'opacity': 0.2
    },
    'Structure': {
        'color' : '#e07a5f',
        'opacity': 0.4        
    },
    'Source': {
        'color' : '#81b29a',
        'opacity': 0.3       
    },
    'Monitor': {
        'color' : '#f2cc8f',
        'opacity': 0.3      
    }
}


def viz_sim(sim: Simulation):
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

    return meshes

def viz_geo(obj: GeometryObject, **kwargs):

    assert hasattr(obj, 'geometry'), f"object '{obj}' does not have a geometry"

    if isinstance(obj.geometry, Box):

        return viz_box(obj.geometry, **kwargs)
    else:
        return None

def viz_box(box: Box, **kwargs):

    ((xm, ym, zm), (xp, yp, zp)) = box.bounds
    coords = np.meshgrid([xm, xp], [ym, yp], [zm, zp])
    x, y, z = tuple(c.flatten() for c in coords)

    mesh = go.Mesh3d(
        alphahull=0,
        x=x, y=y, z=z,
        **kwargs,
        flatshading=True)

    return mesh
