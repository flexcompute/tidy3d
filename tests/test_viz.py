import numpy as np
import plotly.graph_objects as go

import sys
sys.path.append('../')

from tidy3d.tidy3d import *
import tidy3d.web as web
import tidy3d.viz as viz

def _test_box_plot():

    b1 = Box(size=(1,1,1), center=(0,0,0))
    b2 = Box(size=(2,1,1), center=(0,-2,0))
    b3 = Box(size=(1,1,0.01), center=(0,0,1))

    mesh_data = [viz.viz_box(b) for b in [b1, b2, b3]]
    fig = go.Figure(data=mesh_data)
    fig.show()

def _test_sphere_plot():

    s1 = Sphere(radius=0.1, center=(0,0,0))
    s2 = Sphere(radius=0.1, center=(0,-2,0))
    s3 = Sphere(radius=.1, center=(0,0,1))

    mesh_data = [viz.viz_sphere(s, num_pts=15) for s in [s1, s2, s3]]
    fig = go.Figure(data=mesh_data)
    fig.show()


def _test_cyl_plot():

    c1 = Cylinder(radius=0.5, center=(0,0,0), length=1, axis=0)
    c2 = Cylinder(radius=0.5, center=(0,-2,0), length=1, axis=1)
    c3 = Cylinder(radius=0.5, center=(0,0,1), length=1, axis=2)

    mesh_data = [viz.viz_cylinder(c, num_pts=15) for c in [c1, c2, c3]]
    fig = go.Figure(data=mesh_data)
    fig.show()

def test_polyslab_plot():

    def heart_coords(t):
        x = 16*np.sin(t)**3
        y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
        return [x,y]

    tpoints =  np.linspace(0, 2*np.pi, 300)
    vertices = [heart_coords(t) for t in tpoints]

    p1 = PolySlab(vertices=vertices, slab_bounds=(-.5,.5), axis=2)
    mesh_data = [viz.viz_polyslab(p, num_pts=10) for p in [p1]]
    fig = go.Figure(data=mesh_data)
    fig.show()


def _test_sim_plot():

    sim = Simulation(
        geometry=Box(
            size=(4.0, 4.0, 4.0),
            center=(0, 0, 0)
        ),
        mesh=Mesh(
            grid_step=(0.01, 0.01, 0.01),
        ),
        run_time=1e-12,
        structures={
            "box1": Structure(
                geometry=Box(
                    size=(1, 1, 1),
                    center=(-1.5, 0, 0)
                ),
                medium=Medium(permittivity=2.0),
            ),
            "box2": Structure(
                geometry=Box(
                    size=(1, 1, 1),
                    center=(1.5, 0, 0)),
                medium=Medium(permittivity=2.0),
            ),
            "box3": Structure(
                geometry=Sphere(
                    radius=0.5,
                    center=(0, 1.5, 0)),
                medium=Medium(permittivity=2.0),
            ),
            "box4": Structure(
                geometry=Cylinder(
                    radius=0.5,
                    length=1,
                    center=(0,-1.5, 0)),
                medium=Medium(permittivity=2.0),
            )
        },
        sources={
            "plane_source": Source(
                geometry=Box(
                    size=(4, 4, 0.),
                    center=(0, 0, -1.5)),
                source_time=Pulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                polarization=(1, 0, 1),
            )
        },
        monitors={
            "plane": Monitor(
                geometry=Box(size=(4, 4, 0.), center=(0, 0, 1.5)),
            ),
            "point": Monitor(
                geometry=Box(size=(0.1, 0.1, 0.1), center=(0, 0, 0)),
            ),
        },
        symmetry=(0, -1, 1),
        pml_layers=(
            PMLLayer(profile="absorber", num_layers=0),
            PMLLayer(profile="stable", num_layers=0),
            PMLLayer(profile="standard"),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )

    meshes = viz.viz_sim(sim, mesh_lines=True)
    fig = go.Figure(data=meshes)
    fig.show()
