import pydantic

import sys
sys.path.append('./')

from tidy3d import *

s1 = Simulation(
    size=(2.0, 2.0, 2.0),
    grid_size=(0.01, 0.01, 0.01),
    run_time=1e-12,
    structures=[
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Medium(permittivity=2.0),
        ),
        Structure(
            geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=1.0, conductivity=3.0),
        ),
        Structure(
            geometry=Sphere(
                radius=1.4,
                center=(1.0, 0.0, 1.0)
            ),
            medium=Medium()
        ),
        Structure(
            geometry=Cylinder(
                radius=1.4,
                length=2.0,
                center=(1.0, 0.0, -1.0),
                axis=1
            ),
            medium=Medium()
        )        
    ],
    sources={
        "my_dipole": VolumeSource(
            size=(0, 0, 0),
            center=(0, -0.5, 0),
            polarization='Mx',
            source_time=GaussianPulse(
                freq0=1e14,
                fwidth=1e12,
            ),
        )
    },
    monitors={
        "point": FieldMonitor(size=(0,0,0), center=(0,0,0), sampler=FreqSampler(freqs=[1,2])),
        "plane": FluxMonitor(size=(1,1,0), center=(0,0,0), sampler=TimeSampler(times=[1,2]))
    },
    symmetry=(0, -1, 1),
    pml_layers=(
        PMLLayer(profile="absorber", num_layers=20),
        PMLLayer(profile="stable", num_layers=30),
        PMLLayer(profile="standard"),
    ),
    shutoff=1e-6,
    courant=0.8,
    subpixel=False,
)

def test_load_export():
    path = 'tests/tmp/simulation.json'
    s1.export(path)
    s2 = Simulation.load(path)
    assert s1 == s2, "original and loaded simulations are not the same"
