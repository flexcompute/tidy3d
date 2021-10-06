import os

import sys

sys.path.append("./")
from tidy3d import *

""" utilities shared between all tests """


def clear_dir(path: str):
    """clears a dir"""
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if not os.path.isdir(full_path):
            os.remove(full_path)


TMP_DIR = "tests/tmp/"

# decorator that clears the tmp/ diretory before test
def clear_tmp(fn):
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    def new_fn(*args, **kwargs):
        clear_dir(TMP_DIR)
        return fn(*args, **kwargs)

    return new_fn


SIM_MONITORS = Simulation(
    size=(2.0, 2.0, 2.0),
    grid_size=(0.1, 0.1, 0.1),
    monitors={
        "field_freq": FieldMonitor(size=(1, 1, 1), center=(0, 1, 0), freqs=[1, 2, 5, 7, 8]),
        "field_time": FieldTimeMonitor(size=(1, 1, 0), center=(1, 0, 0), times=[1]),
        "eps_freq": PermittivityMonitor(size=(1, 1, 1), center=(0, 1, 0), freqs=[1, 2, 5, 7, 8]),
        "flux_freq": FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[1, 2, 5, 9]),
        "flux_time": FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), times=[1, 2, 3]),
        "mode": ModeMonitor(
            size=(1, 1, 0), center=(0, 0, 0), freqs=[1.90, 2.01, 2.2], modes=[Mode(mode_index=1)]
        ),
    },
)

SIM_FULL = Simulation(
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
        Structure(geometry=Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=Medium()),
        Structure(
            geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
            medium=Medium(),
        ),
    ],
    sources={
        "my_dipole": VolumeSource(
            size=(0, 0, 0),
            center=(0, -0.5, 0),
            polarization="Mx",
            source_time=GaussianPulse(
                freq0=1e14,
                fwidth=1e12,
            ),
        )
    },
    monitors={
        "point": FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1, 2]),
        "plane": FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), times=[1, 2]),
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
