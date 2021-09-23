import sys
sys.path.append('.')

import tidy3d.web as web
from tidy3d import *
from tidy3d.web.task import TaskStatus

sim = Simulation(
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
            Structure(geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1), medium=Medium()),
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
            "point": FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), sampler=FreqSampler(freqs=[1, 2])),
            "plane": FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), sampler=TimeSampler(times=[1, 2])),
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

""" clear the tmp directory and make the necessary directories """
import os

TMP_DIR = 'tests/tmp'

def _clear_dir(path=TMP_DIR):
    """clears a dir"""
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if not os.path.isdir(full_path):
            os.remove(full_path)

PATH_CLIENT = os.path.join(TMP_DIR, 'client')
PATH_SERVER = os.path.join(TMP_DIR, 'server')

_clear_dir()

if os.path.exists(PATH_CLIENT):
    _clear_dir(PATH_CLIENT)
else:
    os.mkdir(PATH_CLIENT)

if os.path.exists(PATH_SERVER):
    _clear_dir(PATH_SERVER)
else:
    os.mkdir(PATH_SERVER)

def test_1_upload():
    task_id = web.upload(sim)

def test_2_info():
    task_id = web.upload(sim)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.INIT

def test_3_run():
    task_id = web.upload(sim)
    web.run(task_id)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.SUCCESS

def test_4_monitor():
    task_id = web.upload(sim)
    web.run(task_id)
    web.monitor(task_id)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.SUCCESS

def test_5_download():
    task_id = web.upload(sim)
    web.run(task_id)    
    task_info = web.get_info(task_id)
    path_data = os.path.join(PATH_CLIENT, f'sim_{task_id}.hdf5')
    web.download(task_id, path=path_data)    












