import pytest
import numpy as np
import os
import sys
from memory_profiler import profile

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source import PointDipole, GaussianPulse
from tidy3d.components.grid.grid_spec import GridSpec

import tidy3d as td

sys.path.append("/users/twhughes/Documents/Flexcompute/tidy3d-core")
from tidy3d_backend.utils import Profile

PATH = "tests/tmp/memory.hdf5"

""" Testing the memory usage of writing SimulationData to and from .hdf5 file.
    
    pip install memory_profiler
    python -m memory_profiler tests/test_data_memory.py

    note, units are in MiB, so need to convert to MB / GB.

    https://www.thecalculatorsite.com/conversions/datastorage.php

"""

# will set size of sim_data1 to give a file size of this many GB
FILE_SIZE_GB = 4.0


def make_sim_data_1(file_size_gb=FILE_SIZE_GB):
    # approximate # of points in the scalar field data

    N = int(2.528e8 / 4 * file_size_gb)

    n = int(N ** (0.25))

    data = (1 + 1j) * np.random.random((n, n, n, n))
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, n)
    f = np.linspace(2e14, 4e14, n)
    src = PointDipole(
        center=(0, 0, 0), source_time=GaussianPulse(freq0=3e14, fwidth=1e14), polarization="Ex"
    )
    coords = dict(x=x, y=y, z=z, f=f)
    Ex = ScalarFieldDataArray(data, coords=coords)
    monitor = FieldMonitor(size=(2, 2, 2), freqs=f, name="test", fields=["Ex"])
    field_data = FieldData(monitor=monitor, Ex=Ex)
    sim = Simulation(
        size=(2, 2, 2),
        grid_spec=GridSpec(wavelength=1),
        monitors=(monitor,),
        sources=(src,),
        run_time=1e-12,
    )
    return SimulationData(
        simulation=sim,
        data=(field_data,),
    )


SIM_DATA_1 = make_sim_data_1()


@profile
def test_memory_1_save():
    print(f'sim_data_size = {SIM_DATA_1.monitor_data["test"].Ex.nbytes:.2e} Bytes')
    SIM_DATA_1.to_file(PATH)
    print(f"file_size = {os.path.getsize(PATH):.2e} Bytes")


@profile
def test_memory_2_load():
    print(f"file_size = {os.path.getsize(PATH):.2e} Bytes")
    return SimulationData.from_file(PATH)


def test_core_profile_small_1_save():

    Nx, Ny, Nz, Nt = 100, 100, 100, 10

    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    t = np.arange(Nt)
    coords = dict(x=x, y=y, z=z, t=t)
    scalar_field = td.ScalarFieldTimeDataArray(np.random.random((Nx, Ny, Nz, Nt)), coords=coords)
    monitor = td.FieldTimeMonitor(size=(2, 4, 6), interval=100, name="field", fields=["Ex", "Hz"])
    data = td.FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field)
    with Profile():
        data.to_file(PATH)
        print(f"file_size = {os.path.getsize(PATH):.2e} Bytes")


def test_core_profile_small_2_load():

    with Profile():
        print(f"file_size = {os.path.getsize(PATH):.2e} Bytes")
        data = td.FieldTimeData.from_file(PATH)


def test_core_profile_large():

    sim_data = make_sim_data_1()

    with Profile():
        sim_data.to_file(PATH)

    print(f"file_size = {os.path.getsize(PATH):.2e} Bytes")

    with Profile():
        sim_data.from_file(PATH)


@profile
def test_speed_many_datasets():

    Nx, Ny, Nz, Nf = 100, 100, 100, 1

    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    f = np.arange(Nf)
    coords = dict(x=x, y=y, z=z, f=f)
    scalar_field = td.ScalarFieldDataArray(np.random.random((Nx, Ny, Nz, Nf)), coords=coords)

    def make_field_data(num_index: int):
        monitor = td.FieldMonitor(
            size=(2, 4, 6),
            # fields=('Ex', 'Ey', 'Ez'),
            freqs=np.linspace(1e14, 2e14, Nf).tolist(),
            name=str(num_index),
        )
        scalar_fields = {fld: scalar_field for fld in monitor.fields}

        return td.FieldData(monitor=monitor, **scalar_fields)

    num_datasets = 100

    data = [make_field_data(n) for n in range(num_datasets)]
    monitors = [d.monitor for d in data]
    src = PointDipole(
        center=(0, 0, 0), source_time=GaussianPulse(freq0=3e14, fwidth=1e14), polarization="Ex"
    )
    sim = Simulation(
        size=(2, 2, 2),
        grid_spec=GridSpec(wavelength=1),
        sources=(src,),
        run_time=1e-12,
        monitors=monitors,
    )

    sim_data = td.SimulationData(
        simulation=sim,
        data=data,
    )

    with Profile():

        sim_data.to_file(PATH)
        sim_data2 = sim_data.from_file(PATH)


if __name__ == "__main__":
    test_memory_1_save()
    sim_data1 = test_memory_2_load()
