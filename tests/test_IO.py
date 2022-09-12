import os
import json

import pydantic
import numpy as np
import os
from time import time
import xarray as xr
import h5py
from dask.base import tokenize


from tidy3d import *
from tidy3d import __version__
import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from .utils import SIM_FULL as SIM
from .utils import SIM_MONITORS as SIM2
from .utils import clear_tmp
from .test_data_monitor import make_flux_data

# Store an example of every minor release simulation to test updater in the future
SIM_DIR = "tests/sims"


@clear_tmp
def test_simulation_load_export():

    path = "tests/tmp/simulation.json"
    SIM.to_file(path)
    SIM2 = Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


@clear_tmp
def test_simulation_load_export_yaml():

    path = "tests/tmp/simulation.yaml"
    SIM.to_file(path)
    SIM2 = Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


@clear_tmp
def test_simulation_load_export_hdf5():

    path = "tests/tmp/simulation.hdf5"
    SIM.to_file(path)
    SIM2 = Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


@clear_tmp
def test_simulation_preserve_types():
    """This test also writes a simulation file to ``tests/sims/simulation_x_y_z.json`` to store
    an example of the current version. Updating all of these files is then checked in
    ``test_simulation_updater``."""

    st = GaussianPulse(freq0=1.0, fwidth=1.0)

    sim_all = Simulation(
        size=(10.0, 10.0, 10.0),
        structures=[
            Structure(geometry=Box(size=(1, 1, 1)), medium=Medium()),
            Structure(geometry=Sphere(radius=1), medium=PoleResidue(eps_inf=1, poles=[])),
            Structure(
                geometry=Cylinder(radius=1, length=1, axis=2),
                medium=Lorentz(eps_inf=1.0, coeffs=[]),
            ),
            Structure(
                geometry=PolySlab(vertices=[(0, 0), (2, 3), (4, 3)], slab_bounds=(-1, 1), axis=2),
                medium=Sellmeier(coeffs=[]),
            ),
            Structure(geometry=Sphere(radius=1), medium=Debye(eps_inf=1.0, coeffs=[]), name="t2"),
            Structure(
                geometry=GeometryGroup(
                    geometries=[
                        PolySlab(vertices=[(0, 0), (2, 3), (4, 3)], slab_bounds=(-1, 1), axis=2),
                        Cylinder(radius=1, length=1, axis=2),
                    ]
                ),
                medium=Drude(coeffs=[[1.0, 1.0]]),
            ),
        ],
        sources=[
            UniformCurrentSource(size=(0, 0, 0), source_time=st, polarization="Ex"),
            PlaneWave(
                center=(0, 0, -4),
                size=(inf, inf, 0),
                source_time=st,
                direction="+",
                pol_angle=2.0,
            ),
            GaussianBeam(
                center=(0, 0, -4),
                size=(1, 1, 0),
                source_time=st,
                direction="+",
                waist_radius=1,
            ),
            ModeSource(
                center=(0, 0, 0),
                size=(0, 1, 1),
                mode_spec=ModeSpec(num_modes=3),
                mode_index=2,
                direction="-",
                source_time=st,
            ),
        ],
        monitors=[
            FieldMonitor(size=(1, 1, 1), freqs=[1, 2, 3], name="field"),
            FieldTimeMonitor(size=(1, 1, 1), start=1e-12, interval=3, name="fieldtime"),
            FluxMonitor(size=(1, 0, 1), freqs=[1, 2, 3], name="flux"),
            FluxTimeMonitor(size=(1, 0, 1), start=1e-12, interval=3, name="fluxtime"),
            ModeMonitor(size=(1, 0, 1), freqs=[1, 2], mode_spec=ModeSpec(num_modes=3), name="mode"),
        ],
        run_time=1e-12,
        symmetry=(0, -1, 1),
        boundary_spec=BoundarySpec(
            x=Boundary(minus=PML(), plus=Absorber()),
            y=Boundary(minus=PMCBoundary(), plus=Periodic()),
            z=Boundary(minus=StablePML(), plus=PECBoundary()),
        ),
    )

    major, minor, patch = __version__.split(".")
    path = os.path.join(SIM_DIR, f"simulation_{major}_{minor}_{patch}.json")
    sim_all.to_file(path)
    sim_2 = Simulation.from_file(path)
    assert sim_all == sim_2

    M_types = [type(s.medium) for s in sim_2.structures]
    for M in (Medium, PoleResidue, Lorentz, Sellmeier, Debye):
        assert M in M_types

    G_types = [type(s.geometry) for s in sim_2.structures]
    for G in (Box, Sphere, Cylinder, PolySlab):
        assert G in G_types

    S_types = [type(s) for s in sim_2.sources]
    for S in (UniformCurrentSource, PlaneWave, GaussianBeam):
        assert S in S_types

    M_types = [type(m) for m in sim_2.monitors]
    for M in (FieldMonitor, FieldTimeMonitor, ModeMonitor, FluxMonitor, FluxTimeMonitor):
        assert M in M_types


def test_1a_simulation_load_export2():
    path = "tests/tmp/simulation.json"
    SIM2.to_file(path)
    SIM3 = Simulation.from_file(path)
    assert SIM2 == SIM3, "original and loaded simulations are not the same"


def test_validation_speed():

    sizes_bytes = []
    times_sec = []
    path = "tests/tmp/simulation.json"

    sim_base = SIM
    N_tests = 10

    # adjust as needed, keeping small to speed tests up
    num_structures = np.logspace(0, 2, N_tests).astype(int)

    for n in num_structures:
        new_structures = []
        for i in range(n):
            new_structure = SIM.structures[0].copy(update={"name": str(i)})
            new_structures.append(new_structure)
        S = SIM.copy(update=dict(structures=new_structures))

        S.to_file(path)
        time_start = time()
        _S = Simulation.from_file(path)
        time_validate = time() - time_start
        times_sec.append(time_validate)
        assert S == _S

        size = os.path.getsize(path)
        sizes_bytes.append(size)

        print(f"{n} structures \t {size:.1e} bytes \t {time_validate:.1f} seconds to validate")


def test_simulation_updater():
    """Test that all simulations in ``SIM_DIR`` can be updated to current version and loaded."""
    sim_files = [os.path.join(SIM_DIR, file) for file in os.listdir(SIM_DIR)]
    for sim_file in sim_files:
        sim_updated = Simulation.from_file(sim_file)
        assert sim_updated.version == __version__, "Simulation not converted properly"

        # just make sure the loaded sim does something properly using this version
        sim_updated.grid


@clear_tmp
def test_yaml():
    path = "tests/tmp/simulation.json"
    SIM.to_file(path)
    sim = Simulation.from_file(path)
    path1 = "tests/tmp/simulation.yaml"
    sim.to_yaml(path1)
    sim1 = Simulation.from_yaml(path1)
    assert sim1 == sim


@clear_tmp
def test_to_json_data():
    """Tests that the json string with data in separate file behaves correctly."""

    # type saved in the combined json file?
    data = make_flux_data()
    json_dict = json.loads(data._json_string())
    assert json_dict["flux"] is not None
    assert json_dict["flux"]["type"] == "FluxDataArray"

    # type saved inside of the separated data file?
    data_file = "tests/tmp/data_file.hdf5"
    json_dict = json.loads(data._json_string(data_file=data_file))
    assert json_dict["flux"] is not None
    assert json_dict["flux"]["data_file"] == data_file
    with h5py.File(data_file, "r") as f:
        type_dataset = f[tokenize(data.flux)]["type"]
        type_str = Tidy3dBaseModel.unpack_dataset(type_dataset, keep_numpy=False)
        assert type_str == "FluxDataArray"

    # type saved to hdf5 file?
    data_file_direct = "tests/tmp/flux_data.hdf5"
    data.to_file(data_file_direct)
    with h5py.File(data_file_direct, "r") as f:
        type_dataset = f["flux"]["type"]
        type_str = Tidy3dBaseModel.unpack_dataset(type_dataset, keep_numpy=False)
        assert type_str == "FluxDataArray"


@clear_tmp
def test_none_hdf5():
    """Tests that values of None where None is not the default are loaded correctly."""

    sim = Simulation(
        size=(1, 1, 1),
        grid_spec=GridSpec(wavelength=1.0),
        run_time=1e-12,
        normalize_index=None,
    )

    assert sim.normalize_index is None, "'normalize_index' of 'None' not initialized correctly."

    fname = "tests/tmp/sim_none.hdf5"
    sim.to_file(fname)
    sim2 = Simulation.from_file(fname)

    assert sim2.normalize_index is None, "'normalize_index' of 'None' not loaded correctly."
