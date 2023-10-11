"""Tests file export and loading."""
import os
import json

import pytest
import numpy as np
from time import time
import dill as pickle


from tidy3d import __version__
import tidy3d as td
from tidy3d.components.base import DATA_ARRAY_MAP
from ..utils import SIM_FULL as SIM
from ..utils import SIM_MONITORS as SIM2
from ..test_data.test_monitor_data import make_flux_data
from ..test_data.test_sim_data import make_sim_data
from tidy3d.components.data.sim_data import DATA_TYPE_MAP

# Store an example of every minor release simulation to test updater in the future
SIM_DIR = "tests/sims"


def set_datasets_to_none(sim):
    sim_dict = sim.dict()
    for src in sim_dict["sources"]:
        if src["type"] == "CustomFieldSource":
            src["field_dataset"] = None
        elif src["type"] == "CustomCurrentSource":
            src["current_dataset"] = None
        if src["source_time"]["type"] == "CustomSourceTime":
            src["source_time"]["source_time_dataset"] = None
    for structure in sim_dict["structures"]:
        if structure["geometry"]["type"] == "TriangleMesh":
            structure["geometry"]["mesh_dataset"] = None
        if structure["geometry"]["type"] == "GeometryGroup":
            for geometry in structure["geometry"]["geometries"]:
                if geometry["type"] == "TriangleMesh":
                    geometry["mesh_dataset"] = None
        if "Custom" in structure["medium"]["type"]:
            if structure["medium"]["type"] == "CustomMedium":
                structure["medium"]["eps_dataset"] = None
            elif structure["medium"]["type"] == "CustomPoleResidue":
                structure["medium"]["poles"] = []
            else:
                structure["medium"]["coeffs"] = []
    return td.Simulation.parse_obj(sim_dict)


def test_simulation_load_export():
    major, minor, patch = __version__.split(".")
    path = os.path.join(SIM_DIR, f"simulation_{major}_{minor}_{patch}.json")
    # saving as .h5 since *.hdf5 is git ignored
    path_hdf5 = os.path.join(SIM_DIR, f"simulation_{major}_{minor}_{patch}.h5")
    SIM.to_file(path)
    SIM.to_hdf5(path_hdf5)
    SIM2 = td.Simulation.from_file(path)
    SIM_HDF5 = td.Simulation.from_hdf5(path_hdf5)
    assert set_datasets_to_none(SIM) == SIM2, "original and loaded simulations are not the same"
    assert SIM == SIM_HDF5, "original and loaded from hdf5 simulations are not the same"


def test_simulation_load_export_yaml(tmp_path):
    path = str(tmp_path / "simulation.yaml")
    SIM.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert set_datasets_to_none(SIM) == SIM2, "original and loaded simulations are not the same"


def test_component_load_export(tmp_path):
    path = str(tmp_path / "medium.json")
    td.Medium().to_file(path)
    M2 = td.Medium.from_file(path)
    assert td.Medium() == M2, "original and loaded medium are not the same"


def test_component_load_export_yaml(tmp_path):
    path = str(tmp_path / "medium.yaml")
    td.Medium().to_file(path)
    M2 = td.Medium.from_file(path)
    assert td.Medium() == M2, "original and loaded medium are not the same"


def test_simulation_load_export_hdf5(tmp_path):
    path = str(tmp_path / "simulation.hdf5")
    SIM.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_gz(tmp_path):
    path = str(tmp_path / "simulation.hdf5.gz")
    SIM.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_explicit(tmp_path):
    path = str(tmp_path / "simulation.hdf5")
    SIM.to_hdf5(path)
    SIM2 = td.Simulation.from_hdf5(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_gz_explicit(tmp_path):
    path = str(tmp_path / "simulation.hdf5.gz")
    SIM.to_hdf5_gz(path)
    SIM2 = td.Simulation.from_hdf5_gz(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_pckl(tmp_path):
    path = str(tmp_path / "simulation.pckl")
    with open(path, "wb") as pickle_file:
        pickle.dump(SIM, pickle_file)
    with open(path, "rb") as pickle_file:
        SIM2 = pickle.load(pickle_file)
    assert SIM == SIM2, "original and loaded simulations are not the same"


def test_simulation_preserve_types(tmp_path):
    """Test that all re-loaded components have the same types."""
    path = str(tmp_path / "simulation.json")
    SIM.to_file(path)
    sim_2 = td.Simulation.from_file(path)
    assert set_datasets_to_none(SIM) == sim_2

    M_types = [type(s.medium) for s in sim_2.structures]
    for M in (td.Medium, td.PoleResidue, td.Lorentz, td.Sellmeier, td.Debye):
        assert M in M_types

    G_types = [type(s.geometry) for s in sim_2.structures]
    for G in (td.Box, td.Sphere, td.Cylinder, td.PolySlab):
        assert G in G_types

    S_types = [type(s) for s in sim_2.sources]
    for S in (td.UniformCurrentSource, td.PlaneWave, td.GaussianBeam):
        assert S in S_types

    M_types = [type(m) for m in sim_2.monitors]
    for M in (
        td.FieldMonitor,
        td.FieldTimeMonitor,
        td.ModeMonitor,
        td.FluxMonitor,
        td.FluxTimeMonitor,
    ):
        assert M in M_types


def test_1a_simulation_load_export2(tmp_path):
    path = str(tmp_path / "simulation.json")
    SIM2.to_file(path)
    SIM3 = td.Simulation.from_file(path)
    assert SIM2 == SIM3, "original and loaded simulations are not the same"


def test_validation_speed(tmp_path):
    sizes_bytes = []
    times_sec = []
    path = str(tmp_path / "simulation.json")

    _ = SIM
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
        _S = td.Simulation.from_file(path)
        time_validate = time() - time_start
        times_sec.append(time_validate)
        assert set_datasets_to_none(S) == _S

        size = os.path.getsize(path)
        sizes_bytes.append(size)

        print(f"{n} structures \t {size:.1e} bytes \t {time_validate:.1f} seconds to validate")


SIM_FILES = [os.path.join(SIM_DIR, file) for file in os.listdir(SIM_DIR)]


@pytest.mark.parametrize("sim_file", SIM_FILES)
def test_simulation_updater(sim_file):
    """Test that all simulations in ``SIM_DIR`` can be updated to current version and loaded."""
    sim_updated = td.Simulation.from_file(sim_file)
    assert sim_updated.version == __version__, "Simulation not converted properly"

    # just make sure the loaded sim does something properly using this version
    sim_updated.grid


def test_yaml(tmp_path):
    path = str(tmp_path / "simulation.json")
    SIM.to_file(path)
    sim = td.Simulation.from_file(path)
    path1 = str(tmp_path / "simulation.yaml")
    sim.to_yaml(path1)
    sim1 = td.Simulation.from_yaml(path1)
    assert sim1 == sim


def test_to_json_data():
    """Tests that the json string with data in separate file behaves correctly."""

    # type saved in the combined json file?
    data = make_flux_data()
    json_dict = json.loads(data._json_string)
    assert json_dict["flux"] in DATA_ARRAY_MAP


def test_to_hdf5_group_path_sim_data(tmp_path):
    """Tests that the json string with data in separate file behaves correctly in SimulationData."""

    # type saved in the combined json file?
    sim_data = make_sim_data()
    FNAME = str(tmp_path / "sim_data.hdf5")
    sim_data.to_file(fname=FNAME)

    for i, monitor in enumerate(sim_data.simulation.monitors):
        group_name = sim_data.get_tuple_group_name(index=i)
        group_path = f"/data/{group_name}"
        MntDataType = DATA_TYPE_MAP[type(monitor)]
        mnt_data = MntDataType.from_file(fname=FNAME, group_path=group_path)
        assert mnt_data == sim_data.monitor_data[monitor.name]


def test_none_hdf5(tmp_path):
    """Tests that values of None where None is not the default are loaded correctly."""

    sim = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        normalize_index=None,
    )

    assert sim.normalize_index is None, "'normalize_index' of 'None' not initialized correctly."

    fname = str(tmp_path / "sim_none.hdf5")
    sim.to_file(fname)
    sim2 = td.Simulation.from_file(fname)

    assert sim2.normalize_index is None, "'normalize_index' of 'None' not loaded correctly."


def test_group_name_tuple():
    """Test conversion of group names and tuples."""

    tidy = td.Medium()
    tuple_values = ["Something", "Another thing", "Something different entirely"]
    test_dict = tidy.tuple_to_dict(tuple_values=tuple_values)

    # make sure we can pick out the index from the dictionary keys and get the correct key
    for true_index, key_name in enumerate(test_dict.keys()):
        index = tidy.get_tuple_index(key_name=key_name)
        assert index == true_index
        group_name = tidy.get_tuple_group_name(index=index)
        assert group_name == key_name
