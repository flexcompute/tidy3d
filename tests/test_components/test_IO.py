"""Tests file export and loading."""

from __future__ import annotations

import json
import math
import os
from time import time

import dill as pickle
import h5py
import numpy as np
import pytest
import yaml

import tidy3d as td
from tidy3d import __version__
from tidy3d.components.base import DATA_ARRAY_MAP, Tidy3dBaseModel
from tidy3d.components.data.data_array import DataArray
from tidy3d.components.data.sim_data import DATA_TYPE_MAP

from ..test_data.test_monitor_data import make_flux_data
from ..test_data.test_sim_data import make_sim_data
from ..utils import SIM_FULL as SIM
from ..utils import SIM_FULL_FIELD_PROJECTION as SIM_PROJECTION
from ..utils import SIM_MONITORS as SIM2
from ..utils import run_emulated

# Store an example of every minor release simulation to test updater in the future
SIM_DIR = "tests/sims"
SIM_STATIC = SIM.to_static()


@pytest.fixture
def split_string(monkeypatch):
    """Lower the max string length in hdf5 read/write, in order to test the string splitting."""
    from tidy3d.components import base

    monkeypatch.setattr(base, "MAX_STRING_LENGTH", 100)


def set_datasets_to_none(sim):
    sim_dict = sim.model_dump()
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
            # all UnstructuredGridDataset will be converted into SpatialDataArray (vacuum)
            for field_name in ["permittivity", "conductivity", "eps_inf"]:
                if field_name in structure["medium"]:
                    if isinstance(structure["medium"][field_name], dict):
                        structure["medium"][field_name] = "SpatialDataArray"
            if structure["medium"]["type"] == "CustomMedium":
                structure["medium"]["eps_dataset"] = None
            elif structure["medium"]["type"] == "CustomPoleResidue":
                structure["medium"]["poles"] = []
            else:
                structure["medium"]["coeffs"] = []
    return td.Simulation.model_validate(sim_dict)


def _average_runtime(func, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        func()

    elapsed = []
    for _ in range(repeats):
        t0 = time()
        func()
        elapsed.append(time() - t0)
    return sum(elapsed) / len(elapsed)


def _make_many_small_array_sim_data(
    num_flux_monitors: int = 64, num_freqs: int = 4
) -> tuple[td.SimulationData, str, int]:
    """Build a compact SimulationData fixture with many small DataArrays.

    This stresses repeated HDF5 open/close overhead without making the benchmark
    dominated by large array reads.
    """

    freqs = [2e14 + ind * 1e12 for ind in range(num_freqs)]
    flux_monitors = [
        td.FluxMonitor(center=(0, 0, 0), size=(1, 1, 0), freqs=freqs, name=f"flux_{ind}")
        for ind in range(num_flux_monitors)
    ]
    field_monitor = td.FieldMonitor(
        center=(0, 0, 0), size=(1, 1, 0), freqs=freqs, name="field_many"
    )

    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        monitors=[field_monitor, *flux_monitors],
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            )
        ],
        run_time=2e-12,
    )

    coords = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0], "f": freqs}
    field_shape = (2, 2, 1, len(freqs))
    field_array = td.ScalarFieldDataArray((1 + 1j) * np.ones(field_shape), coords=coords)
    field_data = td.FieldData(
        monitor=field_monitor,
        Ex=field_array,
        Ey=field_array,
        Ez=field_array,
        Hx=field_array,
        Hy=field_array,
        Hz=field_array,
        grid_expanded=simulation.discretize_monitor(field_monitor),
    )

    flux_data = [
        td.FluxData(
            monitor=monitor,
            flux=td.FluxDataArray(
                np.linspace(ind, ind + 1, len(freqs)),
                coords={"f": freqs},
            ),
        )
        for ind, monitor in enumerate(flux_monitors)
    ]

    sim_data = td.SimulationData(
        simulation=simulation,
        data=(field_data, *flux_data),
        log="benchmark",
    )
    num_data_arrays = num_flux_monitors + 6
    return sim_data, field_monitor.name, num_data_arrays


def test_simulation_load_export(split_string, tmp_path):
    major, minor, patch, *_ = __version__.split(".")
    path = os.path.join(tmp_path, f"simulation_{major}_{minor}_{patch}.json")
    path_hdf5 = os.path.join(tmp_path, f"simulation_{major}_{minor}_{patch}.h5")
    SIM_STATIC.to_file(path)
    SIM_STATIC.to_hdf5(path_hdf5)
    SIM2 = td.Simulation.from_file(path)
    SIM_HDF5 = td.Simulation.from_hdf5(path_hdf5)
    assert set_datasets_to_none(SIM_STATIC)._json_string == SIM2._json_string, (
        "original and loaded simulations are not the same"
    )
    assert SIM_STATIC == SIM_HDF5, "original and loaded from hdf5 simulations are not the same"


def test_simulation_load_export_yaml(tmp_path):
    path = str(tmp_path / "simulation.yaml")
    SIM.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert set_datasets_to_none(SIM)._json_string == SIM2._json_string, (
        "original and loaded simulations are not the same"
    )


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


def test_to_yaml_encodes_inf_nan_as_strings(tmp_path):
    """Lock YAML output format for Infinity/-Infinity/NaN (JSON-aligned string encoding)."""

    class InfNanModel(Tidy3dBaseModel):
        x: float
        y: float
        z: float

    path = str(tmp_path / "inf_nan.yaml")
    InfNanModel(x=math.inf, y=-math.inf, z=math.nan).to_yaml(path)

    with open(path, encoding="utf-8") as f:
        model_dict = yaml.safe_load(f)

    assert model_dict["x"] == "Infinity"
    assert model_dict["y"] == "-Infinity"
    assert model_dict["z"] == "NaN"

    loaded = InfNanModel.from_yaml(path)
    assert math.isinf(loaded.x) and loaded.x > 0
    assert math.isinf(loaded.y) and loaded.y < 0
    assert math.isnan(loaded.z)


def test_simulation_load_export_hdf5(split_string, tmp_path):
    path = str(tmp_path / "simulation.hdf5")
    SIM_STATIC.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert SIM_STATIC == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_gz(split_string, tmp_path):
    path = str(tmp_path / "simulation.hdf5.gz")
    SIM_STATIC.to_file(path)
    SIM2 = td.Simulation.from_file(path)
    assert SIM_STATIC == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_explicit(split_string, tmp_path):
    path = str(tmp_path / "simulation.hdf5")
    SIM_STATIC.to_hdf5(path)
    SIM2 = td.Simulation.from_hdf5(path)
    assert SIM_STATIC == SIM2, "original and loaded simulations are not the same"


def test_simulation_load_export_hdf5_gz_explicit(split_string, tmp_path):
    path = str(tmp_path / "simulation.hdf5.gz")
    SIM_STATIC.to_hdf5_gz(path)
    SIM2 = td.Simulation.from_hdf5_gz(path)
    assert SIM_STATIC == SIM2, "original and loaded simulations are not the same"


def make_many_flux_monitor_sim_data(num_monitors: int = 8) -> td.SimulationData:
    """Create a small SimulationData fixture with many tiny FluxData payloads."""

    flux_data = make_flux_data()
    monitor_freqs = flux_data.monitor.freqs
    freq0 = float(np.mean(monitor_freqs))
    fwidth = float(np.ptp(monitor_freqs)) or freq0 * 0.1
    monitors = tuple(
        td.FluxMonitor(center=(0, 0, 0), size=(1, 1, 0), freqs=monitor_freqs, name=f"flux_{i:03d}")
        for i in range(num_monitors)
    )
    simulation = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        monitors=monitors,
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
            )
        ],
        run_time=2e-12,
    )
    data = tuple(flux_data.updated_copy(monitor=monitor) for monitor in monitors)
    return td.SimulationData(simulation=simulation, data=data)


def test_simulation_data_from_file_monitor_names(split_string, tmp_path):
    sim_data = make_sim_data(symmetry=False)
    path = str(tmp_path / "sim_data.hdf5")
    sim_data.to_file(path)

    selected = td.SimulationData.from_file(path, monitor_names=("field", "flux"))
    expected = [
        monitor_data
        for monitor_data in sim_data.data
        if monitor_data.monitor.name in {"field", "flux"}
    ]

    assert [monitor.name for monitor in selected.simulation.monitors] == [
        monitor_data.monitor.name for monitor_data in expected
    ]
    assert [monitor_data.monitor.name for monitor_data in selected.data] == [
        monitor_data.monitor.name for monitor_data in expected
    ]
    for loaded_data, expected_data in zip(selected.data, expected):
        assert loaded_data == expected_data


def test_simulation_data_from_file_monitor_names_loads_only_selected_data_arrays(
    monkeypatch, tmp_path
):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    loaded_paths = []
    original_from_hdf5 = DataArray.from_hdf5.__func__

    def tracking_from_hdf5(cls, fname, group_path):
        loaded_paths.append(str(group_path).strip("/"))
        return original_from_hdf5(cls, fname, group_path)

    monkeypatch.setattr(DataArray, "from_hdf5", classmethod(tracking_from_hdf5))

    selected = td.SimulationData.from_file(path, monitor_names="flux_003")

    assert [monitor_data.monitor.name for monitor_data in selected.data] == ["flux_003"]
    assert loaded_paths == ["data/3/flux"]


def test_simulation_data_from_file_monitor_names_requires_hdf5(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5.gz")
    sim_data.to_file(path)

    with pytest.raises(ValueError, match="only works with '.hdf5' or '.h5' files"):
        td.SimulationData.from_file(path, monitor_names="flux_000")


def test_simulation_data_lazy_monitor_access_loads_only_requested_data(monkeypatch, tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.h5")
    sim_data.to_hdf5(path)

    loaded_paths = []
    original_from_hdf5 = DataArray.from_hdf5.__func__

    def tracking_from_hdf5(cls, fname, group_path):
        loaded_paths.append(str(group_path).strip("/"))
        return original_from_hdf5(cls, fname, group_path)

    monkeypatch.setattr(DataArray, "from_hdf5", classmethod(tracking_from_hdf5))

    lazy_data = td.SimulationData.from_file(path, lazy=True)

    assert "_lazy_fname" in lazy_data.__dict__
    assert [monitor.name for monitor in lazy_data.simulation.monitors] == [
        "flux_000",
        "flux_001",
        "flux_002",
        "flux_003",
    ]
    assert "_lazy_fname" in lazy_data.__dict__

    flux_data = lazy_data["flux_003"]

    assert isinstance(flux_data, td.FluxData)
    assert flux_data.monitor.name == "flux_003"
    assert loaded_paths == ["data/3/flux"]
    assert "_lazy_fname" in lazy_data.__dict__


def test_simulation_data_lazy_monitor_names_stay_selective(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    lazy_data = td.SimulationData.from_file(
        path, lazy=True, monitor_names=("flux_003", "flux_001", "flux_003")
    )

    assert "_lazy_fname" in lazy_data.__dict__
    assert [monitor.name for monitor in lazy_data.simulation.monitors] == [
        "flux_001",
        "flux_003",
    ]
    assert list(lazy_data.monitor_data) == ["flux_001", "flux_003"]
    assert lazy_data["flux_003"].monitor.name == "flux_003"
    assert "_lazy_fname" in lazy_data.__dict__

    with pytest.raises(KeyError):
        lazy_data["flux_000"]


def test_simulation_data_lazy_monitor_membership_stays_lazy(monkeypatch, tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_hdf5(path)

    loaded_paths = []
    original_from_hdf5 = DataArray.from_hdf5.__func__

    def tracking_from_hdf5(cls, fname, group_path):
        loaded_paths.append(str(group_path).strip("/"))
        return original_from_hdf5(cls, fname, group_path)

    monkeypatch.setattr(DataArray, "from_hdf5", classmethod(tracking_from_hdf5))

    lazy_data = td.SimulationData.from_file(path, lazy=True)

    assert "flux_001" in lazy_data.monitor_data
    assert "flux_999" not in lazy_data.monitor_data
    assert loaded_paths == []

    flux_data = lazy_data.monitor_data["flux_001"]

    assert flux_data.monitor.name == "flux_001"
    assert loaded_paths == ["data/1/flux"]


def test_simulation_data_lazy_get_monitor_by_name_stays_lazy(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    lazy_data = td.SimulationData.from_file(path, lazy=True)

    assert "_lazy_fname" in lazy_data.__dict__
    monitor = lazy_data.get_monitor_by_name("flux_003")

    assert monitor.name == "flux_003"
    assert "_lazy_fname" in lazy_data.__dict__


def test_simulation_data_lazy_data_materializes_selected_subset(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=4)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    lazy_data = td.SimulationData.from_file(path, lazy=True, monitor_names=("flux_001", "flux_003"))

    assert "_lazy_fname" in lazy_data.__dict__

    data = lazy_data.data

    assert "_lazy_fname" not in lazy_data.__dict__
    assert isinstance(lazy_data, td.SimulationData)
    assert [monitor_data.monitor.name for monitor_data in data] == ["flux_001", "flux_003"]
    assert [monitor.name for monitor in lazy_data.simulation.monitors] == [
        "flux_001",
        "flux_003",
    ]


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
    assert set_datasets_to_none(SIM)._json_string == sim_2._json_string

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


def test_projection_simulation_preserve_monitor_types(tmp_path):
    """Test that field-projection monitor types survive IO roundtrips."""

    path = str(tmp_path / "projection_simulation.json")
    SIM_PROJECTION.to_file(path)
    sim_2 = td.Simulation.from_file(path)

    monitor_types = [type(monitor) for monitor in sim_2.monitors]
    for monitor_type in (
        td.FieldProjectionAngleMonitor,
        td.FieldProjectionCartesianMonitor,
        td.FieldProjectionKSpaceMonitor,
        td.DirectivityMonitor,
    ):
        assert monitor_type in monitor_types


def test_1a_simulation_load_export2(tmp_path):
    path = str(tmp_path / "simulation.json")
    SIM2.to_file(path)
    SIM3 = td.Simulation.from_file(path)
    assert SIM2 == SIM3, "original and loaded simulations are not the same"


@pytest.mark.perf
def test_validation_speed(tmp_path):
    sizes_bytes = []
    times_sec = []
    path = str(tmp_path / "simulation.json")

    _ = SIM
    N_tests = 5  # may be increased temporarily, makes it slow for routine tests
    max_structures = np.log10(100)  # may be increased temporarily, makes it slow for routine tests

    # adjust as needed, keeping small to speed tests up
    num_structures = np.logspace(0, max_structures, N_tests).astype(int)

    for n in num_structures:
        new_structures = []
        for i in range(n):
            new_structure = SIM.structures[0].copy(update={"name": str(i)})
            new_structures.append(new_structure)
        S = SIM.copy(update={"structures": tuple(new_structures)})

        S.to_file(path)
        time_start = time()
        _S = td.Simulation.from_file(path)
        time_validate = time() - time_start
        times_sec.append(time_validate)
        assert set_datasets_to_none(S)._json_string == _S._json_string

        size = os.path.getsize(path)
        sizes_bytes.append(size)

        print(f"{n} structures \t {size:.1e} bytes \t {time_validate:.1f} seconds to validate")


@pytest.mark.perf
def test_simulation_data_selective_monitor_load_speed(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=256)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    target_name = "flux_255"
    num_repeats = 5

    # Warm the file-system cache before timing.
    _ = td.SimulationData.from_file(path)
    _ = td.SimulationData.from_file(path, monitor_names=target_name)

    full_times = []
    selective_times = []
    for _ in range(num_repeats):
        time_start = time()
        full_data = td.SimulationData.from_file(path)
        full_times.append(time() - time_start)

        time_start = time()
        selective_data = td.SimulationData.from_file(path, monitor_names=target_name)
        selective_times.append(time() - time_start)

    assert len(full_data.data) == 256
    assert [monitor_data.monitor.name for monitor_data in selective_data.data] == [target_name]

    full_time = np.median(full_times)
    selective_time = np.median(selective_times)
    file_size = os.path.getsize(path)
    speedup = full_time / selective_time

    print(
        f"Selective SimulationData load benchmark \t {file_size:.1e} bytes \t "
        f"full: {full_time * 1e3:.2f} ms \t selective: {selective_time * 1e3:.2f} ms \t "
        f"speedup: {speedup:.2f}x"
    )


@pytest.mark.perf
def test_simulation_data_lazy_monitor_access_speed(tmp_path):
    sim_data = make_many_flux_monitor_sim_data(num_monitors=256)
    path = str(tmp_path / "many_flux.hdf5")
    sim_data.to_file(path)

    target_name = "flux_255"
    num_repeats = 5

    _ = td.SimulationData.from_file(path)[target_name]
    _ = td.SimulationData.from_file(path, lazy=True)[target_name]

    full_times = []
    lazy_times = []
    for _ in range(num_repeats):
        time_start = time()
        full_monitor_data = td.SimulationData.from_file(path)[target_name]
        full_times.append(time() - time_start)

        time_start = time()
        lazy_monitor_data = td.SimulationData.from_file(path, lazy=True)[target_name]
        lazy_times.append(time() - time_start)

    assert full_monitor_data == lazy_monitor_data

    full_time = np.median(full_times)
    lazy_time = np.median(lazy_times)
    file_size = os.path.getsize(path)
    speedup = full_time / lazy_time

    print(
        f"Lazy SimulationData monitor access benchmark \t {file_size:.1e} bytes \t "
        f"full: {full_time * 1e3:.2f} ms \t lazy: {lazy_time * 1e3:.2f} ms \t "
        f"speedup: {speedup:.2f}x"
    )


SIM_FILES = [os.path.join(SIM_DIR, file) for file in os.listdir(SIM_DIR)]


@pytest.mark.parametrize("sim_file", SIM_FILES)
def test_simulation_updater(sim_file):
    """Test that all simulations in ``SIM_DIR`` can be updated to current version and loaded."""
    if "fdtd" in sim_file:
        sim_loaded = td.Simulation.from_file(sim_file)
    else:
        sim_loaded = td.HeatChargeSimulation.from_file(sim_file)
    assert sim_loaded.version == __version__, "Simulation not converted properly"

    # just make sure the loaded sim does something properly using this version
    assert sim_loaded.scene is not None


def test_simulation_updater_v2_10_mode_spec_sort_key_none(tmp_path):
    """Ensure v2.10 files with ``sort_spec.sort_key: null`` still load in current version."""
    sim = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        monitors=(
            td.ModeMonitor(
                center=(0, 0, 0),
                size=(1, 1, 0),
                freqs=[2e14],
                mode_spec=td.ModeSpec(),
                name="mode",
            ),
        ),
    )
    sim_dict = json.loads(sim.model_dump_json())
    sim_dict["version"] = "2.10.0"
    sim_dict["monitors"][0]["mode_spec"]["sort_spec"]["sort_key"] = None
    sim_dict["monitors"][0]["mode_spec"]["sort_spec"]["sort_order"] = "ascending"

    sim_path = tmp_path / "sim_v2_10_sort_key_none.json"
    sim_path.write_text(json.dumps(sim_dict), encoding="utf-8")

    sim_loaded = td.Simulation.from_file(sim_path)
    mode_monitor = next(monitor for monitor in sim_loaded.monitors if monitor.type == "ModeMonitor")
    assert mode_monitor.mode_spec.sort_spec.sort_key == "n_eff"
    assert mode_monitor.mode_spec.sort_spec.sort_order == "descending"
    assert sim_loaded.version == __version__


def test_simulation_updater_v2_10_mode_spec_sort_key_none_with_reference(tmp_path):
    """Ensure legacy null ``sort_key`` falls back to defaults even if sort fields are present."""
    sim = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        monitors=(
            td.ModeMonitor(
                center=(0, 0, 0),
                size=(1, 1, 0),
                freqs=[2e14],
                mode_spec=td.ModeSpec(),
                name="mode",
            ),
        ),
    )
    sim_dict = json.loads(sim.model_dump_json())
    sim_dict["version"] = "2.10.0"
    sort_spec = sim_dict["monitors"][0]["mode_spec"]["sort_spec"]
    sort_spec["sort_key"] = None
    sort_spec["sort_reference"] = 1.5
    sort_spec["sort_order"] = "ascending"

    sim_path = tmp_path / "sim_v2_10_sort_key_none_sort_reference.json"
    sim_path.write_text(json.dumps(sim_dict), encoding="utf-8")

    sim_loaded = td.Simulation.from_file(sim_path)
    mode_monitor = next(monitor for monitor in sim_loaded.monitors if monitor.type == "ModeMonitor")
    assert mode_monitor.mode_spec.sort_spec.sort_key == "n_eff"
    assert mode_monitor.mode_spec.sort_spec.sort_reference is None
    assert mode_monitor.mode_spec.sort_spec.sort_order == "descending"
    assert sim_loaded.version == __version__


def _make_mode_solver_for_legacy_sort_spec_tests() -> td.plugins.mode.ModeSolver:
    sim = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
    )
    return td.plugins.mode.ModeSolver(
        simulation=sim,
        plane=td.Box(center=(0, 0, 0), size=(1, 1, 0)),
        mode_spec=td.ModeSpec(),
        freqs=[2e14],
    )


def test_mode_solver_load_legacy_sort_key_none_without_version(tmp_path):
    """Ensure ModeSolver loads legacy ``sort_key: null`` even without a top-level version."""
    mode_solver = _make_mode_solver_for_legacy_sort_spec_tests()
    mode_solver_dict = json.loads(mode_solver.model_dump_json())

    assert "version" not in mode_solver_dict
    sort_spec = mode_solver_dict["mode_spec"]["sort_spec"]
    sort_spec["sort_key"] = None
    sort_spec["sort_reference"] = 1.5
    sort_spec["sort_order"] = "ascending"

    file_path = tmp_path / "mode_solver_sort_key_none.json"
    file_path.write_text(json.dumps(mode_solver_dict), encoding="utf-8")

    loaded = td.plugins.mode.ModeSolver.from_file(file_path)
    assert loaded.mode_spec.sort_spec.sort_key == "n_eff"
    assert loaded.mode_spec.sort_spec.sort_reference is None
    assert loaded.mode_spec.sort_spec.sort_order == "descending"


def test_mode_solver_load_legacy_sort_spec_none_without_version(tmp_path):
    """Ensure ModeSolver loads legacy ``sort_spec: null`` even without a top-level version."""
    mode_solver = _make_mode_solver_for_legacy_sort_spec_tests()
    mode_solver_dict = json.loads(mode_solver.model_dump_json())

    assert "version" not in mode_solver_dict
    mode_solver_dict["mode_spec"]["sort_spec"] = None

    file_path = tmp_path / "mode_solver_sort_spec_none.json"
    file_path.write_text(json.dumps(mode_solver_dict), encoding="utf-8")

    loaded = td.plugins.mode.ModeSolver.from_file(file_path)
    assert loaded.mode_spec.sort_spec.sort_key == "n_eff"
    assert loaded.mode_spec.sort_spec.sort_order == "descending"


def test_mode_solver_load_missing_sort_key_preserves_sort_fields(tmp_path):
    """Ensure valid payloads omitting ``sort_key`` keep explicit sort fields."""
    mode_solver = _make_mode_solver_for_legacy_sort_spec_tests()
    mode_solver_dict = json.loads(mode_solver.model_dump_json())

    sort_spec = mode_solver_dict["mode_spec"]["sort_spec"]
    sort_spec.pop("sort_key", None)
    sort_spec["sort_reference"] = 1.5
    sort_spec["sort_order"] = "ascending"

    file_path = tmp_path / "mode_solver_missing_sort_key.json"
    file_path.write_text(json.dumps(mode_solver_dict), encoding="utf-8")

    loaded = td.plugins.mode.ModeSolver.from_file(file_path)
    assert loaded.mode_spec.sort_spec.sort_key == "n_eff"
    assert loaded.mode_spec.sort_spec.sort_reference == 1.5
    assert loaded.mode_spec.sort_spec.sort_order == "ascending"


def test_yaml(tmp_path):
    path = str(tmp_path / "simulation.json")
    SIM.to_file(path)
    SIM.to_file("simulation.json")
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


def test_monitor_data_from_file():
    """Test the ability to load specific monitor data from a file."""

    sim = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        monitors=[
            td.FieldMonitor(center=(0, 0, 0), size=(1, 1, 0), freqs=[2e14], name="field"),
            td.ModeMonitor(
                center=(0, 0, 0), size=(1, 1, 0), freqs=[2e14], mode_spec=td.ModeSpec(), name="mode"
            ),
        ],
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            )
        ],
        run_time=2e-12,
    )

    sim_data = run_emulated(sim, task_name="test")

    fname = "tests/data/sim_data.hdf5"
    sim_data.to_file(fname)

    fld_data = td.SimulationData.mnt_data_from_file(fname, mnt_name="field")
    assert isinstance(fld_data, td.FieldData)
    assert fld_data.monitor == sim.monitors[0]

    mode_data = td.SimulationData.mnt_data_from_file(fname, mnt_name="mode")
    assert isinstance(mode_data, td.ModeData)
    assert mode_data.monitor == sim.monitors[1]


def test_simulation_data_from_file_reuses_hdf5_handle(monkeypatch, tmp_path):
    sim_data = make_sim_data()
    path = str(tmp_path / "sim_data.hdf5")
    sim_data.to_file(path)

    seen_handles = []
    real_from_hdf5 = DataArray.from_hdf5.__func__

    def recording_from_hdf5(cls, fname, group_path):
        seen_handles.append(fname)
        return real_from_hdf5(cls, fname, group_path)

    monkeypatch.setattr(DataArray, "from_hdf5", classmethod(recording_from_hdf5))

    loaded = td.SimulationData.from_file(path)

    assert loaded == sim_data
    assert seen_handles
    assert all(isinstance(handle, h5py.File) for handle in seen_handles)
    assert len({id(handle) for handle in seen_handles}) == 1


def test_monitor_data_from_file_reuses_hdf5_handle(split_string, monkeypatch, tmp_path):
    sim_data = make_sim_data()
    path = str(tmp_path / "sim_data_split.hdf5")
    sim_data.to_file(path)

    monitor_name = sim_data.simulation.monitors[0].name
    seen_handles = []
    real_from_hdf5 = DataArray.from_hdf5.__func__

    def recording_from_hdf5(cls, fname, group_path):
        seen_handles.append(fname)
        return real_from_hdf5(cls, fname, group_path)

    monkeypatch.setattr(DataArray, "from_hdf5", classmethod(recording_from_hdf5))

    loaded = td.SimulationData.mnt_data_from_file(path, mnt_name=monitor_name)

    assert loaded == sim_data.monitor_data[monitor_name]
    assert seen_handles
    assert all(isinstance(handle, h5py.File) for handle in seen_handles)
    assert len({id(handle) for handle in seen_handles}) == 1


def test_data_array_to_hdf5(tmp_path):
    values = np.linspace(0, 1, 10)
    coords = {"f": values}
    flux = td.FluxDataArray(values, coords=coords)

    path = str(tmp_path / "flux.hdf5")

    # pass a file handle
    with h5py.File(path, "w") as f_handle:
        flux.to_hdf5(fname=f_handle, group_path="test")

    # pass a file name
    flux.to_hdf5(fname=path, group_path="test")


@pytest.mark.perf
def test_hdf5_handle_reuse_speed(tmp_path):
    sim_data, monitor_name, num_data_arrays = _make_many_small_array_sim_data()
    path = str(tmp_path / "sim_data_perf.hdf5")
    sim_data.to_file(path)
    file_size = os.path.getsize(path)

    current_dict = _average_runtime(lambda: td.SimulationData.dict_from_hdf5(path))
    current_mnt = _average_runtime(lambda: td.SimulationData.mnt_data_from_file(path, monitor_name))

    assert td.SimulationData.from_file(path) == sim_data
    assert (
        td.SimulationData.mnt_data_from_file(path, monitor_name)
        == sim_data.monitor_data[monitor_name]
    )
    assert file_size < 1_000_000

    print(
        "SimulationData.dict_from_hdf5 "
        f"{current_dict * 1e3:.2f} ms ({num_data_arrays} DataArrays, {file_size / 1024:.1f} KiB)"
    )
    print(f"SimulationData.mnt_data_from_file {current_mnt * 1e3:.2f} ms")


def test_data_array_to_hdf5_string_coords(tmp_path):
    """Test HDF5 round-trip for DataArray with string coordinates."""
    from tidy3d.components.data.data_array import FreqTerminalTerminalDataArray

    f = [1e9, 2e9]
    terminal_label_out = ["t0", "t1"]
    terminal_label_in = ["t0", "t1"]
    coords = {
        "f": f,
        "terminal_label_out": terminal_label_out,
        "terminal_label_in": terminal_label_in,
    }
    data = FreqTerminalTerminalDataArray((1 + 1j) * np.random.random((2, 2, 2)), coords=coords)

    path = str(tmp_path / "terminal.hdf5")
    data.to_hdf5(fname=path, group_path="test")
    loaded = FreqTerminalTerminalDataArray.from_hdf5(fname=path, group_path="test")
    np.testing.assert_array_equal(data.values, loaded.values)


def test_inf_attrs(tmp_path):
    """But if infinity added to attrs."""

    path = str(tmp_path / "test.json")
    m = td.Medium(attrs={"a": td.inf})
    m.to_file(path)
    m2 = m.from_file(path)
    m2.to_file(path)
    m.from_file(path)
