"""Tests for the dipole emission plugin shell."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
import tidy3d.plugins.dipole_emission.study as study_module
from tidy3d.exceptions import DataError, SetupError
from tidy3d.plugins import dipole_emission


def bulk_power(study: dipole_emission.DipoleEmissionStudy, index: float) -> np.ndarray:
    """Independently computed bulk emitted power per squared dipole moment in C*um."""
    omega = 2 * np.pi * np.asarray(study.freqs, dtype=float)
    c_si = td.C_0 * 1e-6
    mu_0 = 4e-7 * np.pi
    pulse_spectrum_abs2 = np.asarray(study.pulse_spectrum_abs2, dtype=float)
    return index * omega**4 * mu_0 / (12 * np.pi * c_si) * 1e-12 * pulse_spectrum_abs2


def make_base_sim(**kwargs) -> td.Simulation:
    """Create a minimal source-free simulation."""
    defaults = {
        "size": (4.0, 4.0, 4.0),
        "grid_spec": td.GridSpec.auto(wavelength=1.5),
        "run_time": 1e-12,
    }
    defaults.update(kwargs)
    return td.Simulation(**defaults)


def make_region(**kwargs) -> dipole_emission.EmissionAnalysisRegion:
    """Create a valid analysis region."""
    defaults = {
        "center": (0.0, 0.0, 0.0),
        "size": (2.0, 2.0, 2.0),
        "normal_axis": 2,
        "direction": "+",
        "dl": 0.1,
    }
    defaults.update(kwargs)
    return dipole_emission.EmissionAnalysisRegion(**defaults)


def make_positions(data=None, index=None) -> td.PointDataArray:
    """Create sampled dipole positions."""
    if data is None:
        data = [[0.0, 0.0, 0.0], [0.2, 0.1, -0.1]]
    if index is None:
        index = np.arange(len(data))
    return td.PointDataArray(data, coords={"index": index, "axis": np.arange(3)})


def make_angles(data=None, index=None) -> td.SphericalAngleDataArray:
    """Create far-field observation angles."""
    if data is None:
        data = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    if index is None:
        index = np.arange(len(data))
    return td.SphericalAngleDataArray(
        data,
        coords={"index": index, "spherical_coordinate": ["theta", "phi"]},
    )


def make_study(**kwargs) -> dipole_emission.DipoleEmissionStudy:
    """Create a minimal valid study."""
    defaults = {
        "base_sim": make_base_sim(),
        "analysis_region": make_region(),
        "positions": make_positions(),
        "source_time": td.GaussianPulse(freq0=2e14, fwidth=1e13),
        "freqs": [1.9e14, 2.0e14],
        "angles": make_angles(),
    }
    defaults.update(kwargs)
    return dipole_emission.DipoleEmissionStudy(**defaults)


def make_data_array(value: float = 1.0) -> dipole_emission.DipoleEmissionStudyDataArray:
    """Create compact data array expected by DipoleEmissionStudyData."""
    coords = {
        "dipole_axis": ["x", "y", "z"],
        "polarization": ["p", "s"],
        "angle": [0, 1],
        "f": [1.9e14, 2.0e14],
    }
    shape = tuple(len(coords[dim]) for dim in dipole_emission.DipoleEmissionStudyDataArray._dims)
    return dipole_emission.DipoleEmissionStudyDataArray(
        np.full(shape, value),
        dims=dipole_emission.DipoleEmissionStudyDataArray._dims,
        coords=coords,
    )


def make_position_data_array(
    value: float = 1.0,
    index=None,
) -> dipole_emission.DipoleEmissionStudyPositionDataArray:
    """Create selected-position data array expected by DipoleEmissionStudyData.

    ``index`` defaults to ``[0]``; pass the position labels selected by the
    study's ``store_position_indexes`` so the result-contract check passes.
    """
    coords = {
        "index": [0] if index is None else list(index),
        "dipole_axis": ["x", "y", "z"],
        "polarization": ["p", "s"],
        "angle": [0, 1],
        "f": [1.9e14, 2.0e14],
    }
    shape = tuple(
        len(coords[dim]) for dim in dipole_emission.DipoleEmissionStudyPositionDataArray._dims
    )
    return dipole_emission.DipoleEmissionStudyPositionDataArray(
        np.full(shape, value),
        dims=dipole_emission.DipoleEmissionStudyPositionDataArray._dims,
        coords=coords,
    )


def make_monitor_data(
    study: dipole_emission.DipoleEmissionStudy,
    sim: td.Simulation,
    value: float,
    dtype: np.dtype = np.float32,
) -> td.DipoleEmissionData:
    """Create synthetic reduced monitor data for one study task."""
    monitor = sim.monitors[-1]
    coords = {"dipole_axis": ["x", "y", "z"], "f": study.freqs}
    intensity = td.DipoleEmissionDataArray(
        np.full((3, len(study.freqs)), value, dtype=dtype),
        dims=td.DipoleEmissionDataArray._dims,
        coords=coords,
    )

    position_intensity = None
    if study.store_position_indexes:
        position_coords = {"index": list(study.store_position_indexes), **coords}
        position_shape = (len(study.store_position_indexes), 3, len(study.freqs))
        position_intensity = td.DipoleEmissionPositionDataArray(
            np.full(position_shape, 10 * value, dtype=dtype),
            dims=td.DipoleEmissionPositionDataArray._dims,
            coords=position_coords,
        )

    return td.DipoleEmissionData(
        monitor=monitor,
        radiation_intensity=intensity,
        radiation_intensity_at_positions=position_intensity,
    )


def make_reduced_batch_data(
    study: dipole_emission.DipoleEmissionStudy,
    dtype: np.dtype = np.float32,
):
    """Create synthetic BatchData-like mapping for reduced emission monitor data."""
    batch_data = {}
    for task_offset, (task_name, _, _) in enumerate(study._expected_task_names(), start=1):
        sim = study.to_simulations()[task_name]
        monitor_data = make_monitor_data(
            study=study,
            sim=sim,
            value=float(task_offset),
            dtype=dtype,
        )
        batch_data[task_name] = td.SimulationData(simulation=sim, data=(monitor_data,))
    return batch_data


class FakeSimulationData:
    """Minimal monitor-name lookup object for compose contract tests."""

    def __init__(self, monitor_data_by_name):
        self.monitor_data_by_name = monitor_data_by_name
        self.requested_monitor_names = []

    def __getitem__(self, monitor_name):
        self.requested_monitor_names.append(monitor_name)
        if monitor_name not in self.monitor_data_by_name:
            raise KeyError(monitor_name)
        return self.monitor_data_by_name[monitor_name]


def test_analysis_region_inherits_box_and_helpers():
    region = make_region(normal_axis=0, direction="-", dl=(0.1, 0.2))

    assert isinstance(region, td.Box)
    assert region.center == (0.0, 0.0, 0.0)
    assert region.size == (2.0, 2.0, 2.0)
    assert region.tangential_axes == (1, 2)
    assert region.geometry == td.Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))

    override = region.mesh_override("_dipole_emission_mesh")
    assert override.geometry == region.geometry
    assert override.dl == (None, 0.1, 0.2)
    assert override.enforce

    with pytest.raises(ValidationError):
        make_region(dl=(0.1, 0.2, 0.3))
    with pytest.raises(ValidationError):
        make_region(dl=-0.1)


def test_study_spatial_inputs_are_validated():
    study = make_study(
        positions=make_positions([[0.0, 0.0, 0.0]], index=[7]),
        position_weights=[2.0],
        store_position_indexes=(0,),
    )

    assert isinstance(study.positions, td.PointDataArray)
    assert study.positions.coords["index"].values.tolist() == [7]
    np.testing.assert_allclose(study.position_weights_array, [2.0])
    assert study.store_position_indexes == (0,)

    axis_weighted_study = make_study(position_weights=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        axis_weighted_study.position_weights_array,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    with pytest.raises(ValidationError, match="position_weights"):
        make_study(position_weights=[1.0, 2.0, 3.0])
    with pytest.raises(ValidationError, match="position_weights"):
        make_study(position_weights=[[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValidationError, match="nonnegative"):
        make_study(position_weights=[1.0, -1.0])
    with pytest.raises(ValidationError, match="all zero") as exc_info:
        make_study(position_weights=[0.0, 0.0])
    assert exc_info.value.errors()[0]["loc"] == ("position_weights",)
    with pytest.raises(ValidationError, match="all zero"):
        make_study(position_weights=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # a single zeroed position (not all) and a zeroed orientation column are allowed
    make_study(position_weights=[1.0, 0.0])
    make_study(position_weights=[[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
    with pytest.raises(ValidationError, match="ordered"):
        make_study(
            positions=td.PointDataArray(
                [[2.0, 1.0, 3.0]],
                coords={"index": [0], "axis": ["y", "x", "z"]},
            )
        )
    with pytest.raises(ValidationError, match="duplicates"):
        make_study(store_position_indexes=(0, 0))
    with pytest.raises(ValidationError, match="valid position indexes"):
        make_study(store_position_indexes=(2,))


def test_study_rejects_sources_symmetry_and_reserved_monitor_names():
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        polarization="Ex",
    )
    with pytest.raises(ValidationError, match="must not contain sources") as exc_info:
        make_study(base_sim=make_base_sim(sources=(source,)))
    assert exc_info.value.errors()[0]["loc"] == ("base_sim", "sources")

    with pytest.raises(ValidationError, match="symmetry") as exc_info:
        make_study(base_sim=make_base_sim(symmetry=(1, 0, 0)))
    assert exc_info.value.errors()[0]["loc"] == ("base_sim", "symmetry")

    monitor = td.FieldMonitor(size=(0, 0, 0), freqs=[2e14], name="_dipole_emission_probe")
    with pytest.raises(ValidationError, match="reserved") as exc_info:
        make_study(base_sim=make_base_sim(monitors=(monitor,)))
    assert exc_info.value.errors()[0]["loc"] == ("base_sim", "monitors", 0, "name")


def test_dipole_emission_simulation_rejects_2d():
    # dipole emission is 3D-only (radiation_intensity is per-solid-angle). The guard
    # lives on the Simulation side, so it fires when the study builds its simulations
    # (and equally for a directly built Simulation + DipoleEmissionMonitor + TFSF).
    study = make_study(
        base_sim=make_base_sim(size=(4.0, 4.0, 0.0)),
        positions=make_positions([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]]),
    )
    with pytest.raises(ValidationError, match="three-dimensional") as exc_info:
        study.to_simulations()
    assert exc_info.value.errors()[0]["loc"] == ("size",)


def test_study_rejects_bad_positions_and_freqs():
    with pytest.raises(ValidationError):
        make_study(positions=[0.0, 0.0, 0.0])
    with pytest.raises(ValidationError):
        make_study(positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValidationError, match="exactly three"):
        make_study(positions=td.PointDataArray([[0.0, 0.0]], coords={"index": [0], "axis": [0, 1]}))
    with pytest.raises(ValidationError, match="real-valued"):
        make_study(positions=make_positions([[0.0 + 1.0j, 0.0, 0.0]], index=[0]))

    with pytest.raises(ValidationError, match=r"inside 'analysis_region'") as exc_info:
        make_study(positions=make_positions([[1.5, 0.0, 0.0]]))
    assert exc_info.value.errors()[0]["loc"] == ("positions",)

    with pytest.raises(ValidationError, match="inside the simulation domain") as exc_info:
        make_study(
            base_sim=make_base_sim(size=(2.0, 2.0, 2.0)),
            positions=make_positions([[1.0, 0.0, 0.0]], index=[0]),
        )
    assert exc_info.value.errors()[0]["loc"] == ("positions",)

    with pytest.raises(ValidationError, match=r"inside 'analysis_region'") as exc_info:
        make_study(positions=make_positions([[1.0, 0.0, 0.0]], index=[0]))
    assert exc_info.value.errors()[0]["loc"] == ("positions",)

    with pytest.raises(ValidationError, match="positive"):
        make_study(freqs=[-1.0])

    with pytest.raises(ValidationError, match="unique"):
        make_study(freqs=[2e14, 2e14])

    with pytest.raises(ValidationError, match="nonzero finite source spectrum") as exc_info:
        make_study(source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13, amplitude=0.0))
    assert exc_info.value.errors()[0]["loc"] == ("source_time",)


def test_study_rejects_invalid_tfsf_injection_side_media():
    normal_angles = make_angles([(0.0, 0.0)])
    polarizations = ("p",)
    with pytest.raises(ValidationError, match="real"):
        make_study(
            base_sim=make_base_sim(medium=td.Medium(conductivity=1e-3)),
            angles=normal_angles,
            polarizations=polarizations,
        ).to_simulations()


def test_background_index_uses_tfsf_injection_side_medium():
    entry_cladding = td.Structure(
        geometry=td.Box(center=(0.0, 0.0, 1.4), size=(td.inf, td.inf, 1.2)),
        medium=td.Medium(permittivity=2.25),
    )
    study = make_study(
        base_sim=make_base_sim(structures=(entry_cladding,)),
        analysis_region=make_region(direction="+"),
        angles=make_angles([(0.0, 0.0)]),
        polarizations=("p",),
    )
    sim = next(iter(study.to_simulations().values()))
    source = sim._dipole_emission_tfsf_source()

    assert np.isclose(sim._dipole_emission_background_index(source, study.freqs), 1.5)
    assert np.all(study.pulse_spectrum_abs2 > 0)


def test_study_uses_spherical_angle_data_array():
    study = make_study(
        angles=make_angles(
            [
                (0.1, 0.2),
                (0.3, 0.4),
                [0.5, 0.6],
            ],
            index=[3, 4, 5],
        )
    )

    assert isinstance(study.angles, td.SphericalAngleDataArray)
    assert study.angles.coords["index"].values.tolist() == [3, 4, 5]
    np.testing.assert_allclose(
        study.angles.sel(spherical_coordinate="theta").values,
        [0.1, 0.3, 0.5],
    )

    with pytest.raises(ValidationError):
        make_study(angles=[(0.1, 0.2), (0.3, 0.4)])

    with pytest.raises(ValidationError, match="SphericalAngleDataArray"):
        make_study(
            angles=td.SphericalAngleDataArray(
                [[0.8, 0.7]],
                coords={"index": [0], "spherical_coordinate": ["phi", "theta"]},
            )
        )
    with pytest.raises(ValidationError, match="real values"):
        make_study(angles=make_angles([[0.1 + 0.2j, 0.3]], index=[0]))
    with pytest.raises(ValidationError, match=r"abs\(theta\) < pi/2"):
        make_study(angles=make_angles([[np.pi / 2, 0.0]], index=[0]))
    with pytest.raises(ValidationError, match=r"abs\(theta\) < pi/2"):
        make_study(angles=make_angles([[-np.pi / 2, 0.0]], index=[0]))


def test_polarization_inputs_are_validated():
    assert make_study(polarizations=("p",)).polarizations == ("p",)
    assert make_study(polarizations=("s", "p")).polarizations == ("s", "p")

    with pytest.raises(ValidationError, match="duplicates"):
        make_study(polarizations=("p", "p"))

    with pytest.raises(ValidationError):
        make_study(polarizations=("x",))


def test_study_canonicalizes_unlabeled_point_and_angle_arrays(monkeypatch):
    study = make_study(
        positions=td.PointDataArray([[0.0, 0.0, 0.0]]),
        position_weights=[1.0],
        store_position_indexes=(0,),
        angles=td.SphericalAngleDataArray([[0.1, 0.2]]),
        polarizations=("p",),
    )
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    assert study.positions.dims == ("index", "axis")
    assert study.positions.coords["index"].values.tolist() == [0]
    assert study.positions.coords["axis"].values.tolist() == [0, 1, 2]
    assert study.angles.dims == ("index", "spherical_coordinate")
    assert study.angles.coords["spherical_coordinate"].values.tolist() == ["theta", "phi"]

    data = study.compose(make_reduced_batch_data(study))

    assert data.radiation_intensity_at_positions is not None
    assert data.radiation_intensity_at_positions.coords["index"].values.tolist() == [0]

    with pytest.raises(ValidationError):
        make_study(positions=td.PointDataArray([[0.0, 0.0, 0.0]], dims=("u", "v")))
    with pytest.raises(ValidationError):
        make_study(angles=td.SphericalAngleDataArray([[0.1, 0.2]], dims=("u", "v")))


def test_diagnostic_monitor_batch_size_guard(monkeypatch):
    diagnostic_monitor = td.FieldMonitor(size=(0, 0, 0), freqs=[2e14], name="diagnostic")
    monkeypatch.setattr(
        td.Simulation,
        "monitors_data_size",
        property(lambda self: {"diagnostic": 51e9}),
    )

    with pytest.raises(ValidationError, match="estimated to download"):
        make_study(base_sim=make_base_sim(monitors=(diagnostic_monitor,)))


def test_intrinsic_study_batch_size_guard(monkeypatch):
    monkeypatch.setattr(study_module, "MAX_SIMULATION_DATA_SIZE_GB", 2.0e-7)

    make_study()
    with pytest.raises(ValidationError, match="dipole-emission results"):
        make_study(store_position_indexes=(0,))


def test_intrinsic_study_batch_size_guard_accounts_for_double_precision(monkeypatch):
    monkeypatch.setattr(study_module, "MAX_SIMULATION_DATA_SIZE_GB", 4.0e-7)

    make_study(store_position_indexes=(0,))
    with pytest.raises(ValidationError, match="dipole-emission results"):
        make_study(
            base_sim=make_base_sim(precision="double"),
            store_position_indexes=(0,),
        )


def test_combined_study_batch_size_guard(monkeypatch):
    diagnostic_monitor = td.FieldMonitor(size=(0, 0, 0), freqs=[2e14], name="diagnostic")
    monkeypatch.setattr(study_module, "MAX_SIMULATION_DATA_SIZE_GB", 4.0e-7)
    monkeypatch.setattr(
        td.Simulation,
        "monitors_data_size",
        property(lambda self: {"diagnostic": 50.0}),
    )

    with pytest.raises(ValidationError, match="diagnostic monitors"):
        make_study(base_sim=make_base_sim(monitors=(diagnostic_monitor,)))


def test_to_simulations_uses_tfsf_and_dipole_emission_monitor():
    diagnostic_monitor = td.FieldMonitor(size=(0, 0, 0), freqs=[2e14], name="diagnostic")
    study = make_study(
        base_sim=make_base_sim(monitors=(diagnostic_monitor,)),
        analysis_region=make_region(normal_axis=1, direction="-", dl=0.1),
        angles=make_angles([(0.0, 0.2), (0.3, 0.4)]),
        polarizations=("p", "s"),
        position_weights=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        store_position_indexes=(1,),
    )

    simulations = study.to_simulations()

    assert list(simulations) == [
        "_dipole_emission_angle000_p",
        "_dipole_emission_angle000_s",
        "_dipole_emission_angle001_p",
        "_dipole_emission_angle001_s",
    ]

    normal_p = simulations["_dipole_emission_angle000_p"]
    normal_s = simulations["_dipole_emission_angle000_s"]
    oblique_s = simulations["_dipole_emission_angle001_s"]

    source = normal_p.sources[0]
    assert isinstance(source, td.TFSF)
    assert source.direction == "+"
    assert source.injection_axis == 1
    assert source.angle_theta == 0.0
    assert source.angle_phi == 0.2
    assert source.pol_angle == 0.0
    assert isinstance(source.angular_spec, td.FixedAngleSpec)
    assert normal_s.sources[0].pol_angle == np.pi / 2
    assert isinstance(oblique_s.sources[0].angular_spec, td.FixedAngleSpec)

    monitor = normal_p.monitors[-1]
    assert normal_p.monitors[0].name == "diagnostic"
    assert isinstance(monitor, td.DipoleEmissionMonitor)
    assert monitor.name == study_module.EMISSION_MONITOR_NAME
    assert monitor.fields == ("Ex", "Ey", "Ez")
    assert monitor.store_position_indexes == (1,)
    np.testing.assert_allclose(monitor.position_weights, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(monitor.points.values, study.positions.values)
    np.testing.assert_allclose(monitor.freqs, study.freqs)
    expected_monitor_size = monitor.storage_size(num_cells=0, tmesh=[])
    assert normal_p.monitors_data_size[monitor.name] == expected_monitor_size
    assert normal_p.updated_copy(precision="double").monitors_data_size[monitor.name] == (
        2 * expected_monitor_size
    )

    mesh_override = normal_p.grid_spec.override_structures[-1]
    assert mesh_override.name == study_module.MESH_OVERRIDE_NAME
    assert mesh_override.dl == (0.1, None, 0.1)
    assert mesh_override.enforce
    assert normal_p.normalize_index is None


def test_dipole_emission_monitor_validates_tfsf_pairing():
    study = make_study()
    sim = next(iter(study.to_simulations().values()))
    monitor = sim.monitors[-1]

    with pytest.raises(ValidationError, match="position_weights") as exc_info:
        monitor.updated_copy(position_weights=[1.0, 2.0, 3.0])
    assert exc_info.value.errors()[0]["loc"] == ("position_weights",)

    with pytest.raises(ValidationError, match="all zero") as exc_info:
        monitor.updated_copy(position_weights=[0.0, 0.0])
    assert exc_info.value.errors()[0]["loc"] == ("position_weights",)

    with pytest.raises(ValidationError, match="all zero") as exc_info:
        monitor.updated_copy(position_weights=np.zeros((2, 3)))
    assert exc_info.value.errors()[0]["loc"] == ("position_weights",)

    with pytest.raises(ValidationError, match="store_position_indexes") as exc_info:
        monitor.updated_copy(store_position_indexes=(0, 0))
    assert exc_info.value.errors()[0]["loc"] == ("store_position_indexes",)

    with pytest.raises(ValidationError, match="Input should be 'Ez'") as exc_info:
        monitor.updated_copy(fields=("Ex", "Ey", "Hx"))
    assert exc_info.value.errors()[0]["loc"] == ("fields", 2)

    with pytest.raises(ValidationError, match="exactly one source"):
        make_base_sim(monitors=(monitor,))

    bad_source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        polarization="Ex",
        name="dipole",
    )
    with pytest.raises(ValidationError, match="TFSF source"):
        make_base_sim(sources=(bad_source,), monitors=(monitor,))

    source = sim.sources[0]
    with pytest.raises(ValidationError, match="exactly one source"):
        make_base_sim(sources=(source, bad_source), monitors=(monitor,))

    outside_monitor = monitor.updated_copy(
        points=make_positions([[1.5, 0.0, 0.0]], index=[0]),
        position_weights=[1.0],
    )
    with pytest.raises(ValidationError, match="outside TFSF source"):
        make_base_sim(sources=(source,), monitors=(outside_monitor,))


def test_compose_stacks_reduced_monitor_data(monkeypatch):
    study = make_study(
        angles=make_angles([(0.0, 0.2), (0.3, 0.4)]),
        polarizations=("s", "p"),
    )
    batch_data = make_reduced_batch_data(study)
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    data = study.compose(batch_data)

    assert isinstance(data, dipole_emission.DipoleEmissionStudyData)
    assert data.radiation_intensity.dims == ("dipole_axis", "polarization", "angle", "f")
    assert data.radiation_intensity.dtype == np.float32
    transfer = data.radiation_intensity_transfer(bulk_refractive_index=1.85)
    assert transfer.dtype == np.float32
    # the transfer is a 1/sr ratio, not the intensity units carried by radiation_intensity
    assert transfer.attrs["units"] == "1/sr"
    assert "transfer" in transfer.attrs["long_name"]
    assert data.radiation_intensity.attrs["units"] != transfer.attrs["units"]
    assert data.radiation_intensity.coords["dipole_axis"].values.tolist() == ["x", "y", "z"]
    assert data.radiation_intensity.coords["polarization"].values.tolist() == ["s", "p"]
    assert data.radiation_intensity.coords["angle"].values.tolist() == [0, 1]
    # angle directions live on the study, exposed via theta/phi properties (not array coords)
    assert "theta" not in data.radiation_intensity.coords
    assert "phi" not in data.radiation_intensity.coords
    np.testing.assert_allclose(data.theta, [0.0, 0.3])
    np.testing.assert_allclose(data.phi, [0.2, 0.4])
    np.testing.assert_allclose(data.radiation_intensity.sel(polarization="s", angle=0), 1.0)
    np.testing.assert_allclose(data.radiation_intensity.sel(polarization="p", angle=0), 2.0)
    np.testing.assert_allclose(data.radiation_intensity.sel(polarization="s", angle=1), 3.0)
    total_bulk_power = 2.0 * bulk_power(study, 1.85)
    np.testing.assert_allclose(
        transfer.sel(polarization="p", angle=1),
        np.full((3, len(study.freqs)), 4.0) / total_bulk_power[None, :],
        rtol=1e-6,
    )
    assert "theta" not in transfer.coords
    assert data.radiation_intensity_at_positions is None


def test_compose_stacks_selected_position_data(monkeypatch):
    study = make_study(
        positions=make_positions(index=[10, 11]),
        angles=make_angles([(0.0, 0.2)]),
        polarizations=("p",),
        store_position_indexes=(1,),
    )
    batch_data = make_reduced_batch_data(study)
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    data = study.compose(batch_data)

    assert data.radiation_intensity_at_positions is not None
    transfer_at_positions = data.radiation_intensity_transfer_at_positions(
        bulk_refractive_index=1.85
    )
    assert transfer_at_positions is not None
    assert data.radiation_intensity_at_positions.dtype == np.float32
    assert transfer_at_positions.dtype == np.float32
    assert transfer_at_positions.attrs["units"] == "1/sr"
    assert "transfer" in transfer_at_positions.attrs["long_name"]
    assert data.radiation_intensity_at_positions.dims == (
        "index",
        "dipole_axis",
        "polarization",
        "angle",
        "f",
    )
    assert data.radiation_intensity_at_positions.coords["index"].values.tolist() == [11]
    np.testing.assert_allclose(data.radiation_intensity_at_positions.values, 10.0)
    np.testing.assert_allclose(
        transfer_at_positions.values,
        np.full((1, 3, 1, 1, len(study.freqs)), 10.0)
        / bulk_power(study, 1.85)[None, None, None, None, :],
        rtol=1e-6,
    )


def test_compose_preserves_double_precision_monitor_data(monkeypatch):
    study = make_study(
        angles=make_angles([(0.0, 0.2)]),
        polarizations=("p",),
    )
    batch_data = make_reduced_batch_data(study, dtype=np.float64)
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    data = study.compose(batch_data)

    assert data.radiation_intensity.dtype == np.float64
    assert data.radiation_intensity_transfer(bulk_refractive_index=1.85).dtype == np.float64


def test_compose_ignores_diagnostic_monitors_in_batch_data(monkeypatch):
    diagnostic_monitor = td.FieldMonitor(size=(0, 0, 0), freqs=[2e14], name="diagnostic")
    study = make_study(
        base_sim=make_base_sim(monitors=(diagnostic_monitor,)),
        angles=make_angles([(0.1, 0.2)]),
        polarizations=("p",),
    )
    sim = study.to_simulations()[study._task_name(0, "p")]
    emission_data = make_monitor_data(study=study, sim=sim, value=1.0)
    diagnostic_data = object()
    sim_data = FakeSimulationData(
        {
            study_module.EMISSION_MONITOR_NAME: emission_data,
            "diagnostic": diagnostic_data,
        }
    )
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    data = study.compose({study._task_name(0, "p"): sim_data})

    assert "batch_data" not in data.model_dump()
    assert "diagnostic" not in data.model_dump()
    assert sim_data.requested_monitor_names == [study_module.EMISSION_MONITOR_NAME]
    assert sim_data.monitor_data_by_name["diagnostic"] is diagnostic_data


def test_compose_rejects_missing_or_bad_study_monitor_data(monkeypatch):
    study = make_study(angles=make_angles([(0.1, 0.2)]), polarizations=("p",))
    task_name = study._task_name(0, "p")
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    with pytest.raises(SetupError, match="missing task"):
        study.compose(batch_data={})

    with pytest.raises(SetupError, match="missing monitor"):
        study.compose(batch_data={task_name: FakeSimulationData({})})

    with pytest.raises(SetupError, match="DipoleEmissionData"):
        study.compose(
            batch_data={
                task_name: FakeSimulationData({study_module.EMISSION_MONITOR_NAME: object()})
            }
        )

    sim = study.to_simulations()[task_name]
    other_study = make_study(
        angles=make_angles([(0.1, 0.2)]),
        polarizations=("p",),
        freqs=[1.8e14, 2.1e14],
    )
    other_data = make_monitor_data(
        study=other_study,
        sim=other_study.to_simulations()[task_name],
        value=1.0,
    )
    with pytest.raises(SetupError, match="freqs"):
        study.compose(
            batch_data={
                task_name: FakeSimulationData({study_module.EMISSION_MONITOR_NAME: other_data})
            }
        )

    bad_data = make_monitor_data(study=study, sim=sim, value=1.0)
    bad_data = bad_data.updated_copy(
        radiation_intensity=td.DipoleEmissionDataArray(
            np.ones((2, len(study.freqs))),
            coords={"dipole_axis": ["x", "y"], "f": study.freqs},
        ),
        validate=False,
    )
    with pytest.raises(SetupError, match="has shape"):
        study.compose(
            batch_data={
                task_name: FakeSimulationData({study_module.EMISSION_MONITOR_NAME: bad_data})
            }
        )


def test_compose_rejects_mismatched_source_provenance(monkeypatch):
    study = make_study(angles=make_angles([(0.1, 0.2)]), polarizations=("p",))
    task_name = study._task_name(0, "p")
    sim = study.to_simulations()[task_name]
    monitor_data = make_monitor_data(study=study, sim=sim, value=1.0)
    bad_source = sim.sources[0].updated_copy(angle_phi=0.4)
    bad_sim = sim.updated_copy(sources=(bad_source,))
    sim_data = td.SimulationData(simulation=bad_sim, data=(monitor_data,))
    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", lambda _: None)

    with pytest.raises(SetupError, match="source 'angle_phi'"):
        study.compose(batch_data={task_name: sim_data})


def test_dipole_emission_monitor_data_frequency_error_loc():
    study = make_study(angles=make_angles([(0.1, 0.2)]), polarizations=("p",))
    sim = study.to_simulations()[study._task_name(0, "p")]
    monitor_data = make_monitor_data(study=study, sim=sim, value=1.0)
    bad_intensity = td.DipoleEmissionDataArray(
        np.ones((3, len(study.freqs))),
        coords={"dipole_axis": ["x", "y", "z"], "f": np.asarray(study.freqs) + 1.0},
    )

    with pytest.raises(ValidationError) as exc_info:
        td.DipoleEmissionData(
            monitor=monitor_data.monitor,
            radiation_intensity=bad_intensity,
            radiation_intensity_at_positions=monitor_data.radiation_intensity_at_positions,
        )

    assert exc_info.value.errors()[0]["loc"] == ("radiation_intensity",)


def test_dipole_emission_data_round_trips_hdf5(tmp_path):
    study = make_study(
        angles=make_angles([(0.1, 0.2), (0.3, 0.4)]),
        position_weights=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        store_position_indexes=(0,),
    )
    data = dipole_emission.DipoleEmissionStudyData(
        radiation_intensity=make_data_array(),
        radiation_intensity_at_positions=make_position_data_array(),
        study=study,
    )
    path = tmp_path / "dipole_emission_data.hdf5"

    data.to_hdf5(path)
    loaded = dipole_emission.DipoleEmissionStudyData.from_hdf5(path)

    assert isinstance(loaded.radiation_intensity, dipole_emission.DipoleEmissionStudyDataArray)
    assert isinstance(
        loaded.radiation_intensity_at_positions,
        dipole_emission.DipoleEmissionStudyPositionDataArray,
    )
    assert loaded.radiation_intensity.dims == data.radiation_intensity.dims
    np.testing.assert_allclose(loaded.radiation_intensity.values, data.radiation_intensity.values)

    # angle directions are not stored on the arrays (so the HDF5 round trip cannot
    # drop them); they survive on the embedded study and are read via theta/phi
    assert "theta" not in loaded.radiation_intensity.coords
    np.testing.assert_allclose(loaded.theta, [0.1, 0.3])
    np.testing.assert_allclose(loaded.phi, [0.2, 0.4])

    # per-axis weights enter the position-summed transfer denominator
    total_bulk_power_by_axis = np.array([5.0, 7.0, 9.0])[:, None] * bulk_power(study, 1.85)[None, :]
    np.testing.assert_allclose(
        loaded.radiation_intensity_transfer(bulk_refractive_index=1.85).values,
        data.radiation_intensity.values / total_bulk_power_by_axis[:, None, None, :],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        loaded.radiation_intensity_transfer_at_positions(bulk_refractive_index=1.85).values,
        data.radiation_intensity_at_positions.values
        / bulk_power(study, 1.85)[None, None, None, None, :],
        rtol=1e-6,
    )
    assert "radiation_intensity_transfer" not in loaded.model_dump()
    assert loaded.study.angles.dims == ("index", "spherical_coordinate")
    np.testing.assert_allclose(loaded.study.position_weights_array, study.position_weights_array)
    assert loaded.study.store_position_indexes == (0,)


def test_run_return_contract(monkeypatch):
    study = make_study(angles=make_angles([(0.1, 0.2)]), polarizations=("p",))
    batch_data = object()
    emission_data = object()
    calls = []

    def fake_license(feature_name):
        calls.append(feature_name)

    def fake_run_batch(self, folder_name, batch_kwargs):
        np.testing.assert_allclose(self.angles.values, [[0.1, 0.2]])
        assert self.polarizations == ("p",)
        assert folder_name == "folder"
        assert batch_kwargs == {"verbose": False}
        return batch_data

    def fake_compose(self, data):
        assert data is batch_data
        return emission_data

    monkeypatch.setattr(study_module, "check_tidy3d_extras_licensed_feature", fake_license)
    monkeypatch.setattr(dipole_emission.DipoleEmissionStudy, "_run_batch", fake_run_batch)
    monkeypatch.setattr(dipole_emission.DipoleEmissionStudy, "compose", fake_compose)

    assert (
        study.run(
            folder_name="folder",
            return_batch_data=False,
            verbose=False,
        )
        is emission_data
    )
    assert study.run(
        folder_name="folder",
        return_batch_data=True,
        verbose=False,
    ) == (emission_data, batch_data)
    assert calls == ["dipole_emission", "dipole_emission"]


def test_raw_dipole_emission_monitor_pre_upload_checks_license(monkeypatch):
    study = make_study(angles=make_angles([(0.1, 0.2)]), polarizations=("p",))
    sim = study.to_simulations()[study._task_name(0, "p")]
    calls = []

    monkeypatch.setattr(
        "tidy3d.components.simulation.check_tidy3d_extras_licensed_feature",
        lambda feature_name: calls.append(feature_name),
    )

    sim.validate_pre_upload()

    assert calls == ["dipole_emission"]


def test_run_batch_uses_study_simulations_and_batch_api(monkeypatch, tmp_path):
    study = make_study()
    calls = []

    class FakeBatch:
        def __init__(self, simulations, folder_name, verbose=True, solver_version=None):
            calls.append(
                {
                    "simulations": simulations,
                    "folder_name": folder_name,
                    "verbose": verbose,
                    "solver_version": solver_version,
                }
            )

        def run(self, **kwargs):
            calls.append({"run_kwargs": kwargs})
            return "batch-data"

    monkeypatch.setattr("tidy3d.web.Batch", FakeBatch)

    result = study._run_batch(
        folder_name="folder",
        batch_kwargs={
            "verbose": False,
            "solver_version": "solver-test",
            "path_dir": tmp_path,
            "priority": 3,
        },
    )

    assert result == "batch-data"
    assert list(calls[0]["simulations"]) == [
        "_dipole_emission_angle000_p",
        "_dipole_emission_angle000_s",
        "_dipole_emission_angle001_p",
        "_dipole_emission_angle001_s",
        "_dipole_emission_angle002_p",
        "_dipole_emission_angle002_s",
    ]
    assert calls[0]["folder_name"] == "folder"
    assert calls[0]["verbose"] is False
    assert calls[0]["solver_version"] == "solver-test"
    assert calls[1] == {"run_kwargs": {"path_dir": tmp_path, "priority": 3}}


def test_transfer_per_position_bulk_index():
    study = make_study(
        angles=make_angles([(0.1, 0.2), (0.3, 0.4)]),
        position_weights=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        store_position_indexes=(1,),
    )
    data = dipole_emission.DipoleEmissionStudyData(
        radiation_intensity=make_data_array(6.0),
        # store_position_indexes=(1,) selects position label 1
        radiation_intensity_at_positions=make_position_data_array(3.0, index=[1]),
        study=study,
    )
    # integrated transfer takes one index per sampled position (full set)
    index = np.array([1.5, 2.0])
    transfer = data.radiation_intensity_transfer(bulk_refractive_index=index)
    # per-axis total bulk power: sum_i w_ia * n_i = [1*1.5 + 4*2, 2*1.5 + 5*2, 3*1.5 + 6*2]
    total_index_by_axis = np.array([9.5, 13.0, 16.5])
    expected = 6.0 / (total_index_by_axis[:, None] * bulk_power(study, 1.0)[None, :])
    np.testing.assert_allclose(
        transfer.values,
        np.broadcast_to(expected[:, None, None, :], transfer.shape),
        rtol=1e-6,
    )

    # per-position transfer takes one index per stored position (subset), aligned
    # to the index coordinate of radiation_intensity_at_positions
    index_at = np.array([2.5])
    transfer_at_positions = data.radiation_intensity_transfer_at_positions(
        bulk_refractive_index=index_at
    )
    expected_at = 3.0 / bulk_power(study, 2.5)
    np.testing.assert_allclose(
        transfer_at_positions.values,
        np.broadcast_to(expected_at, transfer_at_positions.shape),
        rtol=1e-6,
    )

    # a full-length index passed to the per-position method is rejected
    with pytest.raises(DataError, match="one value per stored position"):
        data.radiation_intensity_transfer_at_positions(bulk_refractive_index=index)

    # plain Python list behaves identically to the ndarray form
    transfer_list = data.radiation_intensity_transfer(bulk_refractive_index=[1.5, 2.0])
    np.testing.assert_allclose(transfer_list.values, transfer.values, rtol=1e-6)
    transfer_at_list = data.radiation_intensity_transfer_at_positions(bulk_refractive_index=[2.5])
    np.testing.assert_allclose(transfer_at_list.values, transfer_at_positions.values, rtol=1e-6)


def test_transfer_finite_for_float32_band_edges():
    # band-edge frequencies where the source spectrum is ~1e-40: the old
    # |S|^2-only normalization overflowed float32 to inf here
    study = make_study(
        angles=make_angles([(0.1, 0.2), (0.3, 0.4)]),
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        freqs=[1.5e14, 2.0e14],
    )
    coords = {
        "dipole_axis": ["x", "y", "z"],
        "polarization": ["p", "s"],
        "angle": [0, 1],
        "f": study.freqs,
    }
    shape = tuple(len(coords[dim]) for dim in dipole_emission.DipoleEmissionStudyDataArray._dims)
    intensity = dipole_emission.DipoleEmissionStudyDataArray(
        np.full(shape, 1e10, dtype=np.float32),
        dims=dipole_emission.DipoleEmissionStudyDataArray._dims,
        coords=coords,
    )
    data = dipole_emission.DipoleEmissionStudyData(radiation_intensity=intensity, study=study)

    transfer = data.radiation_intensity_transfer(bulk_refractive_index=1.85)

    assert transfer.dtype == np.float32
    assert np.all(np.isfinite(transfer.values))
    np.testing.assert_allclose(
        transfer.values,
        np.full(shape, 1e10) / (2.0 * bulk_power(study, 1.85))[None, None, None, :],
        rtol=1e-6,
    )


def test_transfer_invalid_bulk_index():
    study = make_study(angles=make_angles([(0.1, 0.2), (0.3, 0.4)]))
    data = dipole_emission.DipoleEmissionStudyData(
        radiation_intensity=make_data_array(),
        study=study,
    )

    with pytest.raises(DataError, match="finite positive"):
        data.radiation_intensity_transfer(bulk_refractive_index=-1.85)
    with pytest.raises(DataError, match="real values"):
        data.radiation_intensity_transfer(bulk_refractive_index=np.array([1.5 + 0.1j, 1.6 + 0.2j]))
    with pytest.raises(DataError, match="one value per sampled position"):
        data.radiation_intensity_transfer(bulk_refractive_index=[1.0, 2.0, 3.0])


def test_integrated_transfer_is_weighted_average_of_per_position():
    # Big-picture invariant: with the same reference index, the integrated transfer
    # equals the position_weights-weighted average of the per-position transfers,
    # because radiation_intensity is the weighted sum sum_i w_i * I_i and both sides
    # divide by the same P_bulk.
    study = make_study(
        angles=make_angles([(0.1, 0.2), (0.3, 0.4)]),
        position_weights=[1.0, 3.0],
        store_position_indexes=(0, 1),
    )
    pos_coords = {
        "index": [0, 1],
        "dipole_axis": ["x", "y", "z"],
        "polarization": ["p", "s"],
        "angle": [0, 1],
        "f": study.freqs,
    }
    pos_shape = tuple(
        len(pos_coords[d]) for d in dipole_emission.DipoleEmissionStudyPositionDataArray._dims
    )
    per_position = np.zeros(pos_shape)
    per_position[0] = 2.0  # I_0
    per_position[1] = 5.0  # I_1
    rip = dipole_emission.DipoleEmissionStudyPositionDataArray(
        per_position,
        dims=dipole_emission.DipoleEmissionStudyPositionDataArray._dims,
        coords=pos_coords,
    )
    weights = np.array([1.0, 3.0])
    # radiation_intensity = weighted sum = 1*2 + 3*5 = 17 (constant over axis/pol/angle/f)
    data = dipole_emission.DipoleEmissionStudyData(
        radiation_intensity=make_data_array(float(weights @ np.array([2.0, 5.0]))),
        radiation_intensity_at_positions=rip,
        study=study,
    )

    n_em = 1.85
    transfer = data.radiation_intensity_transfer(bulk_refractive_index=n_em)
    transfer_at_positions = data.radiation_intensity_transfer_at_positions(
        bulk_refractive_index=n_em
    )

    expected = (weights[:, None, None, None, None] * transfer_at_positions.values).sum(
        axis=0
    ) / weights.sum()
    np.testing.assert_allclose(transfer.values, expected, rtol=1e-6)


def test_data_rejects_inconsistent_angle_dimension():
    # _validate_angle_dimension: the array's integer 'angle' axis must match study.angles
    study = make_study(angles=make_angles([(0.1, 0.2), (0.3, 0.4)]))  # 2 angles
    coords = {
        "dipole_axis": ["x", "y", "z"],
        "polarization": ["p", "s"],
        "angle": [0, 1, 2],  # 3 angles -> inconsistent with study.angles (2)
        "f": study.freqs,
    }
    shape = tuple(len(coords[dim]) for dim in dipole_emission.DipoleEmissionStudyDataArray._dims)
    bad = dipole_emission.DipoleEmissionStudyDataArray(
        np.zeros(shape),
        dims=dipole_emission.DipoleEmissionStudyDataArray._dims,
        coords=coords,
    )
    with pytest.raises(ValidationError, match="inconsistent with") as exc_info:
        dipole_emission.DipoleEmissionStudyData(radiation_intensity=bad, study=study)
    assert exc_info.value.errors()[0]["loc"] == ("radiation_intensity",)


def test_data_rejects_inconsistent_position_index():
    # _validate_stored_position_index: the position array's 'index' axis must carry
    # the position labels selected by 'store_position_indexes' (what compose assigns)
    study = make_study(
        angles=make_angles([(0.1, 0.2), (0.3, 0.4)]),  # 2 angles (angle axis consistent)
        store_position_indexes=(1,),  # selects position label 1 -> expects index [1]
    )
    # array carries index=[0] instead of the expected [1]
    bad = make_position_data_array(index=[0])
    with pytest.raises(ValidationError, match="store_position_indexes") as exc_info:
        dipole_emission.DipoleEmissionStudyData(
            radiation_intensity=make_data_array(),
            radiation_intensity_at_positions=bad,
            study=study,
        )
    assert exc_info.value.errors()[0]["loc"] == ("radiation_intensity_at_positions",)
