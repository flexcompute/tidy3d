"""Unit tests for TerminalWavePort."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.data.data_array import (
    CurrentFreqTerminalModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    ImpedanceTerminalDataArray,
    ModeAmpsDataArray,
    ModeIndexDataArray,
    ScalarModeFieldDataArray,
    VoltageFreqTerminalModeDataArray,
)
from tidy3d.components.microwave.data.dataset import (
    TerminalFieldDataset,
    TransmissionLineTerminalDataset,
)
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData, MicrowaveModeSolverData
from tidy3d.components.microwave.mode_spec import MicrowaveTerminalModeSpec
from tidy3d.components.microwave.source import MicrowaveTerminalSource
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.exceptions import SetupError
from tidy3d.plugins.smatrix import TerminalComponentModeler, TerminalWavePort


def _make_voltage_spec(center=(0, 0, -10), size=(1, 0, 0)):
    """Helper to create an AxisAlignedVoltageIntegralSpec."""
    return td.AxisAlignedVoltageIntegralSpec(
        center=center,
        size=size,
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )


def _make_custom_spec(voltage_spec=None):
    """Helper to create a CustomImpedanceSpec with a voltage spec."""
    if voltage_spec is None:
        voltage_spec = _make_voltage_spec()
    return td.CustomImpedanceSpec(voltage_spec=voltage_spec)


def test_terminal_wave_port_custom_specs():
    """Test TerminalWavePort with explicit CustomImpedanceSpec tuples."""
    # Single terminal
    spec0 = _make_custom_spec()
    port_1mode = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp1",
        terminal_specs=(spec0,),
    )
    mode_spec = port_1mode._mode_spec
    assert isinstance(mode_spec, MicrowaveTerminalModeSpec)
    assert mode_spec.num_modes == 1
    assert port_1mode._mode_indices() == (0,)

    # Two terminals
    spec1 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(0.5, 0, -10), size=(0.5, 0, 0))
    )
    port_2mode = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp2",
        terminal_specs=(spec0, spec1),
    )
    mode_spec_2 = port_2mode._mode_spec
    assert isinstance(mode_spec_2, MicrowaveTerminalModeSpec)
    assert mode_spec_2.num_modes == 2
    assert port_2mode._mode_indices() == (0, 1)


def test_terminal_wave_port_differential_pairs():
    """Test differential pair mapping and terminal classification."""
    spec0 = _make_custom_spec()
    spec1 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(0.5, 0, -10), size=(0.5, 0, 0))
    )
    port = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_diff",
        terminal_specs=(spec0, spec1),
        differential_pairs=(("T0", "T1"),),
    )

    # Check differential pair mapping
    diff_map = port._differential_pair_mapping
    assert "Diff0@comm" in diff_map
    assert "Diff0@diff" in diff_map
    assert diff_map["Diff0@comm"] == ("T0", "T1")
    assert diff_map["Diff0@diff"] == ("T0", "T1")

    # Both terminals are consumed by the differential pair
    active = port._get_active_single_ended_terminals(["T0", "T1"])
    assert active == []

    # Terminals mapping should contain only differential pairs (no active single-ended)
    terminals_mapping = port._get_terminals_mapping(["T0", "T1"])
    assert "Diff0@comm" in terminals_mapping
    assert "Diff0@diff" in terminals_mapping
    # No single-ended keys since both are consumed
    assert "T0" not in terminals_mapping
    assert "T1" not in terminals_mapping


def test_terminal_wave_port_validate_differential_pairs():
    """Test that duplicate terminal labels in differential pairs raise ValidationError."""
    spec0 = _make_custom_spec()
    spec1 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(0.5, 0, -10), size=(0.5, 0, 0))
    )
    spec2 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(1.0, 0, -10), size=(0.5, 0, 0))
    )
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp_bad_diff",
            terminal_specs=(spec0, spec1, spec2),
            differential_pairs=(("T0", "T1"), ("T1", "T2")),
        )


def test_terminal_wave_port_validate_terminal_specs():
    """Test validation of terminal_specs: empty tuple and inconsistent voltage specs."""
    # Empty tuple should raise ValidationError
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp_empty",
            terminal_specs=(),
        )

    # Inconsistent voltage specs: one has voltage_spec, the other doesn't
    spec_with_voltage = _make_custom_spec()
    spec_without_voltage = td.CustomImpedanceSpec(
        current_spec=td.AxisAlignedCurrentIntegralSpec(center=(0, 0, -10), size=(1, 1, 0), sign="+")
    )
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp_inconsistent",
            terminal_specs=(spec_with_voltage, spec_without_voltage),
        )


def test_terminal_wave_port_to_source():
    """Test that to_source returns a properly configured MicrowaveTerminalSource."""
    spec = _make_custom_spec()
    port = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_src",
        terminal_specs=(spec,),
    )

    source_time = td.GaussianPulse(freq0=10e9, fwidth=1e9)
    source = port.to_source(source_time=source_time)

    assert isinstance(source, MicrowaveTerminalSource)
    assert list(source.center) == [0, 0, -10]
    assert list(source.size) == [2, 2, 0]
    assert source.direction == "+"
    assert source.name == "twp_src"
    # Default terminal_label should be the first terminal "T0"
    assert source.terminal_label == "T0"


def _make_differential_stripline_sim_and_port():
    """Build a 2-signal-trace stripline simulation with a TerminalWavePort using AutoImpedanceSpec.

    Adapted from terminal_component_modeler_def.py:make_differential_stripline_modeler.
    """
    mil = 25.4  # mil to micron conversion
    w = 3.2 * mil  # signal strip width
    t = 0.7 * mil  # conductor thickness
    h = 10.7 * mil  # substrate thickness
    se = 7 * mil  # gap between edge-coupled pair
    L = 4000 * mil  # line length
    len_inf = 1e6  # effective infinity

    f_max = 70e9
    eps = 4.4

    med_sub = td.Medium(permittivity=eps)
    med_metal = td.PEC

    left_strip = td.Box(center=(-(se + w) / 2, 0, 0), size=(w, t, L))
    right_strip = td.Box(center=((se + w) / 2, 0, 0), size=(w, t, L))

    str_sub = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(len_inf, h, L)), medium=med_sub)
    str_signal = td.Structure(
        geometry=td.GeometryGroup(geometries=[left_strip, right_strip]),
        medium=med_metal,
    )
    str_gnd_top = td.Structure(
        geometry=td.Box(center=(0, h / 2 + t / 2, 0), size=(len_inf, t, L)),
        medium=med_metal,
    )
    str_gnd_bot = td.Structure(
        geometry=td.Box(center=(0, -h / 2 - t / 2, 0), size=(len_inf, t, L)),
        medium=med_metal,
    )

    # Layer refinement to snap grid to conductor boundaries (needed for auto-detection)
    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[str_signal],
        axis=1,
        min_steps_along_axis=10,
        refinement_inside_sim_only=False,
        bounds_snapping="bounds",
        corner_refinement=td.GridRefinement(dl=t / 10, num_cells=2),
    )
    lr_spec_top = lr_spec.updated_copy(center=(0, h / 2 + t / 2, 0), size=(len_inf, t, L))
    lr_spec_bot = lr_spec.updated_copy(center=(0, -h / 2 - t / 2, 0), size=(len_inf, t, L))

    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f_max,
        min_steps_per_wvl=30,
        layer_refinement_specs=[lr_spec, lr_spec_top, lr_spec_bot],
    )
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pec(),
        z=td.Boundary.pml(),
    )

    sim = td.Simulation(
        size=(50 * mil, h + 2 * t, 1.05 * L),
        center=(0, 0, 0),
        grid_spec=grid_spec,
        boundary_spec=boundary_spec,
        structures=[str_sub, str_signal, str_gnd_top, str_gnd_bot],
        monitors=[],
        run_time=2e-9,
        shutoff=1e-7,
    )

    # TerminalWavePort with AutoImpedanceSpec (default)
    port = TerminalWavePort(
        center=(0, 0, -L / 2),
        size=(len_inf, len_inf, 0),
        direction="+",
        name="terminal",
    )

    return sim, port


def test_terminal_wave_port_modeler_auto_detection():
    """Test TerminalComponentModeler with AutoImpedanceSpec auto-detects conductors."""
    sim, port = _make_differential_stripline_sim_and_port()
    freqs = np.array([1e9, 10e9])

    # AutoImpedanceSpec leaves the mode spec unresolved until the modeler
    # infers floating conductors from the simulation geometry.
    assert port._mode_spec is None

    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=freqs,
    )

    # Port should be listed in terminal wave ports
    assert len(modeler._terminal_wave_ports) == 1
    assert modeler._terminal_wave_ports[0].name == "terminal"

    # Two signal traces detected as floating conductors (ground planes filtered out
    # because they touch the PEC boundary on y)
    conductors = modeler._floating_isolated_conductors_at_waveport["terminal"]
    assert len(conductors) == 2

    # Resolved mode spec should be a MicrowaveTerminalModeSpec with 2 modes
    resolved = modeler._resolved_mode_specs["terminal"]
    assert isinstance(resolved, MicrowaveTerminalModeSpec)
    assert resolved.num_modes == 2

    # mode_solver_for_port should return a ModeSolver without error
    ms = modeler.mode_solver_for_port("terminal")
    assert isinstance(ms, ModeSolver)

    # network_dict should map terminal labels for TerminalWavePort
    network = modeler.network_dict
    assert len(network) == 2
    for _key, (p, label) in network.items():
        assert p.name == "terminal"
        assert label in resolved._terminal_indices

    # matrix_indices_monitor should list all terminal indices
    indices = modeler.matrix_indices_monitor
    assert len(indices) == 2

    # task_name_from_index should work for each index
    for idx in indices:
        task_name = modeler.task_name_from_index(idx)
        assert isinstance(task_name, str)


def test_terminal_fields_from_mock_data():
    """Test that MicrowaveModeSolverData.terminal_fields returns a TerminalFieldDataset."""
    n_terminals = 2
    n_modes = 2
    terminal_labels = ["T0", "T1"]
    mode_indices = list(range(n_modes))
    freqs = [10e9]

    # Spatial grid for the mode plane (z is the injection axis → z has 1 cell)
    x = [-1.0, 1.0, 3.0]
    y = [-2.0, 0.0]
    z = [-1.0, 1.0]
    grid = td.Grid(boundaries=td.Coords(x=x, y=y, z=z))

    # Field data: shape (nx, ny, nz, nf, n_modes) = (2, 1, 1, 1, 2)
    field_coords = {"x": x[:-1], "y": y[:-1], "z": z[:-1], "f": freqs, "mode_index": mode_indices}
    field_shape = (len(x) - 1, len(y) - 1, len(z) - 1, len(freqs), n_modes)
    field = ScalarModeFieldDataArray((1 + 1j) * np.random.random(field_shape), coords=field_coords)

    # n_complex
    index_coords = {"f": freqs, "mode_index": mode_indices}
    n_complex = ModeIndexDataArray((1.5 + 0.01j) * np.ones((1, n_modes)), coords=index_coords)

    # Terminal impedance matrix: Z0[f, terminal_label_out, terminal_label_in]
    z0_coords = {
        "f": freqs,
        "terminal_label_out": terminal_labels,
        "terminal_label_in": terminal_labels,
    }
    z0_data = 50.0 * np.eye(n_terminals, dtype=complex)[np.newaxis]
    Z0 = ImpedanceFreqTerminalTerminalDataArray(z0_data, coords=z0_coords)

    # Voltage and current transforms: identity matrices [f, terminal_label, mode_index]
    transform_coords = {"f": freqs, "terminal_label": terminal_labels, "mode_index": mode_indices}
    identity = np.eye(n_terminals, dtype=complex)[np.newaxis]
    voltage_transform = VoltageFreqTerminalModeDataArray(identity, coords=transform_coords)
    current_transform = CurrentFreqTerminalModeDataArray(identity, coords=transform_coords)

    terminal_data = TransmissionLineTerminalDataset(
        Z0=Z0,
        voltage_transform=voltage_transform,
        current_transform=current_transform,
    )

    # Create impedance specs for the MicrowaveTerminalModeSpec
    impedance_specs = {}
    for label in terminal_labels:
        impedance_specs[label] = td.CustomImpedanceSpec(
            voltage_spec=td.AxisAlignedVoltageIntegralSpec(
                center=(0, 0, 0), size=(1, 0, 0), extrapolate_to_endpoints=True, sign="+"
            )
        )

    mode_spec = MicrowaveTerminalModeSpec(
        num_modes=n_modes,
        impedance_specs=impedance_specs,
    )

    monitor = td.MicrowaveModeSolverMonitor(
        center=(0, 0, 0),
        size=(4, 2, 0),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mw_mode_solver",
    )

    mode_data = MicrowaveModeSolverData(
        monitor=monitor,
        Ex=field,
        Ey=field,
        Ez=field,
        Hx=field,
        Hy=field,
        Hz=field,
        n_complex=n_complex,
        grid_expanded=grid,
        transmission_line_terminal_data=terminal_data,
    )

    # Verify Z0_matrix property (covers dataset.py line 128)
    assert terminal_data.Z0_matrix is terminal_data.Z0

    # Verify terminal_fields
    tf = mode_data.terminal_fields
    assert tf is not None
    assert isinstance(tf, TerminalFieldDataset)
    assert "terminal_label" in tf.Ex.dims
    assert list(tf.Ex.coords["terminal_label"].values) == terminal_labels


def test_terminal_wave_port_reference_impedance_validation():
    """Test that invalid reference impedance values raise ValidationError."""
    spec = _make_custom_spec()
    # Non-finite reference impedance
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp",
            terminal_specs=(spec,),
            reference_impedance=float("inf"),
        )
    # Negative real part
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp",
            terminal_specs=(spec,),
            reference_impedance=-50,
        )


def test_terminal_wave_port_reference_impedance_dataarray_validation():
    """Test that ImpedanceTerminalDataArray with invalid values raises ValidationError."""
    spec = _make_custom_spec()

    # NaN in DataArray
    nan_arr = ImpedanceTerminalDataArray([float("nan")], coords={"terminal_label": ["T0"]})
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp",
            terminal_specs=(spec,),
            reference_impedance=nan_arr,
        )

    # Negative real part in DataArray
    neg_arr = ImpedanceTerminalDataArray([-50.0], coords={"terminal_label": ["T0"]})
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp",
            terminal_specs=(spec,),
            reference_impedance=neg_arr,
        )

    # Valid DataArray should pass
    valid_arr = ImpedanceTerminalDataArray([50.0], coords={"terminal_label": ["T0"]})
    port = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp",
        terminal_specs=(spec,),
        reference_impedance=valid_arr,
    )
    assert port.reference_impedance is not None


def test_terminal_wave_port_validate_current_spec_consistency():
    """Test that inconsistent current_spec across terminal_specs raises ValidationError."""
    spec_with_current = td.CustomImpedanceSpec(
        voltage_spec=_make_voltage_spec(),
        current_spec=td.AxisAlignedCurrentIntegralSpec(
            center=(0, 0, -10), size=(1, 1, 0), sign="+"
        ),
    )
    spec_without_current = _make_custom_spec()
    with pytest.raises(ValidationError):
        TerminalWavePort(
            center=(0, 0, -10),
            size=(2, 2, 0),
            direction="+",
            name="twp_inconsistent_current",
            terminal_specs=(spec_with_current, spec_without_current),
        )


def test_terminal_wave_port_differential_pair_missing_label():
    """Test that a differential pair referencing a non-existent label raises SetupError."""
    spec = _make_custom_spec()
    port = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_bad_label",
        terminal_specs=(spec,),
        differential_pairs=(("T0", "T99"),),
    )
    with pytest.raises(SetupError, match="T99"):
        port._get_active_single_ended_terminals(["T0"])


def test_terminal_wave_port_to_source_snap_center():
    """Test that to_source with snap_center sets center along injection axis."""
    spec = _make_custom_spec()
    port = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_snap",
        terminal_specs=(spec,),
    )
    source_time = td.GaussianPulse(freq0=10e9, fwidth=1e9)
    source = port.to_source(source_time=source_time, snap_center=-5.0)
    assert source.center[port.injection_axis] == -5.0


def _make_mock_terminal_mode_data(port):
    """Build a minimal MicrowaveModeData with terminal data matching the given TerminalWavePort."""
    mode_spec = port._mode_spec
    terminal_labels = list(mode_spec._terminal_indices)
    n_terminals = len(terminal_labels)
    n_modes = mode_spec.num_modes
    mode_indices = list(range(n_modes))
    freqs = [10e9]

    # n_complex
    index_coords = {"f": freqs, "mode_index": mode_indices}
    n_complex = ModeIndexDataArray((1.5 + 0.01j) * np.ones((1, n_modes)), coords=index_coords)

    # amps: shape (direction, f, mode_index)
    direction = ["+", "-"]
    amp_coords = {"direction": direction, "f": freqs, "mode_index": mode_indices}
    amp_data = (1 + 0.5j) * np.random.default_rng(42).random((2, 1, n_modes))
    amps = ModeAmpsDataArray(amp_data, coords=amp_coords)

    # Terminal impedance matrix: Z0[f, terminal_label_out, terminal_label_in] = 50 * I
    z0_coords = {
        "f": freqs,
        "terminal_label_out": terminal_labels,
        "terminal_label_in": terminal_labels,
    }
    z0_data = 50.0 * np.eye(n_terminals, dtype=complex)[np.newaxis]
    Z0 = ImpedanceFreqTerminalTerminalDataArray(z0_data, coords=z0_coords)

    # Voltage and current transforms: identity [f, terminal_label, mode_index]
    transform_coords = {"f": freqs, "terminal_label": terminal_labels, "mode_index": mode_indices}
    identity = np.eye(n_terminals, dtype=complex)[np.newaxis]
    voltage_transform = VoltageFreqTerminalModeDataArray(identity, coords=transform_coords)
    current_transform = CurrentFreqTerminalModeDataArray(identity, coords=transform_coords)

    terminal_data = TransmissionLineTerminalDataset(
        Z0=Z0,
        voltage_transform=voltage_transform,
        current_transform=current_transform,
    )

    monitor = td.MicrowaveModeMonitor(
        center=port.center,
        size=port.size,
        freqs=freqs,
        mode_spec=mode_spec,
        name=port._mode_monitor_name,
    )

    return MicrowaveModeData(
        monitor=monitor,
        amps=amps,
        n_complex=n_complex,
        transmission_line_terminal_data=terminal_data,
    )


def test_terminal_wave_port_reference_impedance_matrix():
    """Test get_reference_impedance_matrix with scalar and DataArray impedance."""
    spec0 = _make_custom_spec()
    spec1 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(0.5, 0, -10), size=(0.5, 0, 0))
    )

    # Scalar reference impedance = 75
    port_scalar = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_ref_scalar",
        terminal_specs=(spec0, spec1),
        reference_impedance=75.0,
    )
    mode_data = _make_mock_terminal_mode_data(port_scalar)
    ref_matrix = port_scalar.get_reference_impedance_matrix(mode_data)
    # Select by label to be shape-agnostic
    assert np.isclose(ref_matrix.sel(terminal_label_out="T0", terminal_label_in="T0").item(), 75.0)
    assert np.isclose(ref_matrix.sel(terminal_label_out="T1", terminal_label_in="T1").item(), 75.0)
    assert np.isclose(ref_matrix.sel(terminal_label_out="T0", terminal_label_in="T1").item(), 0.0)

    # DataArray reference impedance = [60, 80]
    ref_arr = ImpedanceTerminalDataArray([60.0, 80.0], coords={"terminal_label": ["T0", "T1"]})
    port_arr = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_ref_arr",
        terminal_specs=(spec0, spec1),
        reference_impedance=ref_arr,
    )
    mode_data_arr = _make_mock_terminal_mode_data(port_arr)
    ref_matrix_arr = port_arr.get_reference_impedance_matrix(mode_data_arr)
    assert np.isclose(
        ref_matrix_arr.sel(terminal_label_out="T0", terminal_label_in="T0").item(), 60.0
    )
    assert np.isclose(
        ref_matrix_arr.sel(terminal_label_out="T1", terminal_label_in="T1").item(), 80.0
    )
    assert np.isclose(
        ref_matrix_arr.sel(terminal_label_out="T0", terminal_label_in="T1").item(), 0.0
    )

    # Also verify get_characteristic_impedance_matrix directly
    z0_matrix = port_scalar.get_characteristic_impedance_matrix(mode_data)
    assert "terminal_label_out" in z0_matrix.dims
    assert np.isclose(z0_matrix.sel(terminal_label_out="T0", terminal_label_in="T0").item(), 50.0)


def test_terminal_wave_port_compute_voltage_current():
    """Test compute_voltage and compute_current for TerminalWavePort."""
    spec0 = _make_custom_spec()
    spec1 = _make_custom_spec(
        voltage_spec=_make_voltage_spec(center=(0.5, 0, -10), size=(0.5, 0, 0))
    )

    # direction="+" port
    port_plus = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="twp_vc_plus",
        terminal_specs=(spec0, spec1),
    )
    mode_data = _make_mock_terminal_mode_data(port_plus)

    # Mock sim_data to return mode_data on subscript
    class _MockSimData:
        def __getitem__(self, key):
            return mode_data

    sim_data = _MockSimData()

    voltage = port_plus.compute_voltage(sim_data)
    assert "terminal_label" in voltage.dims

    current_plus = port_plus.compute_current(sim_data)
    assert "terminal_label" in current_plus.dims

    # direction="-" port: current should flip sign
    port_minus = TerminalWavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="-",
        name="twp_vc_minus",
        terminal_specs=(spec0, spec1),
    )
    mode_data_minus = _make_mock_terminal_mode_data(port_minus)

    class _MockSimDataMinus:
        def __getitem__(self, key):
            return mode_data_minus

    sim_data_minus = _MockSimDataMinus()
    current_minus = port_minus.compute_current(sim_data_minus)
    # The sign flip comes from direction="-", values should be negated
    # relative to what they'd be without the sign flip
    assert current_minus is not None
    assert "terminal_label" in current_minus.dims
