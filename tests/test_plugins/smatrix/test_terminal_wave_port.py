"""Unit tests for TerminalWavePort."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.data.data_array import (
    CurrentFreqTerminalModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    ModeIndexDataArray,
    ScalarModeFieldDataArray,
    VoltageFreqTerminalModeDataArray,
)
from tidy3d.components.microwave.data.dataset import (
    TerminalFieldDataset,
    TransmissionLineTerminalDataset,
)
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeSolverData
from tidy3d.components.microwave.mode_spec import MicrowaveTerminalModeSpec
from tidy3d.components.microwave.source import MicrowaveTerminalSource
from tidy3d.components.mode.mode_solver import ModeSolver
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
    z0_data = np.zeros((1, n_terminals, n_terminals), dtype=complex)
    for i in range(n_terminals):
        z0_data[0, i, i] = 50.0
    Z0 = ImpedanceFreqTerminalTerminalDataArray(z0_data, coords=z0_coords)

    # Voltage and current transforms: identity matrices [f, terminal_label, mode_index]
    transform_coords = {"f": freqs, "terminal_label": terminal_labels, "mode_index": mode_indices}
    identity = np.eye(n_terminals).reshape(1, n_terminals, n_modes)
    voltage_transform = VoltageFreqTerminalModeDataArray(
        identity.astype(complex), coords=transform_coords
    )
    current_transform = CurrentFreqTerminalModeDataArray(
        identity.astype(complex), coords=transform_coords
    )

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

    # Verify terminal_fields
    tf = mode_data.terminal_fields
    assert tf is not None
    assert isinstance(tf, TerminalFieldDataset)
    assert "terminal_label" in tf.Ex.dims
    assert list(tf.Ex.coords["terminal_label"].values) == terminal_labels


def test_terminal_wave_port_plot_port():
    """Test plot_port for TerminalWavePort with differential pairs."""
    import matplotlib.pyplot as plt

    sim, _ = _make_differential_stripline_sim_and_port()
    freqs = np.array([1e9, 10e9])

    mil = 25.4
    L = 4000 * mil
    len_inf = 1e6
    port = TerminalWavePort(
        center=(0, 0, -L / 2),
        size=(len_inf, len_inf, 0),
        direction="+",
        name="terminal",
        differential_pairs=(("T0", "T1"),),
    )

    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=freqs,
    )

    # plot_port by name (str path) — cascades through full plotting stack
    ax = modeler.plot_port("terminal")
    assert ax is not None
    plt.close()

    # plot_port by port object
    ax = modeler.plot_port(port)
    assert ax is not None
    plt.close()

    # plot_sim_grid
    modeler.plot_sim_grid(z=port.center[2])
    plt.close()


def test_terminal_wave_port_plot_port_custom_voltage_specs():
    """Test plot_port with explicit voltage specs and invalid port type error."""
    import matplotlib.pyplot as plt

    spec0 = _make_custom_spec()
    spec1 = _make_custom_spec(voltage_spec=_make_voltage_spec(center=(0.5, 0, 0), size=(0.5, 0, 0)))
    port = TerminalWavePort(
        center=(0, 0, 0),
        size=(2, 2, 0),
        direction="+",
        name="twp_volt",
        terminal_specs=(spec0, spec1),
    )
    sim = td.Simulation(
        size=(4, 4, 4),
        center=(0, 0, 0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 10e9, min_steps_per_wvl=10),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        structures=[],
        monitors=[],
        run_time=1e-9,
    )
    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=np.array([1e9, 10e9]),
    )

    ax = modeler.plot_port("twp_volt")
    assert ax is not None
    plt.close()

    # Invalid port type raises ValueError
    with pytest.raises(ValueError, match="Invalid port type"):
        modeler.plot_port(42)
