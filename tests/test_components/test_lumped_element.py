"""Tests lumped elements."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.geometry.utils import SnapBehavior, SnapLocation
from tidy3d.components.lumped_element import (
    MAX_SPICE_FILE_SIZE_BYTES,
    LumpedCircuitComponent,
    LumpedNodeMapper,
    network_complex_permittivity,
)


def test_lumped_resistor():
    resistor = td.LumpedResistor(
        resistance=50.0,
        center=[0, 0, 0],
        size=[2, 0, 3],
        voltage_axis=0,
        name="R",
    )
    _ = resistor._sheet_conductance
    normal_axis = resistor.normal_axis
    assert normal_axis == 1

    # Check conversion to geometry and to structure
    _ = resistor.to_geometry()
    _ = resistor.to_structure()

    # Check conversion to mesh overrides
    _ = resistor.to_mesh_overrides()
    _ = resistor.to_snapping_points()

    # Check conversion to monitor
    freqs = np.linspace(1e9, 50e9, 101)
    monitor = resistor.to_monitor(freqs=freqs)
    assert monitor.name == resistor.monitor_name

    # error if voltage axis is not in plane with the resistor
    with pytest.raises(ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[2, 0, 3],
            voltage_axis=1,
            name="R",
        )

    # error if not planar
    with pytest.raises(ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[0, 0, 3],
            voltage_axis=2,
            name="R",
        )
    with pytest.raises(ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[2, 1, 3],
            voltage_axis=2,
            name="R",
        )


def test_lumped_resistor_snapping():
    resistor = td.LumpedResistor(
        resistance=50.0,
        center=[0, 0, 0.1],
        size=[1, 1, 0],
        voltage_axis=0,
        name="R",
        enable_snapping_points=False,
        num_grid_cells=None,
    )
    # a box completely covers the lumped element in xy
    box = td.Structure(
        geometry=td.Box(center=(0, 0, -1), size=(1.5, 1.5, 0.2)),
        medium=td.Medium(permittivity=4),
    )
    sim = td.Simulation(
        size=(5, 5, 5),
        grid_spec=td.GridSpec.auto(wavelength=1, min_steps_per_wvl=11.1),
        lumped_elements=[resistor],
        structures=[box],
        run_time=1e-19,
    )

    # snapped version
    resistor_snapped = resistor.updated_copy(enable_snapping_points=True)
    sim_snapped = sim.updated_copy(lumped_elements=(resistor_snapped,))
    # whether lumped element is snapped along normal axis
    assert not any(np.isclose(sim.grid.boundaries.z, 0.1))
    assert any(np.isclose(sim_snapped.grid.boundaries.z, 0.1))

    # whether lumped element is snapped along voltage axis
    assert not any(np.isclose(sim.grid.boundaries.x, 0.5))
    assert any(np.isclose(sim_snapped.grid.boundaries.x, 0.5))
    assert not any(np.isclose(sim.grid.boundaries.x, -0.5))
    assert any(np.isclose(sim_snapped.grid.boundaries.x, -0.5))


def test_coaxial_lumped_resistor_snapping():
    resistor = td.CoaxialLumpedResistor(
        resistance=50.0,
        center=[0, 0, 0.1],
        outer_diameter=1,
        inner_diameter=0.5,
        normal_axis=2,
        name="R",
        enable_snapping_points=False,
        num_grid_cells=None,
    )
    # a box completely covers the lumped element in xy
    box = td.Structure(
        geometry=td.Box(center=(0, 0, -1), size=(1.5, 1.5, 0.2)),
        medium=td.Medium(permittivity=4),
    )
    sim = td.Simulation(
        size=(5, 5, 5),
        grid_spec=td.GridSpec.auto(wavelength=1, min_steps_per_wvl=11.1),
        lumped_elements=[resistor],
        structures=[box],
        run_time=1e-19,
    )

    # snapped version
    resistor_snapped = resistor.updated_copy(enable_snapping_points=True)
    sim_snapped = sim.updated_copy(lumped_elements=(resistor_snapped,))
    # whether lumped element is snapped along normal axis
    assert not any(np.isclose(sim.grid.boundaries.z, 0.1))
    assert any(np.isclose(sim_snapped.grid.boundaries.z, 0.1))


def test_coaxial_lumped_resistor():
    resistor = td.CoaxialLumpedResistor(
        resistance=50.0,
        center=[0, 0, 0],
        outer_diameter=3,
        inner_diameter=1,
        normal_axis=1,
        name="R",
    )

    _ = resistor._sheet_conductance
    normal_axis = resistor.normal_axis
    assert normal_axis == 1

    # Check conversion to geometry and to structure
    _ = resistor.to_geometry()
    _ = resistor.to_structure()

    # Check conversion to mesh overrides
    _ = resistor.to_mesh_overrides()
    _ = resistor.to_snapping_points()

    # error if inner diameter is larger
    with pytest.raises(ValidationError):
        _ = td.CoaxialLumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            outer_diameter=3,
            inner_diameter=4,
            normal_axis=1,
            name="R",
        )

    with pytest.raises(ValidationError):
        _ = td.CoaxialLumpedResistor(
            resistance=50.0,
            center=[0, 0, np.inf],
            outer_diameter=3,
            inner_diameter=1,
            normal_axis=1,
            name="R",
        )


def test_validators_RLC_network():
    """Test that ``RLCNetwork`` is validated correctly."""
    # Must have a defined value for R,L,or C
    with pytest.raises(ValidationError):
        _ = td.RLCNetwork()

    # Must have a valid topology
    with pytest.raises(ValidationError):
        _ = td.RLCNetwork(
            capacitance=0.2e-12,
            network_topology="left",
        )


def test_validators_admittance_network():
    """Test that ``AdmittanceNetwork`` is validated correctly."""
    with pytest.raises(ValidationError):
        _ = td.AdmittanceNetwork()

    a = (0, -1, 2)
    b = (1, 1, 2)
    # non negative a and b
    with pytest.raises(ValidationError):
        _ = td.AdmittanceNetwork(
            a=a,
            b=b,
        )

    a = (0, complex(1, 2), 2)
    b = (1, 1, 2)
    # real a and b
    with pytest.raises(ValidationError):
        _ = td.AdmittanceNetwork(
            a=a,
            b=b,
        )


@pytest.mark.parametrize("Rval", [None, 75])
@pytest.mark.parametrize("Lval", [None, 5e-9])
@pytest.mark.parametrize("Cval", [None, 0.2 * 1e-12])
@pytest.mark.parametrize("topology", ["series", "parallel"])
def test_RLC_and_lumped_network_agreement(Rval, Lval, Cval, topology):
    """Test that the manual conversions in ``RLCNetwork`` match ``AdmittanceNetwork``."""

    if all([not Rval, not Lval, not Cval]):
        return

    # Relax accuracy of check when inductors are present in a parallel configuration (or by themselves)
    # We add a bit of loss to the PoleResidue medium, since lossless second order pole is not supported.
    configuration_includes_parallel_inductor = Lval and (
        topology == "parallel" or (not Rval and not Cval)
    )

    rtol = 1e-7
    if configuration_includes_parallel_inductor:
        rtol = 1e-2

    fstart = 1e9
    fstop = 30e9
    freqs = np.linspace(fstart, fstop, 100)

    RLC = td.RLCNetwork(
        resistance=Rval,
        capacitance=Cval,
        inductance=Lval,
        network_topology=topology,
    )

    linear_element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[2, 0, 3],
        voltage_axis=0,
        network=RLC,
        name="RLC",
    )
    # Check conversion to geometry and to structure
    _ = linear_element.to_geometry()

    sf = linear_element._admittance_transfer_function_scaling()
    med_RLC = RLC._to_medium(sf)
    (a, b) = RLC._as_admittance_function

    eps_from_RLC_med = med_RLC.eps_model(freqs)
    eps_direct = 1 + sf * network_complex_permittivity(a=a, b=b, freqs=freqs)

    assert np.allclose(eps_from_RLC_med, eps_direct, rtol=rtol)

    if configuration_includes_parallel_inductor:
        return

    network = td.AdmittanceNetwork(a=a, b=b)

    (a, b) = network._as_admittance_function
    med_network = network._to_medium(sf)
    # Check conversion to geometry and to structure
    linear_element = linear_element.updated_copy(network=network)
    _ = linear_element.to_geometry()
    assert np.allclose(med_RLC.eps_model(freqs), med_network.eps_model(freqs), rtol=rtol)


@pytest.mark.parametrize("voltage_axis", [0, 1, 2])
@pytest.mark.parametrize("normal_axis_idx", [1, 2])
def test_lumped_element_orientations(voltage_axis, normal_axis_idx):
    """Try all possible orientations of the lumped element."""
    normal_axis = (voltage_axis + normal_axis_idx) % 3

    size = [3, 3, 3]
    size[normal_axis] = 0
    resistor = td.LumpedResistor(
        resistance=50.0,
        center=[0, 0, 0],
        size=size,
        voltage_axis=voltage_axis,
        name="R",
    )

    voltage_axis_2d = resistor._voltage_axis_2d
    assert voltage_axis_2d == 0 or voltage_axis_2d == 1


def test_impedance_admittance_calculation():
    """Test that the impedance and admittance calculations function properly."""

    Rval = 75
    Cval = 0.2 * 1e-12
    fstart = 1e9
    fstop = 30e9
    freqs = np.linspace(fstart, fstop, 100)

    Z_expected = Rval + 1j / (Cval * freqs * np.pi * 2)
    Y_expected = 1 / Z_expected

    RLC = td.RLCNetwork(
        resistance=Rval,
        capacitance=Cval,
        network_topology="series",
    )

    linear_element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[2, 0, 3],
        voltage_axis=0,
        network=RLC,
        name="RLC",
    )

    Z_calc = linear_element.impedance(freqs)
    assert np.allclose(Z_expected, Z_calc)

    Y_calc = linear_element.admittance(freqs)
    assert np.allclose(Y_expected, Y_calc)


def test_1d_lumped_element_not_allowed():
    """Test that 1D lumped elements (two zero-size dimensions) are not allowed.

    Users must provide a finite width along the lateral axis. This ensures the
    normal axis can be properly determined for the underlying 2D material.
    """
    RLC = td.RLCNetwork(resistance=50)

    # Attempting to create a 1D lumped element should raise a validation error
    with pytest.raises(ValidationError):
        td.LinearLumpedElement(
            center=[0, 0, 0],
            size=[1, 0, 0],  # 1D element: only x has non-zero size (invalid)
            voltage_axis=0,
            network=RLC,
            name="1D_RLC",
        )

    # Other 1D configurations should also fail
    with pytest.raises(ValidationError):
        td.LinearLumpedElement(
            center=[0, 0, 0],
            size=[0, 1, 0],  # 1D element: only y has non-zero size (invalid)
            voltage_axis=1,
            network=RLC,
            name="1D_RLC",
        )

    with pytest.raises(ValidationError):
        td.LinearLumpedElement(
            center=[0, 0, 0],
            size=[0, 0, 1],  # 1D element: only z has non-zero size (invalid)
            voltage_axis=2,
            network=RLC,
            name="1D_RLC",
        )

    # Planar elements (one zero-size dimension) should still work
    element_2d = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[1, 0, 1],  # 2D element: x and z have non-zero size (valid)
        voltage_axis=0,
        network=RLC,
        name="2D_RLC",
    )
    assert element_2d.normal_axis == 1
    assert element_2d.size[element_2d.lateral_axis] != 0

    # Verify snapping behavior for planar elements
    snap_spec = element_2d._snapping_spec
    assert snap_spec.location[element_2d.lateral_axis] == SnapLocation.Center
    assert snap_spec.behavior[element_2d.lateral_axis] == SnapBehavior.Expand


@pytest.mark.parametrize("dist_type", ["off", "laterally_only", "on"])
@pytest.mark.parametrize("width", [1e-6, 5])
def test_distribution_variants(dist_type, width):
    """Test different distribution types with varying widths.

    Note: width=0 is no longer allowed (1D elements are not supported).
    Use a small finite width (e.g., 1e-6 mm) for narrow elements.
    """
    mm = 1e3
    RLC = td.RLCNetwork(
        resistance=50,
    )

    linear_element = td.LinearLumpedElement(
        center=[0.5 * mm, 0, 0],
        size=[1 * mm, 0, width * mm],
        voltage_axis=0,
        network=RLC,
        name="RLC",
        dist_type=dist_type,
    )
    x = np.linspace(0, 10 * mm, 25)
    y = np.linspace(-50 * mm, 50 * mm, 200)
    z = np.linspace(-20 * mm, 20 * mm, 100)
    coords = td.Coords(x=x, y=y, z=z)
    grid = td.Grid(boundaries=coords)

    # Grid is coarse enough that there is only one connection made along x
    geometry = linear_element.to_geometry(grid)
    structure = linear_element.to_PEC_connection(grid)
    if dist_type != "on":
        assert len(structure.geometry.geometries) == 1
    else:
        assert structure is None
    network = linear_element.to_structure(grid)
    _L, C = linear_element.estimate_parasitic_elements(grid)
    assert C >= 0

    # Grid is fine enough that there are two connections made along x
    x = np.linspace(0, 10 * mm, 100)
    grid = grid.updated_copy(path="boundaries", x=x)
    geometry = linear_element.to_geometry(grid)
    structure = linear_element.to_PEC_connection(grid)
    if dist_type != "on":
        assert len(structure.geometry.geometries) == 2
    else:
        assert structure is None
    network = linear_element.to_structure(grid)
    _L, C = linear_element.estimate_parasitic_elements(grid)
    assert C >= 0


def test_very_small_lateral_width_snapping():
    """Test that very small lateral width doesn't cause zero-size after snapping.

    When the user-specified lateral width is much smaller than the grid spacing,
    the snapping should auto-expand to at least one grid cell to avoid division
    by zero errors in the admittance scaling calculation.

    The bug occurs when the element center is exactly on a grid center and the
    lateral width is tiny - both bounds snap to the same grid center, resulting
    in zero size.
    """
    mm = 1e3  # 1 mm = 1000 um
    RLC = td.RLCNetwork(resistance=50)

    # Create grid first - with specific cell size
    cell_size = 1000
    grid_x = np.arange(-5000, 10000, cell_size)
    grid_y = np.arange(-5000, 5000, cell_size)
    grid_z = np.arange(-3000, 3000, cell_size)
    coords = td.Coords(x=grid_x, y=grid_y, z=grid_z)
    grid = td.Grid(boundaries=coords)

    # Grid centers along x are at -4500, -3500, ..., 1500, 2500, ...
    # Place element center exactly on a grid center (1500)
    port_width = 1e-5  # Much smaller than grid cell size
    element = td.LinearLumpedElement(
        center=[1500, 0, 0],
        size=[port_width, 0, 1 * mm],  # lateral_axis=0 has very small width
        voltage_axis=2,
        network=RLC,
        name="small_RLC",
    )

    # Before fix: this would fail with division by zero because
    # cell_box.size[lateral_axis] would be 0
    cell_box = element._create_box_for_network(grid)
    assert cell_box.size[element.lateral_axis] > 0, (
        "Cell box lateral size should be non-zero after snapping"
    )

    # Should not raise division by zero error
    structure = element.to_structure(grid)


# --- SPICE parser tests ---


def test_parse_spice_value_no_suffix():
    """Parse numeric values without scale suffix."""
    parse = td.CircuitImpedanceModel._parse_spice_value
    assert parse("50") == 50.0
    assert parse("1.5") == 1.5
    assert parse("1e-12") == 1e-12
    assert parse("+2.5") == 2.5
    assert parse("-3") == -3.0


@pytest.mark.parametrize(
    "s,expected",
    [
        ("1K", 1e3),
        ("1k", 1e3),
        ("2.5K", 2.5e3),
        ("1MEG", 1e6),
        ("1meg", 1e6),
        ("1M", 1e-3),
        ("10m", 0.01),
        ("10M", 0.01),
        ("1u", 1e-6),
        ("1U", 1e-6),
        ("1n", 1e-9),
        ("1p", 1e-12),
        ("1f", 1e-15),
        ("1G", 1e9),
        ("1g", 1e9),
        ("100T", 100e12),
        ("1T", 1e12),
    ],
)
def test_parse_spice_value_scale_suffixes(s, expected):
    """Parse values per SPICE convention: T (tera), G (giga), K (kilo), MEG (mega), m/M (milli), u, n, p, f."""
    assert td.CircuitImpedanceModel._parse_spice_value(s) == pytest.approx(expected)


def test_parse_spice_value_invalid():
    """Invalid value raises ``ValueError``; unknown scale suffix is ignored (scale 1.0)."""
    parse = td.CircuitImpedanceModel._parse_spice_value
    with pytest.raises(ValueError, match=r"Empty value"):
        parse("")
    with pytest.raises(ValueError, match=r"Cannot parse"):
        parse("abc")
    # Unknown scale prefix (e.g. "x") is not an error; value is parsed with scale 1.0
    assert parse("1x") == pytest.approx(1.0)


def test_parse_spice_value_trailing_unit_letters():
    """Scale prefix is matched; trailing unit letters (F, H, Ohm) are ignored."""
    parse = td.CircuitImpedanceModel._parse_spice_value
    assert parse("1pF") == pytest.approx(1e-12)
    assert parse("100nH") == pytest.approx(100e-9)
    assert parse("50Ohm") == pytest.approx(50.0)
    assert parse("2.5uF") == pytest.approx(2.5e-6)


def test_parse_spice_file_no_source_port_from_first_element(tmp_path):
    """With no voltage source, port is taken from the first element's nodes (``_parse_spice_file``)."""
    netlist = """
    Title: simple RC
    * simple RC
    R1 1 0 50
    C1 1 0 1p
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[0].element_type == "R"
    assert comp_list[0].node_plus == "1"
    assert comp_list[0].node_minus == "0"
    assert comp_list[0].value == 50.0
    assert comp_list[0].name == "R1"
    assert comp_list[1].element_type == "C"
    assert comp_list[1].value == 1e-12
    assert port_plus == "1"
    assert port_minus == "0"


def test_parse_spice_file_single_voltage_source_defines_port(tmp_path):
    """Single voltage source defines port nodes (``_parse_spice_file``)."""
    netlist = """
    Title: RC with V source
    V1 1 0 DC 0 AC 1
    R1 1 0 50
    C1 1 0 1p
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert port_plus == "1"
    assert port_minus == "0"


def test_parse_spice_file_comments_and_continuation(tmp_path):
    """Comment lines and + continuation are handled."""
    netlist = """
    Title
    $ comment
    * another comment
    R1 1 0 50
    C1 1 0
    + 1p
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    comp_list, _, _ = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[1].value == 1e-12


def test_parse_spice_file_scale_suffixes_in_netlist(tmp_path):
    """Scale suffixes ``K``, ``M``, ``m``, ``p`` in netlist are parsed."""
    netlist = """
    Title
    R1 1 0 2K
    C1 1 0 0.5p
    L1 2 1 10n
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    comp_list, _, _ = td.CircuitImpedanceModel._parse_spice_file(path)
    assert comp_list[0].value == 2000.0
    assert comp_list[1].value == 0.5e-12
    assert comp_list[2].value == 10e-9


def test_parse_spice_file_two_voltage_sources_raises(tmp_path):
    """At most one voltage source is allowed."""
    netlist = """
    Title
    V1 1 0 DC 0
    V2 2 0 DC 0
    R1 1 0 50
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    with pytest.raises(ValueError, match=r"at most one voltage source"):
        td.CircuitImpedanceModel._parse_spice_file(path)


def test_parse_spice_file_no_rlc_raises(tmp_path):
    """Netlist must contain at least one ``R``, ``C``, or ``L`` (``_parse_spice_file``)."""
    netlist = """
    Title
    V1 1 0 DC 0
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    with pytest.raises(ValueError, match=r"no R, C, or L"):
        td.CircuitImpedanceModel._parse_spice_file(path)


def test_parse_spice_file_component_line_too_short_raises(tmp_path):
    """Malformed component line raises."""
    netlist = """
    Title
    R1 1 0
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    with pytest.raises(ValueError, match=r"Component line must have name"):
        td.CircuitImpedanceModel._parse_spice_file(path)


def test_parse_spice_file_unsupported_lines_skipped_rest_parsed(tmp_path):
    """Unsupported lines (.MODEL, M, etc.) are skipped with a warning; R/C/L/V are still parsed."""
    netlist = """
    My RC circuit
    R1 1 0 50
    .MODEL nmos NMOS level=1
    C1 1 0 1p
    M1 1 2 3 4 nmos
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        comp_list, port_p, port_m = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[0].element_type == "R" and comp_list[0].value == 50.0
    assert comp_list[1].element_type == "C" and comp_list[1].value == 1e-12
    assert port_p == "1" and port_m == "0"
    assert mock_log.warning.call_count == 2
    msg_template = mock_log.warning.call_args_list[0][0][0]
    assert "skipped unrecognized" in msg_template


def test_parse_spice_file_mixed_ground_labels_warns(tmp_path):
    """Mixed ground-like node labels (0, gnd, GND) trigger a warning; each is a distinct node."""
    netlist = """
    Mixed ground
    R1 1 0 50
    C1 1 gnd 1p
    """
    path = tmp_path / "mixed_gnd.cir"
    path.write_text(netlist)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        comp_list, _port_p, _port_m = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    # Parser treats 0 and gnd as different nodes (three nodes: 1, 0, gnd)
    assert mock_log.warning.call_count >= 1
    warnings_text = " ".join(str(c) for c in mock_log.warning.call_args_list)
    assert "mixed ground" in warnings_text.lower() or "ground-like" in warnings_text.lower()


def test_parse_spice_file_unsupported_elements_only_rlc_used(tmp_path):
    """Regression: ideal/constraint/non-RLC netlist lines are skipped; only R/C/L define the circuit.

    Supported subset: flat netlists with R, C, L, and at most one V (for port). Unsupported
    (skipped with warning): .MODEL, .SUBCKT, M (MOSFET), ideal/constraint-style elements,
    and other SPICE constructs. This test ensures such lines do not break parsing and users
    are directed to the supported one-port RLC subset.
    """
    netlist = """
    Circuit with unsupported elements
    R1 1 0 50
    .MODEL myr res R=1
    C1 1 0 1p
    M1 2 3 0 0 nmos
    .SUBCKT block 1 0
    .ENDS
    """
    path = tmp_path / "unsupported.cir"
    path.write_text(netlist)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        comp_list, port_p, port_m = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[0].element_type == "R" and comp_list[1].element_type == "C"
    assert port_p == "1" and port_m == "0"
    # Should warn for each skipped line (only flat R/C/L/V supported)
    assert mock_log.warning.call_count >= 2
    first_msg = mock_log.warning.call_args_list[0][0][0].lower()
    assert "skipped" in first_msg or "unrecognized" in first_msg


def test_parse_spice_file_unsupported_lines_from_spice_succeeds(tmp_path):
    """from_spice(path, frequency_range) succeeds when netlist has unsupported lines; R/C/L are used."""
    netlist = """
    Netlist with subcircuits
    R1 1 0 50
    .SUBCKT block 1 2
    C1 1 0 1p
    .ENDS
    """
    path = tmp_path / "circuit.cir"
    path.write_text(netlist)
    frequency_range = (1e9, 5e9)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        model = td.CircuitImpedanceModel.from_spice(path, frequency_range, show_progress=False)
    assert len(model.components) == 2
    freqs_arr = np.linspace(frequency_range[0], frequency_range[1], 5)
    Y = model._get_effective_admittance(freqs_arr)
    assert len(Y) == len(freqs_arr)
    assert np.all(np.isfinite(Y))
    assert mock_log.warning.call_count >= 2


def test_component_value_must_be_positive():
    """LumpedCircuitComponent ``value`` must be positive (R, L, C); zero or negative raises ValidationError."""
    with pytest.raises(ValidationError):
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=0.0, name="R1"
        )
    with pytest.raises(ValidationError):
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=-50.0, name="R1"
        )
    with pytest.raises(ValidationError):
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=0.0, name="C1"
        )
    LumpedCircuitComponent(
        element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
    )  # ok


def test_component_nodes_must_be_distinct_and_non_empty():
    """LumpedCircuitComponent rejects self-loops (node_plus == node_minus) and empty node labels."""
    with pytest.raises(ValidationError, match=r"node_plus and node_minus must be distinct"):
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="1", value=50.0, name="R1"
        )
    with pytest.raises(ValidationError):
        LumpedCircuitComponent(
            element_type="R", node_plus="", node_minus="0", value=50.0, name="R1"
        )
    with pytest.raises(ValidationError):
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="", value=50.0, name="R1"
        )
    LumpedCircuitComponent(
        element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
    )  # ok


def test_parse_spice_file_zero_value_raises(tmp_path):
    """SPICE netlist with zero or negative component value raises when building ``LumpedCircuitComponent``."""
    for bad_value in ("0", "-1"):
        netlist = f"""
* circuit
R1 1 0 {bad_value}
"""
        path = tmp_path / "bad_value.cir"
        path.write_text(netlist)
        with pytest.raises(ValidationError):
            td.CircuitImpedanceModel._parse_spice_file(path)


def _effective_admittance_from_component_list(
    component_list: list[LumpedCircuitComponent],
    frequencies: np.ndarray,
    port_node_plus: str = "1",
    port_node_minus: str = "0",
) -> np.ndarray:
    """Compute one-port admittance from component list via CircuitImpedanceModel.

    Builds a model with the given components and freq_range=(min(frequencies), max(frequencies)),
    then evaluates and returns admittance at those same frequencies.
    """
    freqs = np.asarray(frequencies, dtype=float)
    f_min, f_max = float(np.min(freqs)), float(np.max(freqs))
    model = td.CircuitImpedanceModel(
        components=tuple(component_list),
        freq_range=(f_min, f_max),
        port_node_plus=port_node_plus,
        port_node_minus=port_node_minus,
    )
    return model._get_effective_admittance(np.asarray(frequencies, dtype=float))


def test_effective_admittance_parallel_rc(tmp_path):
    """LumpedCircuitComponent list, SPICE (with and without voltage source), and analytical agree for parallel RC.

    Circuit: ``R`` and ``C`` both between node 1 and 0. Port 1-0.
    Analytical: ``Y = 1/R + j*omega*C``.
    """
    R, C = 50.0, 1e-12
    component_list = [
        LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="0", value=R, name="R1"),
        LumpedCircuitComponent(element_type="C", node_plus="1", node_minus="0", value=C, name="C1"),
    ]
    freqs = np.linspace(0.2e9, 8e9, 30)
    port_plus, port_minus = "1", "0"
    omega = 2 * np.pi * freqs
    Y_analytical = 1.0 / R + 1j * omega * C

    Y_from_list = _effective_admittance_from_component_list(
        component_list, freqs, port_node_plus=port_plus, port_node_minus=port_minus
    )
    assert np.allclose(Y_from_list, Y_analytical, rtol=1e-14, atol=1e-20)

    # SPICE without voltage source: port from first element
    netlist = f"""
    Title
    R1 1 0 {R}
    C1 1 0 {C}
    """
    path = tmp_path / "parallel_rc.cir"
    path.write_text(netlist)
    comp_list_spice, port_plus_s, port_minus_s = td.CircuitImpedanceModel._parse_spice_file(path)
    assert port_plus_s == port_plus and port_minus_s == port_minus
    Y_from_spice = _effective_admittance_from_component_list(
        comp_list_spice, freqs, port_node_plus=port_plus_s, port_node_minus=port_minus_s
    )
    assert np.allclose(Y_from_list, Y_from_spice, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_spice, Y_analytical, rtol=1e-14, atol=1e-20)

    # SPICE with voltage source: port from V
    netlist_v = f"""
    Title
    V1 1 0 DC 0 AC 1
    R1 1 0 {R}
    C1 1 0 {C}
    """
    path_v = tmp_path / "with_v.cir"
    path_v.write_text(netlist_v)
    comp_list_spice_v, port_plus_v, port_minus_v = td.CircuitImpedanceModel._parse_spice_file(
        path_v
    )
    assert port_plus_v == "1" and port_minus_v == "0"
    Y_from_spice_v = _effective_admittance_from_component_list(
        comp_list_spice_v, freqs, port_node_plus=port_plus_v, port_node_minus=port_minus_v
    )
    assert np.allclose(Y_from_list, Y_from_spice_v, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_spice_v, Y_analytical, rtol=1e-14, atol=1e-20)


def test_effective_admittance_series_rc(tmp_path):
    """LumpedCircuitComponent list, SPICE, and analytical agree for series RC.

    Circuit: node 1 -- ``R`` -- node 2 -- ``C`` -- node 0. Port 1-0.
    Analytical: ``Y = 1/(R + 1/(j*omega*C)) = j*omega*C / (1 + j*omega*R*C)``.
    """
    R, C = 50.0, 1e-12
    component_list = [
        LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="2", value=R, name="R1"),
        LumpedCircuitComponent(element_type="C", node_plus="2", node_minus="0", value=C, name="C1"),
    ]
    freqs = np.linspace(0.2e9, 8e9, 30)
    port_plus, port_minus = "1", "0"
    omega = 2 * np.pi * freqs
    Y_analytical = (1j * omega * C) / (1.0 + 1j * omega * R * C)

    Y_from_list = _effective_admittance_from_component_list(
        component_list, freqs, port_node_plus=port_plus, port_node_minus=port_minus
    )
    assert np.allclose(Y_from_list, Y_analytical, rtol=1e-14, atol=1e-20)

    # SPICE: use voltage source to define port 1-0 (no single element has both 1 and 0 in series RC)
    netlist = f"""
    Title
    V1 1 0 DC 0 AC 1
    R1 1 2 {R}
    C1 2 0 {C}
    """
    path = tmp_path / "series_rc.cir"
    path.write_text(netlist)
    comp_list_spice, port_plus_s, port_minus_s = td.CircuitImpedanceModel._parse_spice_file(path)
    assert port_plus_s == "1" and port_minus_s == "0"
    Y_from_spice = _effective_admittance_from_component_list(
        comp_list_spice, freqs, port_node_plus=port_plus_s, port_node_minus=port_minus_s
    )
    assert np.allclose(Y_from_list, Y_from_spice, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_spice, Y_analytical, rtol=1e-14, atol=1e-20)


def test_effective_admittance_series_rcl(tmp_path):
    """LumpedCircuitComponent list, SPICE, and analytical agree for ``L`` in parallel with (``R`` in series ``C``).

    Circuit: ``L`` between 1-0, ``R`` between 1-2, ``C`` between 2-0. Port 1-0.
    Analytical: ``Y = 1/(j*omega*L) + 1/(R + 1/(j*omega*C))``.
    """
    R, C, L = 50.0, 1e-12, 2e-9
    component_list = [
        LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="2", value=R, name="R1"),
        LumpedCircuitComponent(element_type="C", node_plus="2", node_minus="0", value=C, name="C1"),
        LumpedCircuitComponent(element_type="L", node_plus="1", node_minus="0", value=L, name="L1"),
    ]
    freqs = np.linspace(0.2e9, 8e9, 30)
    port_plus, port_minus = "1", "0"
    omega = 2 * np.pi * freqs
    Y_L = 1.0 / (1j * omega * L)
    Z_RC = R + 1.0 / (1j * omega * C)
    Y_analytical = Y_L + 1.0 / Z_RC

    Y_from_list = _effective_admittance_from_component_list(
        component_list, freqs, port_node_plus=port_plus, port_node_minus=port_minus
    )
    assert np.allclose(Y_from_list, Y_analytical, rtol=1e-14, atol=1e-20)

    netlist = f"""
    Title
    L1 1 0 {L}
    R1 1 2 {R}
    C1 2 0 {C}
    """
    path = tmp_path / "series_rcl.cir"
    path.write_text(netlist)
    comp_list_spice, port_plus_s, port_minus_s = td.CircuitImpedanceModel._parse_spice_file(path)
    assert port_plus_s == "1" and port_minus_s == "0", "port from first element (L1 1 0)"
    Y_from_spice = _effective_admittance_from_component_list(
        comp_list_spice, freqs, port_node_plus=port_plus_s, port_node_minus=port_minus_s
    )
    assert np.allclose(Y_from_list, Y_from_spice, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_spice, Y_analytical, rtol=1e-14, atol=1e-20)


def test_effective_admittance_port_minus_not_ground():
    """Driving-point admittance with port_node_minus not ground matches analytical.

    Circuit: node 1 -- R -- node 2 -- C -- node 0 (series RC). Port 1-2 (reference = node 2).
    With node 2 as reference, the admittance at node 1 is 1/R (only R connects 1 to 2).
    """
    R, C = 50.0, 1e-12
    component_list = [
        LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="2", value=R, name="R1"),
        LumpedCircuitComponent(element_type="C", node_plus="2", node_minus="0", value=C, name="C1"),
    ]
    freqs = np.linspace(0.2e9, 8e9, 30)
    Y_analytical = np.full(len(freqs), 1.0 / R, dtype=complex)  # real 1/R

    Y_from_list = _effective_admittance_from_component_list(
        component_list, freqs, port_node_plus="1", port_node_minus="2"
    )
    assert np.allclose(Y_from_list, Y_analytical, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_list.real, 1.0 / R, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_from_list.imag, 0.0, rtol=1e-14, atol=1e-20)


def test_effective_admittance_port_nodes_same_raises():
    """Port nodes must be distinct; same node raises ValueError."""
    component_list = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
    ]
    freqs = np.linspace(1e9, 2e9, 5)
    with pytest.raises(ValueError, match=r"distinct|cannot be the same"):
        _effective_admittance_from_component_list(
            component_list, freqs, port_node_plus="1", port_node_minus="1"
        )
    with pytest.raises(ValueError, match=r"distinct|cannot be the same"):
        _effective_admittance_from_component_list(
            component_list, freqs, port_node_plus="0", port_node_minus="0"
        )


# --- Coverage: LumpedNodeMapper, LumpedCircuitComponent, SPICE parser, CircuitImpedanceModel ---


def test_nodemapper_indices_by_first_appearance():
    """Node indices are assigned in order of first appearance; no special treatment for any label."""
    mapper = LumpedNodeMapper()
    assert mapper.get_or_create_index("1") == 0
    assert mapper.get_or_create_index("2") == 1
    assert mapper.get_or_create_index("0") == 2
    assert mapper.total_nodes() == 3
    assert mapper.lookup_index("1") == 0
    assert mapper.lookup_index("2") == 1
    assert mapper.lookup_index("0") == 2
    # No ground alias normalization: "0", "GND", "gnd" are distinct nodes
    mapper2 = LumpedNodeMapper()
    assert mapper2.get_or_create_index("0") == 0
    assert mapper2.get_or_create_index("GND") == 1
    assert mapper2.get_or_create_index("gnd") == 2
    assert mapper2.total_nodes() == 3


def test_nodemapper_mixed_ground_labels_are_distinct_nodes():
    """Mixed reference labels (0 and GND) in the same circuit map to distinct nodes; consistent labeling is required."""
    mapper = LumpedNodeMapper()
    mapper.get_or_create_index("1")
    mapper.get_or_create_index("0")
    mapper.get_or_create_index("GND")
    assert mapper.total_nodes() == 3
    # So a netlist that uses both 0 and GND for the same physical ground would be wrong unless normalized by the user
    Y_mixed = _effective_admittance_from_component_list(
        [
            LumpedCircuitComponent(
                element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
            ),
            LumpedCircuitComponent(
                element_type="C", node_plus="1", node_minus="GND", value=1e-12, name="C1"
            ),
        ],
        np.linspace(1e9, 5e9, 5),
        port_node_plus="1",
        port_node_minus="0",
    )
    # Single-reference circuit (consistent "0")
    Y_single = _effective_admittance_from_component_list(
        [
            LumpedCircuitComponent(
                element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
            ),
            LumpedCircuitComponent(
                element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
            ),
        ],
        np.linspace(1e9, 5e9, 5),
        port_node_plus="1",
        port_node_minus="0",
    )
    # Mixed 0/GND is a different topology (three nodes, different Y), so admittances differ
    assert not np.allclose(Y_mixed, Y_single, rtol=1e-14, atol=1e-20)


def test_nodemapper_lowercase_gnd_same_admittance_as_zero():
    """Circuit with reference ``gnd`` yields same effective admittance as with ``0`` when each uses consistent labeling."""
    R, C = 50.0, 1e-12
    freqs = np.linspace(0.2e9, 8e9, 20)
    comp_list_zero = [
        LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="0", value=R, name="R1"),
        LumpedCircuitComponent(element_type="C", node_plus="1", node_minus="0", value=C, name="C1"),
    ]
    comp_list_gnd = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="gnd", value=R, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="gnd", value=C, name="C1"
        ),
    ]
    Y_zero = _effective_admittance_from_component_list(
        comp_list_zero, freqs, port_node_plus="1", port_node_minus="0"
    )
    Y_gnd = _effective_admittance_from_component_list(
        comp_list_gnd, freqs, port_node_plus="1", port_node_minus="gnd"
    )
    assert np.allclose(Y_zero, Y_gnd, rtol=1e-14, atol=1e-20)


def test_parse_spice_file_comment_title_first_line_components_not_skipped(tmp_path):
    """When the first line is a comment (e.g. ``* Title``), it is the title and skipped; next lines are components."""
    netlist = """* Title
R1 1 0 50
C1 1 0 1p
"""
    path = tmp_path / "comment_title.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[0].element_type == "R"
    assert comp_list[0].name == "R1"
    assert comp_list[0].value == 50.0
    assert comp_list[1].element_type == "C"
    assert comp_list[1].value == 1e-12
    assert port_plus == "1" and port_minus == "0"


def test_parse_spice_file_plain_title_first_line_skipped(tmp_path):
    """First non-empty line is always the SPICE title and is skipped (e.g. ``My circuit``)."""
    netlist = """My circuit
R1 1 0 50
"""
    path = tmp_path / "plain_title.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 1
    assert comp_list[0].element_type == "R"
    assert comp_list[0].name == "R1"
    assert port_plus == "1" and port_minus == "0"


def test_parse_spice_file_first_line_component_like_warns_and_skipped(tmp_path):
    """When the first line looks like R/C/L/V, it is still skipped as title and a warning is logged."""
    netlist = """R1 1 0 50
C1 1 0 1p
"""
    path = tmp_path / "no_title.cir"
    path.write_text(netlist)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    # First line R1 was treated as title and skipped; only C1 remains
    assert len(comp_list) == 1
    assert comp_list[0].element_type == "C"
    assert port_plus == "1" and port_minus == "0"
    assert mock_log.warning.call_count >= 1
    msg = mock_log.warning.call_args_list[0][0][0]
    assert "first line" in msg.lower() and "title" in msg.lower()


def test_parse_spice_file_lowercase_gnd_parsed(tmp_path):
    """SPICE netlist with consistent ``gnd`` label parses; admittance matches equivalent circuit using ``0``."""
    netlist = """* parallel RC
R1 1 gnd 50
C1 1 gnd 1p
"""
    path = tmp_path / "lowercase_gnd.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 2
    assert comp_list[0].node_minus == "gnd"
    assert comp_list[1].node_minus == "gnd"
    assert port_plus == "1" and port_minus == "gnd"
    # Same admittance as 1-0 circuit
    comp_list_zero = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
        ),
    ]
    freqs = np.linspace(0.2e9, 8e9, 15)
    Y_gnd = _effective_admittance_from_component_list(
        comp_list, freqs, port_node_plus="1", port_node_minus="gnd"
    )
    Y_zero = _effective_admittance_from_component_list(
        comp_list_zero, freqs, port_node_plus="1", port_node_minus="0"
    )
    assert np.allclose(Y_gnd, Y_zero, rtol=1e-14, atol=1e-20)


def test_parse_spice_file_voltage_source_between_1_and_2_infers_port(tmp_path):
    """SPICE with V between nodes 1 and 2 (no 0/GND) infers port; admittance matches component_list."""
    netlist = """Two-node circuit with V and R
V1 1 2 0
R1 1 2 50
"""
    path = tmp_path / "v_1_2.cir"
    path.write_text(netlist)
    comp_list, port_plus, port_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert port_plus == "1" and port_minus == "2"
    assert len(comp_list) == 1
    assert comp_list[0].node_plus == "1" and comp_list[0].node_minus == "2"
    freqs = np.linspace(0.2e9, 8e9, 15)
    Y_spice = _effective_admittance_from_component_list(
        comp_list, freqs, port_node_plus=port_plus, port_node_minus=port_minus
    )
    comp_list_direct = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="2", value=50.0, name="R1"
        ),
    ]
    Y_direct = _effective_admittance_from_component_list(
        comp_list_direct, freqs, port_node_plus="1", port_node_minus="2"
    )
    assert np.allclose(Y_spice, Y_direct, rtol=1e-14, atol=1e-20)
    assert np.allclose(Y_spice.real, 1.0 / 50.0, rtol=1e-14, atol=1e-20)


def test_lookup_index_raises_when_node_not_in_circuit():
    """CircuitImpedanceModel with port node not in the circuit raises at construction."""
    component_list = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
    ]
    freqs = np.linspace(1e9, 2e9, 5)
    with pytest.raises((ValueError, ValidationError), match=r"not a node in the circuit"):
        _effective_admittance_from_component_list(
            component_list, freqs, port_node_plus="99", port_node_minus="0"
        )
    with pytest.raises((ValueError, ValidationError), match=r"not a node in the circuit"):
        _effective_admittance_from_component_list(
            component_list, freqs, port_node_plus="1", port_node_minus="99"
        )


def test_component_default_name_set_in_post_init():
    """LumpedCircuitComponent with empty name gets a default name from validator."""
    comp = LumpedCircuitComponent(
        element_type="R", node_plus="1", node_minus="0", value=50.0, name=""
    )
    assert comp.name
    assert comp.name.startswith("R")


def test_admittance_network_warns_circuit_impedance_model_does_not():
    """CircuitImpedanceModel constructor does not emit the AdmittanceNetwork rename warning."""
    component_list = [
        td.LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
    ]
    freqs = np.linspace(1e9, 10e9, 5)
    with patch("tidy3d.components.lumped_element.log") as mock_log:
        _ = td.CircuitImpedanceModel(
            components=tuple(component_list),
            freq_range=(float(freqs[0]), float(freqs[-1])),
            fit_show_progress=False,
        )
        assert not any(
            "AdmittanceNetwork" in (c[0][0] if c[0] else "")
            for c in mock_log.warning.call_args_list
        )


def test_parse_spice_file_voltage_source_too_short_raises(tmp_path):
    """Voltage source line with fewer than two nodes raises."""
    netlist = """
    Title
    V1 1
    """
    path = tmp_path / "bad_v.cir"
    path.write_text(netlist)
    with pytest.raises(ValueError, match=r"Voltage source line needs at least two nodes"):
        td.CircuitImpedanceModel._parse_spice_file(path)


def test_parse_spice_file_not_a_file_raises(tmp_path):
    """_parse_spice_file raises FileNotFoundError when path is a directory or does not exist."""
    with pytest.raises(FileNotFoundError, match=r"not a file or does not exist"):
        td.CircuitImpedanceModel._parse_spice_file(tmp_path)  # directory
    with pytest.raises(FileNotFoundError, match=r"not a file or does not exist"):
        td.CircuitImpedanceModel._parse_spice_file(tmp_path / "nonexistent.cir")


def test_parse_spice_file_too_large_raises(tmp_path):
    """_parse_spice_file raises ValueError when file exceeds MAX_SPICE_FILE_SIZE_BYTES."""
    path = tmp_path / "huge.cir"
    path.write_text("Title\nR1 1 0 50\n")
    path.write_bytes(b"x" * (MAX_SPICE_FILE_SIZE_BYTES + 1))
    with pytest.raises(ValueError, match=r"exceeds maximum allowed"):
        td.CircuitImpedanceModel._parse_spice_file(path)


def test_circuit_impedance_model_disconnected_components_raises():
    """CircuitImpedanceModel with disconnected components raises at construction."""
    disconnected_comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="R", node_plus="2", node_minus="3", value=100.0, name="R2"
        ),
    )
    freqs = (1e9, 2e9)
    with pytest.raises(ValueError, match=r"disconnected"):
        td.CircuitImpedanceModel(
            components=disconnected_comps,
            freq_range=(freqs[0], freqs[1]),
        )


def test_circuit_impedance_model_port_nodes_same_raises():
    """CircuitImpedanceModel with port_node_plus == port_node_minus raises at construction."""
    comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
    )
    freqs = (1e9, 2e9)
    with pytest.raises((ValueError, ValidationError), match=r"distinct|cannot be the same"):
        td.CircuitImpedanceModel(
            components=comps,
            freq_range=(freqs[0], freqs[1]),
            port_node_plus="1",
            port_node_minus="1",
        )


def test_circuit_impedance_model_port_node_not_in_circuit_raises():
    """CircuitImpedanceModel with port nodes not in the component graph raises at construction."""
    comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
    )
    freqs = (1e9, 2e9)
    with pytest.raises(
        (ValueError, ValidationError), match=r"port_node_plus.*not a node|not.*endpoint"
    ):
        td.CircuitImpedanceModel(
            components=comps,
            freq_range=(freqs[0], freqs[1]),
            port_node_plus="2",
            port_node_minus="0",
        )
    with pytest.raises(
        (ValueError, ValidationError), match=r"port_node_minus.*not a node|not.*endpoint"
    ):
        td.CircuitImpedanceModel(
            components=comps,
            freq_range=(freqs[0], freqs[1]),
            port_node_plus="1",
            port_node_minus="2",
        )


def test_circuit_impedance_model_freq_range_optional():
    """CircuitImpedanceModel with freq_range=None works when frequency_range is passed to to_structure; _resolve_fit_freqs(None) raises."""
    comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
        ),
    )
    model = td.CircuitImpedanceModel(components=comps, freq_range=None)
    # CircuitImpedanceModel does not provide _as_admittance_function (only RLCNetwork/AdmittanceNetwork do)
    with pytest.raises(AttributeError, match=r"_as_admittance_function"):
        _ = model._as_admittance_function
    # admittance() on element uses _get_effective_admittance and works
    freqs = np.linspace(0.2e9, 8e9, 20)
    element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[2, 0, 3],
        voltage_axis=0,
        network=model,
        name="RC",
    )
    Y = element.admittance(freqs)
    assert len(Y) == len(freqs)
    assert np.all(np.isfinite(Y))
    # When freq_range is None and no frequency_range passed, _resolve_freq_range raises (used inside _to_medium)
    with pytest.raises(ValueError, match=r"freq_range"):
        model._resolve_fit_freqs(None)
    # to_structure with frequency_range works (use grid with multiple cells so snapped box has non-zero lateral size)
    grid = td.Grid(boundaries=td.Coords(x=[0, 1, 2], y=[0, 1], z=[0, 1, 2]))
    struct = element.to_structure(grid, frequency_range=(0.2e9, 8e9))
    assert struct is not None
    assert struct.geometry is not None


def test_terminal_modeler_injects_freq_range_for_circuit_impedance_model():
    """_inject_fit_freqs_into_lumped_elements sets freq_range=(min(freqs), max(freqs)) on CircuitImpedanceModel(freq_range=None); to_structures() then works without passing frequency_range."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import (
        _inject_fit_freqs_into_lumped_elements,
    )

    comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
        ),
    )
    model = td.CircuitImpedanceModel(components=comps, freq_range=None)
    element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[1, 0, 1],
        voltage_axis=0,
        network=model,
        name="RC",
    )
    freqs = (1e9, 2e9, 5e9)
    injected = _inject_fit_freqs_into_lumped_elements([element], freqs)
    assert len(injected) == 1
    assert injected[0].network.freq_range == (1e9, 5e9)
    # to_structures without frequency_range works because freq_range was injected
    grid = td.Grid(boundaries=td.Coords(x=[0, 1, 2], y=[0, 1], z=[0, 1, 2]))
    structures = injected[0].to_structures(grid)
    assert len(structures) >= 1


def test_get_effective_admittance_zero_or_negative_frequency_raises():
    """_get_effective_admittance with zero or negative frequency raises (inductor singular at DC)."""
    comps = (
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="L", node_plus="1", node_minus="0", value=1e-9, name="L1"
        ),
    )
    model = td.CircuitImpedanceModel(components=comps, freq_range=(1e9, 2e9))
    with pytest.raises(ValueError, match=r"positive|singular"):
        model._get_effective_admittance(np.array([0.0, 1e9]))
    with pytest.raises(ValueError, match=r"positive|singular"):
        model._get_effective_admittance(np.array([-1e9, 1e9]))


def test_circuit_impedance_model_constructor_returns_model():
    """CircuitImpedanceModel constructor with freq_range and default n_freqs builds a model; fit path (_get_effective_admittance at sampled freqs) is covered."""
    component_list = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
        ),
    ]
    freqs = np.linspace(0.2e9, 8e9, 20)
    model = td.CircuitImpedanceModel(
        components=tuple(component_list),
        freq_range=(float(freqs[0]), float(freqs[-1])),
        fit_show_progress=False,
    )
    Y = model._get_effective_admittance(np.array(freqs))
    assert len(Y) == len(freqs)
    assert np.all(np.isfinite(Y))


def test_circuit_impedance_model_serialization_roundtrip():
    """CircuitImpedanceModel round-trips through dict/JSON; components and freq_range are preserved."""
    component_list = [
        LumpedCircuitComponent(
            element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"
        ),
        LumpedCircuitComponent(
            element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"
        ),
    ]
    freqs = np.linspace(0.2e9, 8e9, 15)
    model = td.CircuitImpedanceModel(
        components=tuple(component_list),
        freq_range=(float(freqs[0]), float(freqs[-1])),
        fit_show_progress=False,
    )
    # Round-trip through dict
    data = model.model_dump()
    restored = td.CircuitImpedanceModel.model_validate(data)
    assert len(restored.components) == len(model.components)
    assert restored.port_node_plus == model.port_node_plus
    assert restored.port_node_minus == model.port_node_minus
    freqs_arr = np.array(freqs)
    Y1 = model._get_effective_admittance(freqs_arr)
    Y2 = restored._get_effective_admittance(freqs_arr)
    assert len(Y1) == len(Y2)
    np.testing.assert_allclose(Y1, Y2, rtol=1e-9, atol=1e-12)


def test_from_spice_port_override(tmp_path):
    """from_spice(path, frequency_range, port_node_plus=..., port_node_minus=...) uses the overrides and builds a model with that freq_range."""
    netlist = """
    Title
    R1 1 0 50
    C1 1 0 1p
    """
    path = tmp_path / "rc.cir"
    path.write_text(netlist)
    frequency_range = (0.2e9, 8e9)
    model = td.CircuitImpedanceModel.from_spice(
        path, frequency_range, port_node_plus="1", port_node_minus="0", show_progress=False
    )
    assert len(model.components) >= 1
    freqs = np.linspace(frequency_range[0], frequency_range[1], 15)
    Y = model._get_effective_admittance(freqs)
    assert len(Y) == len(freqs)
    assert np.all(np.isfinite(Y))


def test_from_touchstone_not_implemented():
    """from_touchstone raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match=r"not yet implemented"):
        td.CircuitImpedanceModel.from_touchstone("dummy.s1p")


def test_parse_spice_file_blank_line_in_body_skipped(tmp_path):
    """Blank lines (after strip) in the netlist are skipped; parsing still succeeds."""
    netlist = """
    Title

    R1 1 0 50
    """
    path = tmp_path / "with_blank.cir"
    path.write_text(netlist)
    comp_list, p_plus, p_minus = td.CircuitImpedanceModel._parse_spice_file(path)
    assert len(comp_list) == 1
    assert comp_list[0].element_type == "R"
    assert p_plus == "1" and p_minus == "0"
