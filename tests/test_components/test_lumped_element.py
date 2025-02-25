"""Tests lumped elements."""

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.lumped_element import NetworkConversions


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

    # Check conversion to monitor
    freqs = np.linspace(1e9, 50e9, 101)
    monitor = resistor.to_monitor(freqs=freqs)
    assert monitor.name == resistor.monitor_name

    # error if voltage axis is not in plane with the resistor
    with pytest.raises(pydantic.ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[2, 0, 3],
            voltage_axis=1,
            name="R",
        )

    # error if not planar
    with pytest.raises(pydantic.ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[0, 0, 3],
            voltage_axis=2,
            name="R",
        )
    with pytest.raises(pydantic.ValidationError):
        _ = td.LumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            size=[2, 1, 3],
            voltage_axis=2,
            name="R",
        )


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

    # error if inner diameter is larger
    with pytest.raises(pydantic.ValidationError):
        _ = td.CoaxialLumpedResistor(
            resistance=50.0,
            center=[0, 0, 0],
            outer_diameter=3,
            inner_diameter=4,
            normal_axis=1,
            name="R",
        )

    with pytest.raises(pydantic.ValidationError):
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.RLCNetwork()

    # Must have a valid topology
    with pytest.raises(pydantic.ValidationError):
        _ = td.RLCNetwork(
            capacitance=0.2e-12,
            network_topology="left",
        )


def test_validators_admittance_network():
    """Test that ``AdmittanceNetwork`` is validated correctly."""
    with pytest.raises(pydantic.ValidationError):
        _ = td.AdmittanceNetwork()

    a = (0, -1, 2)
    b = (1, 1, 2)
    # non negative a and b
    with pytest.raises(pydantic.ValidationError):
        _ = td.AdmittanceNetwork(
            a=a,
            b=b,
        )

    a = (0, complex(1, 2), 2)
    b = (1, 1, 2)
    # real a and b
    with pytest.raises(pydantic.ValidationError):
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
    eps_direct = 1 + sf * NetworkConversions.complex_permittivity(a=a, b=b, freqs=freqs)

    assert np.allclose(eps_from_RLC_med, eps_direct, rtol=rtol)

    if configuration_includes_parallel_inductor:
        return

    network = td.AdmittanceNetwork(
        a=a,
        b=b,
    )

    (a, b) = network._as_admittance_function
    med_network = network._to_medium(sf)
    # Check conversion to geometry and to structure
    linear_element = linear_element.updated_copy(
        network=network,
    )
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


@pytest.mark.parametrize("dist_type", ["off", "laterally_only", "on"])
@pytest.mark.parametrize("width", [0, 5])
def test_distribution_variants(dist_type, width):
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
    L, C = linear_element.estimate_parasitic_elements(grid)
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
    L, C = linear_element.estimate_parasitic_elements(grid)
    assert C >= 0
