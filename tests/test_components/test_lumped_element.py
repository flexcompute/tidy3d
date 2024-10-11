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
    _ = resistor.to_geometry
    _ = resistor.to_structure

    # Check conversion to mesh overrides
    _ = resistor.to_mesh_overrides()

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
    _ = resistor.to_geometry
    _ = resistor.to_structure

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
            circuit_topology="left",
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
        circuit_topology=topology,
    )

    linear_element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[2, 0, 3],
        voltage_axis=0,
        network=RLC,
        name="RLC",
    )
    # Check conversion to geometry and to structure
    _ = linear_element.to_geometry
    _ = linear_element.to_structure

    sf = linear_element._admittance_transfer_function_scaling
    med_RLC = RLC._to_medium(sf)
    (a, b) = RLC._to_admittance_function

    eps_from_RLC_med = med_RLC.eps_model(freqs)
    eps_direct = 1 + sf * NetworkConversions.complex_permittivity(a=a, b=b, freqs=freqs)

    assert np.allclose(eps_from_RLC_med, eps_direct, rtol=rtol)

    if configuration_includes_parallel_inductor:
        return

    network = td.AdmittanceNetwork(
        a=a,
        b=b,
    )

    med_network = network._to_medium(sf)
    # Check conversion to geometry and to structure
    linear_element = linear_element.updated_copy(
        network=network,
    )
    _ = linear_element.to_geometry
    _ = linear_element.to_structure
    assert np.allclose(med_RLC.eps_model(freqs), med_network.eps_model(freqs), rtol=rtol)
