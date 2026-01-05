"""Tests boundary conditions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.boundary import (
    MIN_NUM_ABSORBER_LAYERS,
    MIN_NUM_PML_LAYERS,
    MIN_NUM_STABLE_PML_LAYERS,
    PML,
    Absorber,
    BlochBoundary,
    Boundary,
    BoundarySpec,
    PECBoundary,
    Periodic,
    PMCBoundary,
    StablePML,
)
from tidy3d.components.source.current import PointDipole
from tidy3d.components.source.field import PlaneWave
from tidy3d.components.source.time import GaussianPulse
from tidy3d.exceptions import DataError, SetupError

from ..utils import AssertLogLevel


def test_bloch_phase():
    bb = BlochBoundary(bloch_vec=1.0)
    _ = bb.bloch_phase


@pytest.mark.parametrize("dimension", ["x", "y", "z"])
def test_getitem(dimension):
    spec = BoundarySpec.pml(y=True, z=True)
    _ = spec[dimension]


def test_getitem_not_a_dim():
    spec = BoundarySpec.pml(y=True, z=True)
    with pytest.raises(DataError):
        _ = spec["NOT_A_DIMENSION"]


@pytest.mark.parametrize("plane_wave_dir", ["+", "-"])
def test_boundaryedge_types(plane_wave_dir):
    """Test that each type of boundary condition can be defined."""
    _ = Periodic()
    _ = PECBoundary()
    _ = PMCBoundary()

    _ = BlochBoundary(bloch_vec=1)
    pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    source = PlaneWave(
        size=(td.inf, td.inf, 0),
        source_time=pulse,
        direction=plane_wave_dir,
        angle_theta=1.5,
        angle_phi=0.3,
    )
    _ = BlochBoundary.from_source(source=source, domain_size=5, axis=0)

    # Bloch boundaries should raise errors if incorrectly defined
    with pytest.raises(SetupError):
        _ = BlochBoundary.from_source(source=source, domain_size=5, axis=2)
    with pytest.raises(SetupError):
        pt_dipole = PointDipole(center=(1, 2, 3), source_time=pulse, polarization="Ex")
        _ = BlochBoundary.from_source(source=pt_dipole, domain_size=5, axis=0)

    _ = PML(num_layers=10)
    _ = StablePML(num_layers=40)
    _ = Absorber(num_layers=40)


def test_boundary_validators():
    """Test the validators in class ``Boundary``"""

    bloch = BlochBoundary(bloch_vec=1)
    pec = PECBoundary()
    pml = PML(num_layers=10)
    periodic = Periodic()

    # test `bloch_on_both_sides`
    with pytest.raises(ValidationError):
        _ = Boundary(plus=bloch, minus=pec)

    # test `periodic_with_pml`
    with pytest.raises(ValidationError):
        _ = Boundary(plus=periodic, minus=pml)


@pytest.mark.parametrize("boundary, log_level", [(PMCBoundary(), None), (Periodic(), "WARNING")])
def test_boundary_validator_warnings(boundary, log_level):
    """Test the validators in class ``Boundary`` which should show a warning but not an error"""
    with AssertLogLevel(log_level):
        _ = Boundary(plus=PECBoundary(), minus=boundary)


@pytest.mark.parametrize("boundary, log_level", [(PMCBoundary(), None), (Periodic(), "WARNING")])
def test_boundary_validator_warnings_switched(boundary, log_level):
    """Test the validators in class ``Boundary`` which should show a warning but not an error"""
    with AssertLogLevel(log_level):
        _ = Boundary(minus=PECBoundary(), plus=boundary)


def test_boundary():
    """Test that the various classmethods and combinations for ``Boundary`` to work correctly."""

    # periodic
    boundary = Boundary.periodic()
    assert isinstance(boundary.plus, Periodic) and isinstance(boundary.minus, Periodic)

    # pec
    boundary = Boundary.pec()
    assert isinstance(boundary.plus, PECBoundary) and isinstance(boundary.minus, PECBoundary)

    # pmc
    boundary = Boundary.pmc()
    assert isinstance(boundary.plus, PMCBoundary) and isinstance(boundary.minus, PMCBoundary)

    # bloch
    boundary = Boundary.bloch(bloch_vec=1)
    assert isinstance(boundary.plus, BlochBoundary) and isinstance(boundary.minus, BlochBoundary)

    # bloch from source
    pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    source = PlaneWave(
        size=(td.inf, td.inf, 0), source_time=pulse, direction="+", angle_theta=1.5, angle_phi=0.3
    )
    boundary = Boundary.bloch_from_source(source=source, domain_size=5, axis=0)
    assert isinstance(boundary.plus, BlochBoundary) and isinstance(boundary.minus, BlochBoundary)

    # pml and related
    boundary = Boundary.pml()
    assert isinstance(boundary.plus, PML) and isinstance(boundary.minus, PML)
    boundary = Boundary.stable_pml()
    assert isinstance(boundary.plus, StablePML) and isinstance(boundary.minus, StablePML)
    boundary = Boundary.absorber()
    assert isinstance(boundary.plus, Absorber) and isinstance(boundary.minus, Absorber)

    # combinations
    boundary = Boundary(plus=PECBoundary(), minus=PML())
    assert isinstance(boundary.plus, PECBoundary) and isinstance(boundary.minus, PML)


def test_boundaryspec_classmethods():
    """Test that the classmethods for ``BoundarySpec`` work correctly."""

    # pml
    boundary_spec = BoundarySpec.pml(x=False, y=True, z=True)
    boundaries = boundary_spec.to_list
    assert (
        isinstance(boundaries[0][0], Periodic)
        and isinstance(boundaries[0][1], Periodic)
        and isinstance(boundaries[1][0], PML)
        and isinstance(boundaries[1][1], PML)
        and isinstance(boundaries[2][0], PML)
        and isinstance(boundaries[2][1], PML)
    )

    # pec
    boundary_spec = BoundarySpec.pec(x=True, z=True)
    boundaries = boundary_spec.to_list
    assert (
        isinstance(boundaries[0][0], PECBoundary)
        and isinstance(boundaries[0][1], PECBoundary)
        and isinstance(boundaries[1][0], PML)
        and isinstance(boundaries[1][1], PML)
        and isinstance(boundaries[2][0], PECBoundary)
        and isinstance(boundaries[2][1], PECBoundary)
    )

    # pmc
    boundary_spec = BoundarySpec.pmc(y=True)
    boundaries = boundary_spec.to_list
    assert (
        isinstance(boundaries[0][0], PML)
        and isinstance(boundaries[0][1], PML)
        and isinstance(boundaries[1][0], PMCBoundary)
        and isinstance(boundaries[1][1], PMCBoundary)
        and isinstance(boundaries[2][0], PML)
        and isinstance(boundaries[2][1], PML)
    )

    # all_sides
    boundary_spec = BoundarySpec.all_sides(boundary=PML())
    boundaries = boundary_spec.to_list
    assert all(
        isinstance(boundary, PML) for boundary_dim in boundaries for boundary in boundary_dim
    )


def test_extrude_structures_to_pml():
    """Test ``extrude_structures`` field in PML/Absorber API."""

    # check default state
    boundary_pml = PML()
    boundary_abs = Absorber()
    boundary_bloch = BlochBoundary(bloch_vec=1)
    boundary_pec = PECBoundary()

    assert boundary_pml.extrude_structures is True
    assert boundary_abs.extrude_structures is False

    # make sure attribute error is raised if other BC attempt to access/use the feature
    with pytest.raises(AttributeError):
        boundary_bloch.extrude_structures
    with pytest.raises(AttributeError):
        boundary_pec.extrude_structures

    # change state of boundary condition
    boundary_pml = PML(extrude_structures=False)
    boundary_abs = Absorber(extrude_structures=True)

    # make sure field values were correctly updated
    assert boundary_pml.extrude_structures is False
    assert boundary_abs.extrude_structures is True


@pytest.mark.parametrize("absorber_type", [PML, StablePML, Absorber])
def test_num_layers_validator(absorber_type):
    """Test the Field validators that enforce ``num_layers>0``."""
    with pytest.raises(ValidationError):
        _ = absorber_type(num_layers=0)


@pytest.mark.parametrize(
    "absorber_type, num_layers",
    [
        (PML, MIN_NUM_PML_LAYERS),
        (StablePML, MIN_NUM_STABLE_PML_LAYERS),
        (Absorber, MIN_NUM_ABSORBER_LAYERS),
    ],
)
def test_num_layers_validator_warning(absorber_type, num_layers):
    """Test the validators in ``PML`` which should display a warning not an error."""
    with AssertLogLevel(None):
        _ = absorber_type(num_layers=num_layers)
    with AssertLogLevel("WARNING"):
        _ = absorber_type(num_layers=num_layers - 1)
