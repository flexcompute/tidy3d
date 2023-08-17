"""Tests boundary conditions."""

import pytest
import pydantic.v1 as pydantic

import tidy3d as td
from tidy3d.components.boundary import BoundarySpec, Boundary
from tidy3d.components.boundary import Periodic, PECBoundary, PMCBoundary, BlochBoundary
from tidy3d.components.boundary import PML, StablePML, Absorber
from tidy3d.components.source import GaussianPulse, PlaneWave, PointDipole
from tidy3d.exceptions import SetupError, DataError
from ..utils import assert_log_level, log_capture


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
    """Test the validators in class `Boundary`"""

    bloch = BlochBoundary(bloch_vec=1)
    pec = PECBoundary()
    pml = PML(num_layers=10)
    periodic = Periodic()

    # test `bloch_on_both_sides`
    with pytest.raises(pydantic.ValidationError):
        _ = Boundary(plus=bloch, minus=pec)

    # test `periodic_with_pml`
    with pytest.raises(pydantic.ValidationError):
        _ = Boundary(plus=periodic, minus=pml)


@pytest.mark.parametrize("boundary, log_level", [(PMCBoundary(), None), (Periodic(), "WARNING")])
def test_boundary_validator_warnings(log_capture, boundary, log_level):
    """Test the validators in class `Boundary` which should show a warning but not an error"""
    boundary = Boundary(plus=PECBoundary(), minus=boundary)
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("boundary, log_level", [(PMCBoundary(), None), (Periodic(), "WARNING")])
def test_boundary_validator_warnings_switched(log_capture, boundary, log_level):
    """Test the validators in class `Boundary` which should show a warning but not an error"""
    boundary = Boundary(minus=PECBoundary(), plus=boundary)
    assert_log_level(log_capture, log_level)


def test_boundary():
    """Test that the various classmethods and combinations for Boundary work correctly."""

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
    """Test that the classmethods for BoundarySpec work correctly."""

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
        [isinstance(boundary, PML) for boundary_dim in boundaries for boundary in boundary_dim]
    )
