"""Tests for the relax_courant simulation feature and its validators."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

import tidy3d as td


def _make_relax_courant_sim(**kwargs):
    """Helper to build a minimal simulation with relax_courant and PML along x."""
    defaults = {
        "size": (2.0, 2.0, 2.0),
        "run_time": 1e-12,
        "grid_spec": td.GridSpec.uniform(dl=0.1),
        "sources": [
            td.UniformCurrentSource(
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
                size=(0, 0, 0),
                polarization="Ex",
            )
        ],
        "relax_courant": True,
    }
    defaults.update(kwargs)
    return td.Simulation(**defaults)


def test_relax_courant_valid():
    """A basic simulation with relax_courant=True should be accepted."""
    sim = _make_relax_courant_sim()
    assert sim.relax_courant is True


def test_relax_courant_dt_increased():
    """dt with relax_courant=True should be strictly larger than without."""
    has_tidy3d_extras = importlib.util.find_spec("tidy3d_extras") is not None
    sim_normal = _make_relax_courant_sim(relax_courant=False)
    sim_relaxed = _make_relax_courant_sim(relax_courant=True)
    if has_tidy3d_extras:
        assert sim_relaxed.dt > sim_normal.dt
    else:
        with pytest.raises(ImportError):
            _ = sim_relaxed.dt


def test_relax_courant_rejects_internal_absorbers():
    """relax_courant should reject simulations with internal absorbers."""
    absorber = td.InternalAbsorber(
        center=(0, 0, 0),
        size=(0, 2, 2),
        direction="+",
        boundary_spec=td.ABCBoundary(permittivity=1),
    )
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(internal_absorbers=[absorber])


def test_relax_courant_rejects_adiabatic_absorber():
    """relax_courant should reject adiabatic absorber boundaries."""
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary(plus=td.Absorber(), minus=td.Absorber()),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )


def test_relax_courant_rejects_tfsf():
    """relax_courant should reject TFSF sources."""
    tfsf = td.TFSF(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        size=(1, 1, 1),
        direction="+",
        injection_axis=0,
    )
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(sources=[tfsf])


def test_relax_courant_rejects_fixed_angle_planewave():
    """relax_courant should reject fixed-angle PlaneWave sources."""
    pw = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        size=(0, td.inf, td.inf),
        direction="+",
        angular_spec=td.FixedAngleSpec(),
    )
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(sources=[pw])


def test_relax_courant_rejects_fully_anisotropic():
    """relax_courant should reject fully anisotropic mediums."""
    aniso = td.FullyAnisotropicMedium(
        permittivity=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        conductivity=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=aniso)
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(structures=[struct])


def test_relax_courant_rejects_nonlinear():
    """relax_courant should reject nonlinear mediums."""
    nl_medium = td.Medium(
        permittivity=4.0,
        nonlinear_spec=td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=1e-19)]),
    )
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=nl_medium)
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(structures=[struct])


def test_relax_courant_rejects_time_modulated():
    """relax_courant should reject time-modulated mediums."""
    mod_spec = td.ModulationSpec(
        permittivity=td.SpaceTimeModulation(
            time_modulation=td.ContinuousWaveTimeModulation(freq0=1e12, amplitude=0.01),
        ),
    )
    mod_medium = td.Medium(permittivity=4.0, modulation_spec=mod_spec)
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=mod_medium)
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(structures=[struct])


def test_relax_courant_rejects_periodic_x():
    """relax_courant should reject periodic boundary conditions along x."""
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.periodic(),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )


def test_relax_courant_rejects_bloch_x():
    """relax_courant should reject Bloch boundary conditions along x."""
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.bloch(bloch_vec=1.0),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )


def test_relax_courant_rejects_2d_sim():
    """relax_courant should reject simulations with a zero-size (collapsed) dimension."""
    with pytest.raises(ValidationError):
        _make_relax_courant_sim(size=(0, 2.0, 2.0))
