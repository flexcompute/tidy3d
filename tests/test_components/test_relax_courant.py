"""Tests for the relax_courant simulation feature and its validators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components import simulation as simulation_module

from ..utils import assert_single_value_error_loc


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


def test_relax_courant_dt_does_not_check_client_license(monkeypatch):
    """Enterprise licensing for relax_courant is enforced by server metadata."""

    def fail_client_license_check(*_args, **_kwargs):
        raise AssertionError("relax_courant should not use the client-side extras license gate")

    fake_extras = SimpleNamespace(extension=SimpleNamespace(_relax_courant=lambda **_kwargs: 1.0))
    monkeypatch.setitem(simulation_module.tidy3d_extras, "mod", fake_extras)
    monkeypatch.setattr(
        simulation_module,
        "check_tidy3d_extras_licensed_feature",
        fail_client_license_check,
        raising=False,
    )

    sim = _make_relax_courant_sim()

    assert sim.requires_enterprise_license()
    assert sim.dt > 0


def test_relax_courant_rejects_internal_absorbers():
    """relax_courant should reject simulations with internal absorbers."""
    absorber = td.InternalAbsorber(
        center=(0, 0, 0),
        size=(0, 2, 2),
        direction="+",
        boundary_spec=td.ABCBoundary(permittivity=1),
    )
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(internal_absorbers=[absorber])
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), "Internal absorbers are not supported"
    )


def test_relax_courant_rejects_adiabatic_absorber():
    """relax_courant should reject adiabatic absorber boundaries."""
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary(plus=td.Absorber(), minus=td.Absorber()),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )
    assert_single_value_error_loc(excinfo, ("relax_courant",), "Adiabatic absorber boundary")


def test_relax_courant_rejects_tfsf():
    """relax_courant should reject TFSF sources."""
    tfsf = td.TFSF(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        size=(1, 1, 1),
        direction="+",
        injection_axis=0,
    )
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(sources=[tfsf])
    assert_single_value_error_loc(excinfo, ("relax_courant",), "TFSF source")


def test_relax_courant_rejects_fixed_angle_planewave():
    """relax_courant should reject fixed-angle PlaneWave sources."""
    pw = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        size=(0, td.inf, td.inf),
        direction="+",
        angular_spec=td.FixedAngleSpec(),
    )
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(sources=[pw])
    assert_single_value_error_loc(excinfo, ("relax_courant",), "Fixed-angle PlaneWave source")


def test_relax_courant_rejects_fully_anisotropic():
    """relax_courant should reject fully anisotropic mediums."""
    aniso = td.FullyAnisotropicMedium(
        permittivity=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        conductivity=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=aniso)
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(structures=[struct])
    assert_single_value_error_loc(excinfo, ("relax_courant",), "FullyAnisotropicMedium")


def test_relax_courant_rejects_nonlinear():
    """relax_courant should reject nonlinear mediums."""
    nl_medium = td.Medium(
        permittivity=4.0,
        nonlinear_spec=td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=1e-19)]),
    )
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=nl_medium)
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(structures=[struct])
    assert_single_value_error_loc(excinfo, ("relax_courant",), "nonlinear medium")


def test_relax_courant_rejects_time_modulated():
    """relax_courant should reject time-modulated mediums."""
    mod_spec = td.ModulationSpec(
        permittivity=td.SpaceTimeModulation(
            time_modulation=td.ContinuousWaveTimeModulation(freq0=1e12, amplitude=0.01),
        ),
    )
    mod_medium = td.Medium(permittivity=4.0, modulation_spec=mod_spec)
    struct = td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)), medium=mod_medium)
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(structures=[struct])
    assert_single_value_error_loc(excinfo, ("relax_courant",), "time-modulated medium")


def test_relax_courant_rejects_periodic_x():
    """relax_courant should reject periodic boundary conditions along x."""
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.periodic(),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), "Periodic or Bloch boundary condition along x"
    )


def test_relax_courant_rejects_bloch_x():
    """relax_courant should reject Bloch boundary conditions along x."""
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.bloch(bloch_vec=1.0),
                y=td.Boundary.pml(),
                z=td.Boundary.pml(),
            )
        )
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), "Periodic or Bloch boundary condition along x"
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_relax_courant_rejects_abc(axis):
    """relax_courant should reject ABCBoundary on any axis."""
    abc = td.Boundary(plus=td.ABCBoundary(), minus=td.ABCBoundary())
    boundaries = {"x": td.Boundary.pml(), "y": td.Boundary.pml(), "z": td.Boundary.pml(), axis: abc}
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(boundary_spec=td.BoundarySpec(**boundaries))
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), f"ABC or ModeABC boundary condition along {axis}"
    )


def test_relax_courant_rejects_2d_sim():
    """relax_courant should reject simulations with a zero-size (collapsed) dimension."""
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(size=(0, 2.0, 2.0))
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), "Zero-size (collapsed) simulation dimensions"
    )


def test_relax_courant_rejects_single_cell_axis():
    """relax_courant should reject quasi-2D simulations even when all sizes are
    positive. ``self.grid.num_cells`` includes PML padding, so triggering the
    ``num_cells <= 1`` guard without a zero size requires both a coarse x-axis
    grid (dl > size) and non-PML x-boundaries that don't inflate the cell count.
    Pins this branch separately from the older zero-size check.
    """
    grid_spec = td.GridSpec(
        grid_x=td.UniformGrid(dl=3.0),  # > size to force num_cells == 1
        grid_y=td.UniformGrid(dl=0.1),
        grid_z=td.UniformGrid(dl=0.1),
    )
    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PECBoundary(), minus=td.PECBoundary()),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )
    with pytest.raises(ValidationError) as excinfo:
        _make_relax_courant_sim(grid_spec=grid_spec, boundary_spec=boundary_spec)
    assert_single_value_error_loc(
        excinfo, ("relax_courant",), "Single-cell x-axis (quasi-2D simulation)"
    )
