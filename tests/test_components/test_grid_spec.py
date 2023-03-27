"""Tests GridSpec."""
import pytest
import numpy as np

import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError


def make_grid_spec():
    return td.GridSpec(wavelength=1.0)


def test_add_pml_to_bounds():
    gs = make_grid_spec()
    bounds = np.array([1.0])
    cs = gs.grid_x._add_pml_to_bounds(3, bounds=bounds)
    assert np.all(cs == bounds)


def test_make_coords():
    gs = make_grid_spec()
    cs = gs.grid_x.make_coords(
        axis=0,
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(2, 0.3, 1)), medium=td.Medium(permittivity=2)),
        ],
        symmetry=(1, 0, -1),
        periodic=(True, False, False),
        wavelength=1.0,
        num_pml_layers=(10, 4),
    )


def test_make_coords_2d():
    gs = make_grid_spec()
    cs = gs.grid_x.make_coords(
        axis=1,
        structures=[
            td.Structure(geometry=td.Box(size=(1, 0, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(2, 0, 1)), medium=td.Medium(permittivity=2)),
        ],
        symmetry=(1, 0, -1),
        periodic=(True, True, False),
        wavelength=1.0,
        num_pml_layers=(10, 4),
    )


def test_wvl_from_sources():

    # no sources
    with pytest.raises(SetupError):
        td.GridSpec.wavelength_from_sources(sources=[])

    freqs = [2e14, 3e14]
    sources = [
        td.PointDipole(source_time=td.GaussianPulse(freq0=f0, fwidth=1e14), polarization="Ex")
        for f0 in freqs
    ]

    # sources at different frequencies
    with pytest.raises(SetupError):
        td.GridSpec.wavelength_from_sources(sources=sources)

    # sources at same frequency
    freq0 = 2e14
    sources = [
        td.PointDipole(source_time=td.GaussianPulse(freq0=freq0, fwidth=1e14), polarization="Ex")
        for _ in range(4)
    ]
    wvl = td.GridSpec.wavelength_from_sources(sources=sources)
    assert np.isclose(wvl, td.C_0 / freq0), "wavelength did not match source central wavelengths."


def test_auto_grid_from_sources():
    src = td.PointDipole(source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14), polarization="Ex")
    grid_spec = td.GridSpec.auto()
    assert grid_spec.wavelength is None
    assert grid_spec.auto_grid_used
    grid_spec.make_grid(
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium()),
        ],
        symmetry=(0, 1, -1),
        periodic=(False, False, True),
        sources=[src],
        num_pml_layers=((10, 10), (0, 5), (0, 0)),
    )


RTOL = 0.01


def test_autogrid_2dmaterials():
    sigma = 0.45
    thickness = 0.01
    medium = td.Medium2D.from_medium(td.Medium(conductivity=sigma), thickness=thickness)
    grid_dl = 0.03
    box = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 0), center=(0, 0, 1)), medium=medium)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=1.5e14, fwidth=0.5e14),
        size=(0, 0, 0),
        polarization="Ex",
    )
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=[box],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
        ),
        grid_spec=td.GridSpec.auto(),
        run_time=1e-12,
    )
    assert np.isclose(sim.volumetric_structures[0].geometry.center[2], 1, rtol=RTOL)
    grid_dl = sim.discretize(box.geometry).sizes.z[0]
    assert np.isclose(sim.volumetric_structures[0].geometry.size[2], grid_dl, rtol=RTOL)

    # now if we increase conductivity, the in-plane grid size should decrease
    sigma2 = 4.5
    medium2 = td.Medium2D.from_medium(td.Medium(conductivity=sigma2), thickness=thickness)
    box2 = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 0), center=(0, 0, 1)), medium=medium2)

    sim2 = td.Simulation(
        size=(10, 10, 10),
        structures=[box2],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
        ),
        grid_spec=td.GridSpec.auto(),
        run_time=1e-12,
    )
    grid_dl1_inplane = sim.discretize(box.geometry).sizes.x[0]
    grid_dl2_inplane = sim2.discretize(box2.geometry).sizes.x[0]
    # This is commented out until inplane AutoGrid for 2D materials is enabled
    # assert grid_dl1_inplane > grid_dl2_inplane

    # should error if two 2d materials have different normals and both autogrid
    box2 = td.Structure(geometry=td.Box(size=(td.inf, 0, td.inf), center=(0, 0, 1)), medium=medium)
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=[box, box2],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
        ),
        grid_spec=td.GridSpec.auto(),
        run_time=1e-12,
    )

    # Commented until inplane AutoGrid for 2D materials is enabled
    # with pytest.raises(ValidationError):
    #    _ = sim.grid
