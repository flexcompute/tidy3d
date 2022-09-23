"""Tests GridSpec."""
import pytest
import numpy as np

import tidy3d as td
from tidy3d.log import SetupError


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
        sources=[src],
        num_pml_layers=((10, 10), (0, 5), (0, 0)),
    )
