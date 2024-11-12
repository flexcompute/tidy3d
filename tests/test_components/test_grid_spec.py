"""Tests GridSpec."""

import numpy as np
import pytest
import tidy3d as td
from tidy3d.exceptions import SetupError


def make_grid_spec():
    return td.GridSpec(wavelength=1.0)


def test_add_pml_to_bounds():
    gs = make_grid_spec()
    bounds = np.array([1.0])
    cs = gs.grid_x._add_pml_to_bounds(3, bounds=bounds)
    assert np.all(cs == bounds)


def test_make_coords():
    gs = make_grid_spec()
    _ = gs.grid_x.make_coords(
        axis=0,
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(2, 0.3, 1)), medium=td.Medium(permittivity=2)),
        ],
        symmetry=(1, 0, -1),
        periodic=(True, False, False),
        wavelength=1.0,
        num_pml_layers=(10, 4),
        snapping_points=(),
    )


def test_make_coords_with_snapping_points():
    """Test the behavior of snapping points"""
    gs = make_grid_spec()
    make_coords_args = dict(
        structures=[
            td.Structure(geometry=td.Box(size=(2, 2, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=4)),
        ],
        symmetry=(0, 0, 0),
        periodic=(False, False, False),
        wavelength=1.0,
        num_pml_layers=(0, 0),
        axis=0,
    )

    # 1) no snapping points, 0.85 is not on any grid boundary
    coord_original = gs.grid_x.make_coords(
        snapping_points=(),
        **make_coords_args,
    )
    assert not np.any(np.isclose(coord_original, 0.85))

    # 2) with snapping points at 0.85, grid should pass through 0.85
    coord = gs.grid_x.make_coords(
        snapping_points=((0.85, 0, 0),),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, 0.85))

    #  snapping still takes effect if the point is completely outside along other axes
    coord = gs.grid_x.make_coords(
        snapping_points=((0.85, 10, 0),),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, 0.85))

    coord = gs.grid_x.make_coords(
        snapping_points=((0.85, 0, -10),),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, 0.85))

    # 3) snapping takes no effect if it's too close to interval boundaries
    coord = gs.grid_x.make_coords(
        snapping_points=((0.98, 0, 0),),
        **make_coords_args,
    )
    assert np.allclose(coord_original, coord)

    # and no snapping if it's compeletely outside the simulation domain
    coord = gs.grid_x.make_coords(
        snapping_points=((10, 0, 0),),
        **make_coords_args,
    )
    assert np.allclose(coord_original, coord)

    coord = gs.grid_x.make_coords(
        snapping_points=((-10, 0, 0),),
        **make_coords_args,
    )
    assert np.allclose(coord_original, coord)


def test_make_coords_2d():
    gs = make_grid_spec()
    _ = gs.grid_x.make_coords(
        axis=1,
        structures=[
            td.Structure(geometry=td.Box(size=(1, 0, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(2, 0, 1)), medium=td.Medium(permittivity=2)),
        ],
        symmetry=(1, 0, -1),
        periodic=(True, True, False),
        wavelength=1.0,
        num_pml_layers=(10, 4),
        snapping_points=(),
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
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.auto(),
        run_time=1e-12,
    )
    assert np.isclose(sim.volumetric_structures[0].geometry.bounding_box.center[2], 1, rtol=RTOL)
    sim.discretize(box.geometry).sizes.z[0]
    assert np.isclose(sim.volumetric_structures[0].geometry.bounding_box.size[2], 0, rtol=RTOL)

    # now if we increase conductivity, the in-plane grid size should decrease
    sigma2 = 4.5
    medium2 = td.Medium2D.from_medium(td.Medium(conductivity=sigma2), thickness=thickness)
    box2 = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 0), center=(0, 0, 1)), medium=medium2)

    sim2 = td.Simulation(
        size=(10, 10, 10),
        structures=[box2],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
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
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.auto(),
        run_time=1e-12,
    )

    # Commented until inplane AutoGrid for 2D materials is enabled
    # with pytest.raises(ValidationError):
    #    _ = sim.grid


def test_zerosize_dimensions():
    wvl = 1.55
    res = 20
    dl = wvl / res

    # auto grid
    sim = td.Simulation(
        size=(0, 10, 10),
        boundary_spec=td.BoundarySpec.pec(
            x=True,
            y=True,
            z=True,
        ),
        grid_spec=td.GridSpec.auto(wavelength=wvl, min_steps_per_wvl=res),
        run_time=1e-12,
    )

    assert np.allclose(sim.grid.boundaries.x, [-dl / 2, dl / 2])

    # uniform grid
    sim = td.Simulation(
        size=(5, 0, 10),
        boundary_spec=td.BoundarySpec.pec(
            x=True,
            y=True,
            z=True,
        ),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
    )

    assert np.allclose(sim.grid.boundaries.y, [0, dl])

    # custom grid
    custom_grid = td.CustomGrid(dl=tuple([0.25] * 40))
    sim = td.Simulation(
        size=(5, 0, 10),
        boundary_spec=td.BoundarySpec.pec(
            x=True,
            y=True,
            z=True,
        ),
        grid_spec=td.GridSpec(
            grid_x=custom_grid, grid_y=td.CustomGrid(dl=(dl,)), grid_z=custom_grid
        ),
        run_time=1e-12,
    )

    assert np.allclose(sim.grid.boundaries.y, [-dl / 2, dl / 2])

    with pytest.raises(SetupError):
        sim = td.Simulation(
            size=(5, 0, 10),
            boundary_spec=td.BoundarySpec.pec(
                x=True,
                y=True,
                z=True,
            ),
            grid_spec=td.GridSpec(
                grid_x=custom_grid,
                grid_y=td.CustomGrid(dl=(dl,), custom_offset=10),
                grid_z=custom_grid,
            ),
            run_time=1e-12,
        )

    with pytest.raises(SetupError):
        sim = td.Simulation(
            size=(5, 3, 10),
            boundary_spec=td.BoundarySpec.pec(
                x=True,
                y=True,
                z=True,
            ),
            grid_spec=td.GridSpec(
                grid_x=custom_grid.updated_copy(custom_offset=20),
                grid_y=custom_grid,
                grid_z=custom_grid,
            ),
            run_time=1e-12,
        )


def test_custom_grid_boundaries():
    custom = td.CustomGridBoundaries(coords=np.linspace(-1, 1, 11))
    grid_spec = td.GridSpec(grid_x=custom, grid_y=custom, grid_z=custom)
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e14), polarization="Ex"
    )

    # matches exactly
    sim = td.Simulation(
        size=(2, 2, 2),
        sources=[source],
        grid_spec=grid_spec,
        run_time=1e-12,
        medium=td.Medium(permittivity=4),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert np.allclose(sim.grid.boundaries.x, custom.coords)

    # chop off
    sim_chop = sim.updated_copy(size=(1, 1, 1))
    assert np.allclose(sim_chop.grid.boundaries.x, np.linspace(-0.4, 0.4, 5))

    sim_chop = sim.updated_copy(size=(1.2, 1, 1))
    assert np.allclose(sim_chop.grid.boundaries.x, np.linspace(-0.6, 0.6, 7))

    # expand
    sim_expand = sim.updated_copy(size=(4, 4, 4))
    assert np.allclose(sim_expand.grid.boundaries.x, np.linspace(-2, 2, 21))

    # pml
    num_layers = 10
    sim_pml = sim.updated_copy(
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=num_layers))
    )
    assert np.allclose(sim_pml.grid.boundaries.x, np.linspace(-3, 3, 31))
