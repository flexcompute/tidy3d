"""Tests GridSpec."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.exceptions import SetupError

from ..utils import assert_single_value_error_loc


def make_grid_spec():
    return td.GridSpec(wavelength=1.0)


def make_auto_grid_spec(dl_min=0.02):
    return td.GridSpec.auto(wavelength=1.0, dl_min=dl_min)


def make_single_axis_auto_grid_spec(wavelength=5e-6):
    grid_spec_kwargs = {}
    if wavelength is not None:
        grid_spec_kwargs["wavelength"] = wavelength
    return td.GridSpec(
        grid_x=td.AutoGrid(
            min_steps_per_wvl=10,
            min_steps_per_sim_size=1,
        ),
        grid_y=td.UniformGrid(dl=1e-6),
        grid_z=td.UniformGrid(dl=1e-6),
        **grid_spec_kwargs,
    )


def make_single_axis_quasiuniform_grid_spec(dl=5e-7):
    return td.GridSpec(
        grid_x=td.QuasiUniformGrid(dl=dl),
        grid_y=td.UniformGrid(dl=1e-6),
        grid_z=td.UniformGrid(dl=1e-6),
    )


def make_snapping_auto_grid_spec():
    return td.GridSpec(
        grid_x=td.AutoGrid(
            min_steps_per_wvl=10,
            min_steps_per_sim_size=1,
        ),
        grid_y=td.UniformGrid(dl=1e-6),
        grid_z=td.UniformGrid(dl=1e-6),
        wavelength=3e-5,
        snapping_points=((0, 0, 0), (5e-7, 0, 0)),
    )


def make_grid_validation_kwargs():
    return {
        "structures": [
            td.Structure(
                geometry=td.Box(size=(4e-6, 4e-6, 4e-6)),
                medium=td.Medium(),
            )
        ],
        "symmetry": (0, 0, 0),
        "periodic": (False, False, False),
        "sources": [],
        "num_pml_layers": ((0, 0), (0, 0), (0, 0)),
    }


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
    make_coords_args = {
        "structures": [
            td.Structure(geometry=td.Box(size=(2, 2, 1)), medium=td.Medium()),
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=4)),
        ],
        "symmetry": (0, 0, 0),
        "periodic": (False, False, False),
        "wavelength": 1.0,
        "num_pml_layers": (0, 0),
        "axis": 0,
    }

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
    # forming the simulation boundary
    coord = gs.grid_x.make_coords(
        snapping_points=((0.98, 0, 0), (-0.98, 0, 0)),
        **make_coords_args,
    )
    assert np.allclose(coord_original, coord)

    # and no snapping if it's completely outside the simulation domain
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

    # Switch to GridSpec emulating a user setting dl_min
    gs = make_auto_grid_spec()
    # 4) Test override behavior
    # Snapping point should override original interval boundary
    coord = gs.grid_x.make_coords(
        snapping_points=((0.56789, 0, 0),),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, 0.56789))
    # Same if on the lower side of the initial interval boundary
    coord = gs.grid_x.make_coords(
        snapping_points=((-0.56789, 0, 0),),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, -0.56789))

    gs = make_auto_grid_spec(dl_min=0.5)

    # 5) Test override behavior special cases
    # Snapping point at 0 should be added when there is space between interval boundaries
    coord = gs.grid_x.make_coords(
        snapping_points=((-0.26, 0, 0), (0.26, 0, 0), (0, 0, 0)),
        **make_coords_args,
    )
    assert np.any(np.isclose(coord, 0.0))

    # Snapping point at 0 should NOT be added when there is not enough space
    # between interval boundaries
    coord = gs.grid_x.make_coords(
        snapping_points=((-0.25, 0, 0), (0.25, 0, 0), (0, 0, 0)),
        **make_coords_args,
    )
    assert not np.any(np.isclose(coord, 0.0))


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


def test_simulation_auto_grid_missing_wavelength_validates_before_grid(monkeypatch):
    """Test that missing AutoGrid wavelength is caught before grid construction."""

    def fail_make_grid_and_snapping_lines(*args, **kwargs):
        pytest.fail("Grid generation should not run without wavelength or sources.")

    monkeypatch.setattr(
        td.GridSpec, "_make_grid_and_snapping_lines", fail_make_grid_and_snapping_lines
    )

    with pytest.raises(ValidationError, match="wavelength"):
        _ = td.Simulation(
            size=(1, 1, 1),
            grid_spec=td.GridSpec.auto(),
            run_time=1e-12,
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
        current_amplitude_definition="total",
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

    with pytest.raises(ValidationError):
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

    with pytest.raises(ValidationError):
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

    # estimated minimal step size
    estimated_dl = grid_spec.grid_x.estimated_min_dl(wavelength=1, structure_list=[], sim_size=[])
    assert np.isclose(estimated_dl, 0.2)

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


def test_small_sim_with_min_steps_per_sim_size():
    """Test if `min_steps_per_sim_size` takes effect for a small simulation domain."""

    # only single grid because simulation domain too small
    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1e4, min_steps_per_sim_size=1),
        boundary_spec=td.BoundarySpec.pml(),
    )
    assert sim.num_cells == 1

    # apply default min_steps_per_sim_size
    sim = sim.updated_copy(grid_spec=td.GridSpec.auto(wavelength=1e4, min_steps_per_sim_size=10))
    assert sim.num_cells > 500


def test_autogrid_estimated_dl():
    """Test that estimiated minimal step size in AutoGrid works as expected."""
    box = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 1)), medium=td.Medium(permittivity=4)
    )
    sim_size = [10, 10, 10]
    wavelength = 1.0
    grid_spec = td.AutoGrid(min_steps_per_wvl=10)

    # decided by medium
    estimated = grid_spec.estimated_min_dl(wavelength, [box], sim_size)
    assert np.isclose(estimated, wavelength / 10 / 2)

    # overridden by dl_min
    grid_spec_min = td.AutoGrid(min_steps_per_wvl=10, dl_min=0.1)
    estimated = grid_spec_min.estimated_min_dl(wavelength, [box], sim_size)
    assert np.isclose(estimated, 0.1)

    # decided by sim_size
    sim_size = [0.1, 0.1, 0.1]
    estimated = grid_spec.estimated_min_dl(wavelength, [box], sim_size)
    assert np.isclose(estimated, 0.1 / 10)


def test_quasiuniform_grid():
    """Test grid is quasi-uniform that adjusts to structure boundaries."""
    box = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5)),
        medium=td.Medium(permittivity=25),
    )
    sim = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.quasiuniform(dl=0.1),
        boundary_spec=td.BoundarySpec.pml(),
        structures=[box],
        run_time=1e-12,
    )

    # snapped to the boundary 0.25 because of box
    assert any(np.isclose(sim.grid.boundaries.x, 0.25))

    # add snapping points
    pos = 0.281
    snapping_points = [(pos, pos, pos)]
    sim2 = sim.updated_copy(
        grid_spec=td.GridSpec.quasiuniform(dl=0.1, snapping_points=snapping_points)
    )
    assert any(np.isclose(sim2.grid.boundaries.x, pos))

    # override structures of dl=None takes no effect
    sim3 = sim.updated_copy(
        grid_spec=td.GridSpec.quasiuniform(
            dl=0.1,
            override_structures=[
                td.MeshOverrideStructure(
                    geometry=td.Box(size=(0.2, 0.2, 0.2)), dl=[None, None, None]
                ),
            ],
        )
    )
    np.allclose(sim.grid.boundaries.x, sim3.grid.boundaries.x)

    # a larger dl_min can override step size
    grid_1d = td.QuasiUniformGrid(dl=0.1, dl_min=0.2)
    sim4 = sim.updated_copy(grid_spec=td.GridSpec(grid_x=grid_1d, grid_y=grid_1d, grid_z=grid_1d))
    assert np.all(sim4.grid.sizes.x > 0.11)

    # 2d simulation
    sim5 = sim.updated_copy(size=(0, 1, 1))
    assert np.isclose(sim5.grid.sizes.x, 0.1)


def test_domain_mismatch():
    """Test that generated grids match simulation domain. Previously it errors for z-axis."""
    lam0 = 5

    length = 5
    width = 5
    top_thickness = 0.2
    bottom_thickness = 0.2

    bottom = td.Structure(
        geometry=td.Box(
            center=[0, 0, 0],
            size=[width, length, bottom_thickness],
        ),
        medium=td.Medium(permittivity=4),
    )

    top = td.Structure(
        geometry=td.Box(
            center=[0, 0, bottom_thickness / 2 + top_thickness / 2],
            size=[width, length, top_thickness],
        ),
        medium=td.Medium(),
    )

    sim = td.Simulation(
        center=[0, 0, 0],
        size=[6, 6, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=lam0,
        ),
        structures=[bottom, top],
        sources=[],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.pml(),
    )
    z = sim.grid.boundaries.z


@pytest.mark.parametrize(
    ("dl", "expect_exception"),
    [
        (1e-7, True),  # Below 1e-6 => fail
        (1e-6, False),  # Exactly at lower bound => pass
        (0.0, True),  # Zero => fail
    ],
)
def test_uniform_grid_dl_validation(dl, expect_exception):
    """Test the validator that checks 'dl' is at least 1e-6 µm."""
    if expect_exception:
        with pytest.raises(ValidationError):
            _ = td.Simulation(
                size=(1, 1, 1),
                grid_spec=td.GridSpec.uniform(dl=dl),
                run_time=1e-12,
            )
    else:
        _ = td.Simulation(
            size=(1, 1, 1),
            grid_spec=td.GridSpec.uniform(dl=dl),
            run_time=1e-12,
        )


@pytest.mark.parametrize(
    ("wavelength", "expect_exception"),
    [
        (5e-6, True),
        (3e-5, False),
    ],
)
def test_autogrid_generated_dl_validation(wavelength, expect_exception):
    """Test that AutoGrid-generated spacing below the unit-check threshold errors."""
    sim_kwargs = {
        "size": (4e-6, 4e-6, 4e-6),
        "boundary_spec": td.BoundarySpec.pec(x=True, y=True, z=True),
        "grid_spec": make_single_axis_auto_grid_spec(wavelength),
        "run_time": 1e-12,
    }

    if expect_exception:
        with pytest.raises(ValidationError) as excinfo:
            _ = td.Simulation(**sim_kwargs)
        assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "AutoGrid generated")
    else:
        sim = td.Simulation(**sim_kwargs)
        assert sim.num_cells < 100


def test_autogrid_make_grid_generated_dl_validation(monkeypatch):
    """Test that direct GridSpec grid generation catches too-small AutoGrid spacing."""
    grid_spec = make_single_axis_auto_grid_spec()

    def fail_make_coords(*args, **kwargs):
        raise AssertionError("AutoGrid.make_coords should not run after a failed size estimate.")

    monkeypatch.setattr(td.AutoGrid, "make_coords", fail_make_coords)

    with pytest.raises(SetupError, match="AutoGrid generated"):
        _ = grid_spec.make_grid(**make_grid_validation_kwargs())


def test_autogrid_make_grid_layer_refinement_validates_generated_grid(monkeypatch):
    """Test that layer-refinement bounds do not fail before generated grid validation."""
    grid_spec = make_single_axis_auto_grid_spec(wavelength=3e-5).updated_copy(
        layer_refinement_specs=[
            td.LayerRefinementSpec(
                axis=0,
                center=(0, 0, 0),
                size=(4e-6, 4e-6, 4e-6),
                corner_finder=None,
                corner_refinement=None,
                bounds_refinement=None,
            )
        ]
    )

    monkeypatch.setattr(td.LayerRefinementSpec, "suggested_dl_min", lambda *args, **kwargs: 1.5e-6)
    monkeypatch.setattr(td.AutoGrid, "make_coords", lambda *args, **kwargs: np.array([-2e-6, 2e-6]))

    grid = grid_spec.make_grid(**make_grid_validation_kwargs())

    assert np.min(grid.sizes.x) > 1e-6


def test_autogrid_make_grid_gap_refinement_validation(monkeypatch):
    """Test that gap meshing validates the generated grid after remeshing."""
    grid_spec = make_single_axis_auto_grid_spec(wavelength=3e-5).updated_copy(
        layer_refinement_specs=[
            td.LayerRefinementSpec(
                axis=0,
                center=(0, 0, 0),
                size=(4e-6, 4e-6, 4e-6),
                corner_finder=None,
                corner_refinement=None,
                bounds_refinement=None,
                gap_meshing_iters=1,
                dl_min_from_gap_width=True,
            )
        ]
    )

    monkeypatch.setattr(td.LayerRefinementSpec, "_merged_geos", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        td.LayerRefinementSpec,
        "_resolve_gaps",
        lambda *args, **kwargs: ([(0, None, None)], 1e-6),
    )

    make_grid_one_iteration = td.GridSpec._make_grid_one_iteration
    num_grid_iterations = 0
    old_grid = None

    def wrapped_make_grid_one_iteration(*args, **kwargs):
        nonlocal num_grid_iterations, old_grid
        num_grid_iterations += 1
        if num_grid_iterations == 1:
            old_grid = make_grid_one_iteration(*args, **kwargs)
            return old_grid
        return old_grid

    monkeypatch.setattr(td.GridSpec, "_make_grid_one_iteration", wrapped_make_grid_one_iteration)

    _ = grid_spec.make_grid(**make_grid_validation_kwargs())
    assert num_grid_iterations == 2


def test_autogrid_make_grid_reuses_preprocessed_structures(monkeypatch):
    """Test that grid generation does not rebuild override-augmented structures."""
    grid_spec = td.GridSpec.auto(wavelength=1.55, min_steps_per_sim_size=1)
    get_all_structures_affecting_grid = td.GridSpec._get_all_structures_affecting_grid
    num_calls = 0

    def wrapped_get_all_structures_affecting_grid(*args, **kwargs):
        nonlocal num_calls
        num_calls += 1
        return get_all_structures_affecting_grid(*args, **kwargs)

    monkeypatch.setattr(
        td.GridSpec, "_get_all_structures_affecting_grid", wrapped_get_all_structures_affecting_grid
    )

    _ = grid_spec.make_grid(
        **(
            make_grid_validation_kwargs()
            | {
                "structures": [
                    td.Structure(geometry=td.Box(size=(4, 4, 4)), medium=td.Medium()),
                ]
            }
        )
    )
    assert num_calls == 1


def test_autogrid_make_grid_snapping_validation():
    """Test that snapping points cannot produce too-small generated spacing."""
    with pytest.raises(SetupError, match="AutoGrid generated"):
        _ = make_snapping_auto_grid_spec().make_grid(
            **(
                make_grid_validation_kwargs()
                | {
                    "structures": [
                        td.Structure(
                            geometry=td.Box(size=(1e-5, 4e-6, 4e-6)),
                            medium=td.Medium(),
                        )
                    ]
                }
            )
        )


def test_autogrid_snapping_validation_loc():
    """Test that snapping-driven generated spacing errors anchor on the grid spec."""
    with pytest.raises(ValidationError) as excinfo:
        _ = td.Simulation(
            size=(1e-5, 4e-6, 4e-6),
            boundary_spec=td.BoundarySpec.pec(x=True, y=True, z=True),
            grid_spec=make_snapping_auto_grid_spec(),
            run_time=1e-12,
        )
    assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "AutoGrid generated")


def test_autogrid_validation_uses_zero_dim_normalized_boundaries(monkeypatch):
    """Test that grid validation sees boundary updates for zero-size dimensions."""
    boundary_types_seen = []
    make_grid_and_snapping_lines = td.GridSpec._make_grid_and_snapping_lines

    def wrapped_make_grid_and_snapping_lines(*args, **kwargs):
        boundary_types_seen.append(kwargs["boundary_types"])
        return make_grid_and_snapping_lines(*args, **kwargs)

    monkeypatch.setattr(
        td.GridSpec, "_make_grid_and_snapping_lines", wrapped_make_grid_and_snapping_lines
    )

    sim = td.Simulation(
        size=(0, 4, 4),
        boundary_spec=td.BoundarySpec.pml(x=True),
        grid_spec=td.GridSpec.auto(wavelength=1.55),
        run_time=1e-12,
    )

    assert boundary_types_seen[0][0] == ["periodic", "periodic"]
    assert isinstance(sim.boundary_spec.x.minus, td.Periodic)
    assert isinstance(sim.boundary_spec.x.plus, td.Periodic)


def test_autogrid_gap_refinement_validation_loc(monkeypatch):
    """Test that gap-refinement generated spacing errors anchor on the grid spec."""
    grid_spec = make_single_axis_auto_grid_spec(wavelength=3e-5).updated_copy(
        layer_refinement_specs=[
            td.LayerRefinementSpec(
                axis=0,
                center=(0, 0, 0),
                size=(4e-6, 4e-6, 4e-6),
                corner_finder=None,
                corner_refinement=None,
                bounds_refinement=None,
                gap_meshing_iters=1,
                dl_min_from_gap_width=True,
            )
        ]
    )

    monkeypatch.setattr(td.LayerRefinementSpec, "_merged_geos", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        td.LayerRefinementSpec,
        "_resolve_gaps",
        lambda *args, **kwargs: ([(0, None, None)], 1e-6),
    )

    make_grid_one_iteration = td.GridSpec._make_grid_one_iteration
    old_grid = None

    def wrapped_make_grid_one_iteration(*args, **kwargs):
        nonlocal old_grid
        if old_grid is None:
            old_grid = make_grid_one_iteration(*args, **kwargs)
            return old_grid
        return old_grid.updated_copy(
            boundaries=old_grid.boundaries.updated_copy(x=np.array([-2e-6, -1.5e-6, 2e-6]))
        )

    monkeypatch.setattr(td.GridSpec, "_make_grid_one_iteration", wrapped_make_grid_one_iteration)

    with pytest.raises(ValidationError) as excinfo:
        _ = td.Simulation(
            size=(4e-6, 4e-6, 4e-6),
            boundary_spec=td.BoundarySpec.pec(x=True, y=True, z=True),
            grid_spec=grid_spec,
            run_time=1e-12,
        )
    assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "AutoGrid generated")


def test_quasiuniform_generated_dl_validation():
    """Test that quasi-uniform generated spacing below the unit-check threshold errors."""
    with pytest.raises(ValidationError) as excinfo:
        _ = td.Simulation(
            size=(4e-6, 4e-6, 4e-6),
            boundary_spec=td.BoundarySpec.pec(x=True, y=True, z=True),
            grid_spec=make_single_axis_quasiuniform_grid_spec(),
            run_time=1e-12,
        )
    assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "QuasiUniformGrid generated")


def test_quasiuniform_make_grid_generated_dl_validation(monkeypatch):
    """Test that direct quasi-uniform grid generation catches too-small spacing."""
    grid_spec = make_single_axis_quasiuniform_grid_spec()

    monkeypatch.setattr(
        td.QuasiUniformGrid,
        "make_coords",
        lambda *args, **kwargs: pytest.fail("QuasiUniformGrid.make_coords should not run."),
    )

    with pytest.raises(SetupError, match="QuasiUniformGrid generated"):
        _ = grid_spec.make_grid(**make_grid_validation_kwargs())


@pytest.mark.parametrize(
    ("sim_cls", "sim_kwargs"),
    [
        (
            td.ModeSimulation,
            {
                "size": (4e-6, 4e-6, 0),
                "mode_spec": td.ModeSpec(),
                "boundary_spec": td.BoundarySpec(
                    x=td.Boundary.pec(),
                    y=td.Boundary.pec(),
                    z=td.Boundary.periodic(),
                ),
            },
        ),
        (
            td.EMESimulation,
            {
                "size": (4e-6, 4e-6, 4e-6),
                "axis": 2,
                "eme_grid_spec": td.EMEUniformGrid(
                    num_cells=1,
                    mode_spec=td.EMEModeSpec(num_modes=1),
                ),
                "boundary_spec": td.BoundarySpec.pec(x=True, y=True, z=True),
            },
        ),
    ],
    ids=["mode", "eme"],
)
def test_y_grid_simulation_autogrid_generated_dl_validation_loc(sim_cls, sim_kwargs):
    """Test that Yee-grid simulations anchor too-small AutoGrid validation on their grid spec."""
    with pytest.raises(ValidationError) as excinfo:
        _ = sim_cls(
            **sim_kwargs,
            freqs=[td.C_0 / 5e-6],
            grid_spec=make_single_axis_auto_grid_spec(wavelength=None),
        )
    assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "AutoGrid generated")


@pytest.mark.parametrize(
    ("sim_cls", "sim_kwargs"),
    [
        (
            td.ModeSimulation,
            {
                "size": (1e-5, 4e-6, 0),
                "mode_spec": td.ModeSpec(),
                "boundary_spec": td.BoundarySpec(
                    x=td.Boundary.pec(),
                    y=td.Boundary.pec(),
                    z=td.Boundary.periodic(),
                ),
            },
        ),
        (
            td.EMESimulation,
            {
                "size": (1e-5, 4e-6, 4e-6),
                "axis": 2,
                "eme_grid_spec": td.EMEUniformGrid(
                    num_cells=1,
                    mode_spec=td.EMEModeSpec(num_modes=1),
                ),
                "boundary_spec": td.BoundarySpec.pec(x=True, y=True, z=True),
            },
        ),
    ],
    ids=["mode", "eme"],
)
def test_y_grid_simulation_autogrid_snapping_validation_loc(sim_cls, sim_kwargs):
    """Test that Yee-grid simulations anchor snapping-driven grid errors on their grid spec."""
    with pytest.raises(ValidationError) as excinfo:
        _ = sim_cls(
            **sim_kwargs,
            freqs=[td.C_0 / 1.55],
            grid_spec=make_snapping_auto_grid_spec(),
        )
    assert_single_value_error_loc(excinfo, ("grid_spec", "grid_x"), "AutoGrid generated")


def test_custom_grid_boundary_validation():
    """Tests that the 'coords' is at least length 2 and sorted in ascending order."""

    with pytest.raises(ValidationError):
        _ = td.CustomGridBoundaries(coords=[10])

    with pytest.raises(ValidationError):
        _ = td.CustomGridBoundaries(coords=[9, 10, 9, 10, 11, 9, 8])


def test_grid_spec_localized_copy_filters_quasiuniform_entities():
    region = td.Box(center=(0, 0, 0), size=(4.0, 6.0, td.inf))
    keep_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 0, 0), size=(1, 1, 1)),
        dl=(None, 0.1, 0.1),
    )
    drop_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 10, 0), size=(1, 1, 1)),
        dl=(0.1, 0.1, 0.1),
    )
    grid_spec = td.GridSpec.quasiuniform(
        dl=0.25,
        override_structures=(keep_override, drop_override),
        snapping_points=((0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (10.0, 10.0, 5.0)),
    )

    localized = grid_spec._localized_copy(region=region)

    assert len(localized.override_structures) == 2
    assert localized.override_structures[0].dl == (None, 0.1, 0.1)
    assert localized.override_structures[1].dl == (None, None, 0.1)
    assert localized.snapping_points == (
        (0.0, 0.0, 5.0),
        (None, 0.0, 5.0),
        (None, None, 5.0),
    )


def test_grid_spec_localized_copy_preserves_override_structure_order():
    region = td.Box(center=(0, 0, 0), size=(4.0, 6.0, td.inf))
    mesh_override_a = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 0, 0), size=(1, 1, 1)),
        dl=(0.1, 0.1, 0.1),
        name="mesh_override_a",
    )
    structure_override_b = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
        medium=td.Medium(permittivity=2.0),
        name="structure_override_b",
    )
    mesh_override_c = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 10, 0), size=(1, 1, 1)),
        dl=(0.1, 0.1, 0.1),
        name="mesh_override_c",
    )
    grid_spec = td.GridSpec.auto(
        wavelength=1.0,
        override_structures=(mesh_override_a, structure_override_b, mesh_override_c),
    )

    localized = grid_spec._localized_copy(region=region)

    assert [struct.name for struct in localized.override_structures] == [
        "mesh_override_a",
        "structure_override_b",
        "mesh_override_c",
    ]
    assert localized.override_structures[0].dl == (None, 0.1, 0.1)
    assert localized.override_structures[2].dl == (0.1, None, 0.1)


def test_grid_spec_localized_copy_filters_min_steps_per_size():
    """'min_steps_per_size' overrides are filtered to the region like 'dl' overrides."""
    region = td.Box(center=(0, 0, 0), size=(4.0, 6.0, 8.0))
    # x is outside the region, y and z are inside -> x axis disabled, structure kept
    partial = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 0, 0), size=(2, 2, 2)),
        min_steps_per_size=(10, 10, 10),
        name="partial",
    )
    # entirely outside the region -> dropped, no effective grid size remains
    fully_out = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 10, 10), size=(2, 2, 2)),
        min_steps_per_size=(10, 10, 10),
        name="fully_out",
    )
    grid_spec = td.GridSpec.auto(
        wavelength=1.0,
        override_structures=(partial, fully_out),
    )

    localized = grid_spec._localized_copy(region=region)

    assert [struct.name for struct in localized.override_structures] == ["partial"]
    kept = localized.override_structures[0]
    # filtering bakes the resolved grid size into 'dl' and clears 'min_steps_per_size';
    # the effective '_dl' is unchanged along the kept axes
    assert kept._dl == (None, 0.2, 0.2)
    assert kept.dl == (None, 0.2, 0.2)
    assert kept.min_steps_per_size == (None, None, None)
