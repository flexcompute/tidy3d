"""Tests 2d corner finder."""

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.grid.corner_finder import CornerFinderSpec
from tidy3d.components.grid.grid_spec import GridRefinement, LayerRefinementSpec

CORNER_FINDER = CornerFinderSpec()
GRID_REFINEMENT = GridRefinement()
LAYER_REFINEMENT = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 2))
LAYER2D_REFINEMENT = LayerRefinementSpec(axis=2, size=(td.inf, td.inf, 0))


def test_2dcorner_finder_filter_collinear_vertex():
    """In corner finder, test that collinear vertices are filtered"""
    # 2nd and 3rd vertices are on a collinear line
    vertices = ((0, 0), (0.1, 0), (0.5, 0), (1, 0), (1, 1))
    polyslab = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 3

    # if angle threshold is 0, collinear vertex will not be filtered
    corner_finder = CORNER_FINDER.updated_copy(angle_threshold=0)
    corners = corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 5


def test_2dcorner_finder_filter_nearby_vertex():
    """In corner finder, test that vertices that are very close are filtered"""
    # filter duplicate vertices
    vertices = ((0, 0), (0, 0), (1e-4, -1e-4), (1, 0), (1, 1))
    polyslab = td.PolySlab(vertices=vertices, axis=2, slab_bounds=[-1, 1])
    structures = [td.Structure(geometry=polyslab, medium=td.PEC)]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 4

    # filter very close vertices
    corner_finder = CORNER_FINDER.updated_copy(distance_threshold=2e-4)
    corners = corner_finder.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 3


def test_2dcorner_finder_medium():
    """No corner found if the medium is dielectric while asking to search for corner of metal."""
    structures = [td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    assert len(corners) == 0


def test_2dcorner_finder_polygon_with_hole():
    """Find corners related to interior holes of a polygon."""
    structures = [
        td.Structure(geometry=td.Box(size=(2, 2, 2)), medium=td.PEC),
        td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium()),
    ]
    corners = CORNER_FINDER.corners(normal_axis=2, coord=0, structure_list=structures)
    # 4 interior, 4 exterior
    assert len(corners) == 8


def test_gridrefinement():
    """Test GradRefinement is working as expected."""

    # generate override structures for z-axis
    center = [None, None, 0]
    grid_size_in_vaccum = 1
    structure = GRID_REFINEMENT.override_structure(center, grid_size_in_vaccum)
    assert not structure.shadow
    for axis in range(2):
        assert structure.dl[axis] is None
        assert structure.geometry.size[axis] == td.inf
    dl = grid_size_in_vaccum / GRID_REFINEMENT._refinement_factor
    assert np.isclose(structure.dl[2], dl)
    assert np.isclose(structure.geometry.size[2], dl * GRID_REFINEMENT.num_cells)

    # explicitly define step size in refinement region that is smaller than that of refinement_factor
    dl = 1
    grid_refinement = GRID_REFINEMENT.updated_copy(dl=dl)
    structure = grid_refinement.override_structure(center, grid_size_in_vaccum)
    for axis in range(2):
        assert structure.dl[axis] is None
        assert structure.geometry.size[axis] == td.inf
    assert np.isclose(structure.dl[2], dl)
    assert np.isclose(structure.geometry.size[2], dl * GRID_REFINEMENT.num_cells)


def test_layerrefinement():
    """Test LayerRefinementSpec is working as expected."""

    # size along axis must be inf
    with pytest.raises(pydantic.ValidationError):
        _ = LayerRefinementSpec(axis=0, size=(td.inf, 0, 0))

    # classmethod
    for axis in range(3):
        layer = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, 1))
        assert layer.center[axis] == 0.5
        assert layer.size[axis] == 1
        assert layer.size[(axis + 1) % 3] == td.inf
        assert layer.size[(axis + 2) % 3] == td.inf
        assert layer._is_inplane_unbounded

    layer = LayerRefinementSpec.from_bounds(axis=axis, rmin=(0, 0, 0), rmax=(1, 2, 3))
    layer = LayerRefinementSpec.from_bounds(rmin=(0, 0, 0), rmax=(1, 2, 3))
    assert layer.axis == 0
    assert np.isclose(layer.length_axis, 1)
    assert np.isclose(layer.center_axis, 0.5)
    assert not layer._is_inplane_unbounded

    # from structures
    structures = [td.Structure(geometry=td.Box(size=(td.inf, 2, 3)), medium=td.Medium())]
    layer = LayerRefinementSpec.from_structures(structures)
    assert layer.axis == 1

    with pytest.raises(pydantic.ValidationError):
        structures = [
            td.Structure(geometry=td.Box(size=(td.inf, td.inf, td.inf)), medium=td.Medium())
        ]
        layer = LayerRefinementSpec.from_structures(structures)

    with pytest.raises(pydantic.ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, td.inf))
    with pytest.raises(pydantic.ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(td.inf, 0))
    with pytest.raises(pydantic.ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(-td.inf, 0))
    with pytest.raises(pydantic.ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(1, -1))


def test_layerrefinement_inplane_inside():
    # inplane inside
    layer = LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(0, 1))
    assert layer._inplane_inside([3e3, 4e4])
    layer = LayerRefinementSpec(axis=1, size=(1, 0, 1))
    assert layer._inplane_inside([0, 0])
    assert not layer._inplane_inside([2, 0])


def test_layerrefinement_snapping_points():
    """Test snapping points for LayerRefinementSpec is working as expected."""

    # snapping points for layer bounds
    points = LAYER2D_REFINEMENT._snapping_points_along_axis
    assert len(points) == 1
    assert points[0] == (None, None, 0)

    points = LAYER_REFINEMENT._snapping_points_along_axis
    assert len(points) == 1
    assert points[0] == (None, None, -1)


def test_grid_spec_with_layers():
    """Test the application of layer_specs to GridSpec."""

    thickness = 1e-3
    # a PEC thin layer structure
    box1 = td.Box(size=(thickness, 2, 2))
    box2 = td.Box(center=(0, -1, 0), size=(thickness, 1, 1))
    pec_str = td.Structure(geometry=box1 - box2, medium=td.PEC)
    # a pin
    pin_str = td.Structure(
        geometry=td.Cylinder(axis=0, radius=0.1, length=1.1, center=(-0.5, 0, 0)), medium=td.PEC
    )
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ]
    )
    assert layer.axis == 0

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer]
        ),
        boundary_spec=td.BoundarySpec.pml(),
        structures=[pec_str, pin_str],
        run_time=1e-12,
    )
    # lower bound is snapped
    assert any(np.isclose(sim.grid.boundaries.x, -thickness / 2))
    # corner snapped
    assert any(np.isclose(sim.grid.boundaries.y, -0.5))
    assert any(np.isclose(sim.grid.boundaries.z, -0.5))
    assert any(np.isclose(sim.grid.boundaries.z, 0.5))

    # differnt laye parameters
    def update_sim_with_newlayer(layer):
        return sim.updated_copy(
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer]
            )
        )

    # bounds snapping
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_snapping="bounds",
    )
    sim2 = update_sim_with_newlayer(layer)
    assert any(np.isclose(sim2.grid.boundaries.x, -thickness / 2))
    assert any(np.isclose(sim2.grid.boundaries.x, thickness / 2))

    # layer thickness refinement
    def count_grids_within_layer(sim_t):
        float_relax = 1.001
        x = sim_t.grid.boundaries.x
        x = x[x >= -thickness / 2 * float_relax]
        x = x[x <= thickness / 2 * float_relax]
        return len(x)

    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) == 4

    # layer thickness refinement + bounds refinement, but the latter is too coarse so that it's abandoned
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
        bounds_refinement=td.GridRefinement(),
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) == 4
    # much finer refinement to be included
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        min_steps_along_axis=3,
        bounds_refinement=td.GridRefinement(dl=thickness / 10),
    )
    sim2 = update_sim_with_newlayer(layer)
    assert count_grids_within_layer(sim2) > 10

    # layer bounds refinement: combined into one structure when they overlap
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_refinement=td.GridRefinement(dl=thickness * 1.1, num_cells=1),
        corner_finder=None,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert len(sim2.grid_spec.all_override_structures(list(sim2.structures), 1.0, sim2.size)) == 1

    # separate when they don't overlap
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_refinement=td.GridRefinement(dl=thickness * 0.9, num_cells=1),
        corner_finder=None,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert len(sim2.grid_spec.all_override_structures(list(sim2.structures), 1.0, sim2.size)) == 2
