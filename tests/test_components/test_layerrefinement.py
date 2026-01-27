"""Tests 2d corner finder."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

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
    structure = GRID_REFINEMENT.override_structure(center, grid_size_in_vaccum, True)
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
    structure = grid_refinement.override_structure(center, grid_size_in_vaccum, True)
    for axis in range(2):
        assert structure.dl[axis] is None
        assert structure.geometry.size[axis] == td.inf
    assert np.isclose(structure.dl[2], dl)
    assert np.isclose(structure.geometry.size[2], dl * GRID_REFINEMENT.num_cells)


def test_layerrefinement():
    """Test LayerRefinementSpec is working as expected."""

    # size along axis must be inf
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec(axis=0, size=(td.inf, 0, 0))

    # classmethod
    for axis in range(3):
        layer = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, 1))
        assert layer.center[axis] == 0.5
        assert layer.size[axis] == 1
        assert layer.size[(axis + 1) % 3] == td.inf
        assert layer.size[(axis + 2) % 3] == td.inf
        assert not layer._is_inplane_bounded(layer)

    layer = LayerRefinementSpec.from_bounds(axis=axis, rmin=(0, 0, 0), rmax=(1, 2, 3))
    layer = LayerRefinementSpec.from_bounds(rmin=(0, 0, 0), rmax=(1, 2, 3))
    assert layer.axis == 0
    assert np.isclose(layer.length_axis, 1)
    assert np.isclose(layer.center_axis, 0.5)
    assert layer._is_inplane_bounded(layer)

    # from structures
    structures = [td.Structure(geometry=td.Box(size=(td.inf, 2, 3)), medium=td.Medium())]
    layer = LayerRefinementSpec.from_structures(structures)
    assert layer._is_inplane_bounded(layer)
    assert layer.axis == 1

    with pytest.raises(ValidationError):
        structures = [
            td.Structure(geometry=td.Box(size=(td.inf, td.inf, td.inf)), medium=td.Medium())
        ]
        layer = LayerRefinementSpec.from_structures(structures)

    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(0, td.inf))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(td.inf, 0))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(-td.inf, 0))
    with pytest.raises(ValidationError):
        _ = LayerRefinementSpec.from_layer_bounds(axis=axis, bounds=(1, -1))


def test_layerrefinement_inplane_inside():
    # inplane inside
    layer = LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(0, 1))
    assert not layer._is_inplane_bounded(layer)
    assert layer._inplane_inside(layer, [3e3, 4e4])
    layer = LayerRefinementSpec(axis=1, size=(1, 0, 1))
    assert layer._inplane_inside(layer, [0, 0])
    assert not layer._inplane_inside(layer, [2, 0])


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
    lumped_elements = []
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
    assert (
        len(
            sim2.grid_spec.all_override_structures(
                list(sim2.structures),
                1.0,
                lumped_elements,
                sim2._internal_layerrefinement_boundary_types,
                sim2.bounds,
            )
        )
        == 1
    )

    # separate when they don't overlap
    layer = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        bounds_refinement=td.GridRefinement(dl=thickness * 0.9, num_cells=1),
        corner_finder=None,
    )
    sim2 = update_sim_with_newlayer(layer)
    assert (
        len(
            sim2.grid_spec.all_override_structures(
                list(sim2.structures),
                1.0,
                lumped_elements,
                sim2._internal_layerrefinement_boundary_types,
                sim2.bounds,
            )
        )
        == 2
    )


def test_grid_spec_with_layers_interior_disjoint():
    """Test the application of layer_specs with interior_disjoint assumption to GridSpec."""

    thickness = 1e-3
    lumped_elements = []
    # a PEC thin layer structure
    box1 = td.Box(size=(thickness, 2, 2))
    box2 = td.Box(center=(0, -1, 0), size=(thickness, 1, 1))
    pec_str = td.Structure(geometry=box1 - box2, medium=td.PEC)
    # a dielectric that overlaps with PEC structure, hence breaking interior-disjoint assumption
    die_str = td.Structure(
        geometry=td.Box(size=(thickness, 0.5, 0.5)),
        medium=td.Medium(),
    )
    # layer spec assuming interior disjoint geometries
    layer_disjoint = LayerRefinementSpec.from_structures(
        [
            pec_str,
        ],
        interior_disjoint_geometries=True,
    )
    # layer spec for general geometries
    layer_general = layer_disjoint.updated_copy(interior_disjoint_geometries=False)

    sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=11, wavelength=1, layer_refinement_specs=[layer_disjoint]
        ),
        boundary_spec=td.BoundarySpec.pml(),
        structures=[pec_str],
        run_time=1e-12,
    )
    sim_general = sim.updated_copy(
        grid_spec=sim.grid_spec.updated_copy(layer_refinement_specs=[layer_general])
    )

    # for simulations with interior-disjoint geometries, same corner finder results for both layer settings
    def is_equal_snapping_points(plist1, plist2):
        if len(plist1) != len(plist2):
            return False
        for p1, p2 in zip(plist1, plist2):
            if p1 != p2:
                return False
        return True

    assert is_equal_snapping_points(
        sim.internal_snapping_points, sim_general.internal_snapping_points
    )

    # for simulations with overlapping geometries, assuming interior-disjoint misses some corners in this simulation
    sim = sim.updated_copy(structures=[pec_str, die_str])
    sim_general = sim_general.updated_copy(structures=[pec_str, die_str])
    assert len(sim_general.internal_snapping_points) > len(sim.internal_snapping_points)


@pytest.mark.parametrize("gap_meshing_iters", [0, 1])
def test_corner_refinement_outside_domain(gap_meshing_iters):
    """Test the behavior of corner refinement if corners are outside the simulation domain."""

    # CPW waveguides that goes through the simulation domain along x-axis, so that the corners
    # are outside the simulation domain.
    wg_length = 10
    wg1 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1)),
        medium=td.PEC,
    )

    wg2 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1), center=(0, 0.25, 0)),
        medium=td.PEC,
    )

    wg3 = td.Structure(
        geometry=td.Box(size=(wg_length, 0.2, 1), center=(0, -0.25, 0)),
        medium=td.PEC,
    )
    structures = [wg1, wg2, wg3]

    # 1) not refined in the gap along y-axis because corners are outside the simulation domain
    layer = td.LayerRefinementSpec.from_structures(
        structures,
        axis=2,
        corner_refinement=td.GridRefinement(refinement_factor=5),
        refinement_inside_sim_only=True,
        gap_meshing_iters=gap_meshing_iters,
    )

    sim = td.Simulation(
        size=(2, 2, 2),
        structures=structures,
        grid_spec=td.GridSpec.auto(
            wavelength=1,
            layer_refinement_specs=[layer],
        ),
        run_time=1e-20,
    )

    def count_grids_within_gap(sim_t):
        float_relax = 1.001
        y = sim_t.grid.boundaries.y
        y = y[y * float_relax >= 0.1]
        y = y[y <= 0.15 * float_relax]
        return len(y)

    # just 2 grids sampling the gap
    assert count_grids_within_gap(sim) == gap_meshing_iters + 2

    # 2) refined if corners outside simulation domain is accounted for.
    layer = layer.updated_copy(refinement_inside_sim_only=False)
    sim = sim.updated_copy(grid_spec=td.GridSpec.auto(wavelength=1, layer_refinement_specs=[layer]))

    assert count_grids_within_gap(sim) > 2


def test_dl_min_from_smallest_feature():
    structure = td.Structure(
        geometry=td.PolySlab(
            vertices=[
                [0, 0],
                [2, 0],
                [2, 1],
                [1, 1],
                [1, 1.1],
                [2, 1.1],
                [2, 2],
                [1, 2],
                [1, 2.2],
                [0.7, 2.2],
                [0.7, 2],
                [0, 2],
            ],
            slab_bounds=[-1, 1],
            axis=2,
        ),
        medium=td.PECMedium(),
    )
    boundary_types = [[None] * 2] * 3
    sim_bounds = [
        [-td.inf] * 3,
        [td.inf] * 3,
    ]
    # check expected dl_min
    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(
            convex_resolution=10,
        ),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.3 / 10, dl_min)

    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(mixed_resolution=10),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.2 / 10, dl_min)

    layer_spec = td.LayerRefinementSpec(
        axis=2,
        size=(td.inf, td.inf, 2),
        corner_finder=td.CornerFinderSpec(
            concave_resolution=10,
        ),
    )
    dl_min = layer_spec._dl_min_from_smallest_feature([structure], sim_bounds, boundary_types)
    assert np.allclose(0.1 / 10, dl_min)

    # check grid is generated succesfully
    sim = td.Simulation(
        size=(5, 5, 5),
        structures=[structure],
        grid_spec=td.GridSpec.auto(layer_refinement_specs=[layer_spec], wavelength=100 * td.C_0),
        run_time=1e-20,
    )

    _ = sim.grid


def test_gap_meshing():
    w = 1
    length = 10

    l_shape_1 = td.Structure(
        medium=td.PECMedium(),
        geometry=td.PolySlab(
            axis=2,
            slab_bounds=[0, 2],
            vertices=[
                [0, 0],
                [length, 0],
                [length, w],
                [w, w],
                [w, length],
                [0, length],
            ],
        ),
    )

    gap = 0.1
    l_shape_2 = td.Structure(
        medium=td.PECMedium(),
        geometry=td.PolySlab(
            axis=2,
            slab_bounds=[0, 2],
            vertices=[
                [w + gap, w + gap],
                [w + gap + length, w + gap],
                [w + gap + length, w + gap + w],
                [w + gap + w, w + gap + w],
                [w + gap + w, w + gap + length],
                [w + gap, w + gap + length],
                [0, w + gap + length],
                [0, gap + length],
                [w + gap, gap + length],
            ],
        ),
    )

    # ax = l_shape_1.plot(z=1)
    # l_shape_2.plot(z=1, ax=ax)

    for num_iters in range(2):
        grid_spec = td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=10),
            grid_y=td.AutoGrid(min_steps_per_wvl=10),
            grid_z=td.AutoGrid(min_steps_per_wvl=10),
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=2,
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=num_iters,
                    size=[td.inf, td.inf, 2],
                    center=[0, 0, 1],
                )
            ],
            wavelength=7,
        )

        sim = td.Simulation(
            structures=[l_shape_1, l_shape_2],
            grid_spec=grid_spec,
            size=(1.2 * length, 1.2 * length, 2),
            center=(0.5 * length, 0.5 * length, 0),
            run_time=1e-15,
        )

        # ax = sim.plot(z=1)
        # sim.plot_grid(z=1, ax=ax)
        # ax.set_xlim([0, 2])
        # ax.set_ylim([0, 2])

        resolved_x = np.any(
            np.logical_and(sim.grid.boundaries.x > w, sim.grid.boundaries.x < w + gap)
        )
        resolved_y = np.any(
            np.logical_and(sim.grid.boundaries.y > w, sim.grid.boundaries.y < w + gap)
        )

        if num_iters == 0:
            assert (not resolved_x) and (not resolved_y)
        else:
            assert resolved_x and resolved_y

    # test ingored small feature
    sim_size = (1, 1, 1)
    box = td.Structure(
        geometry=td.Box(size=(0.95, 0.2, 0.95)),
        medium=td.PECMedium(),
    )

    reentry_gap = td.Structure(
        geometry=td.PolySlab(
            slab_bounds=(-0.2, 0.2),
            axis=1,
            vertices=[(-0.3, 0.52), (-0.05, 0.3), (0.2, 0.52)],
        ),
        medium=td.Medium(),
    )

    aux = td.Structure(
        geometry=td.Box(center=(0.0, 0, 0.5), size=(2, 0.4, 0.23)),
        medium=td.Medium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[box, reentry_gap, aux],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    assert not any(
        np.isclose(sim.grid.boundaries.x, reentry_gap.geometry.bounding_box.center[0], rtol=1e-2)
    )
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test internal polygon

    gap_z = td.Structure(
        geometry=td.Box(center=(0.3, 0, 0.1), size=(0.05, 0.4, 0.5)),
        medium=td.Medium(),
    )

    gap_x = td.Structure(
        geometry=td.Box(center=(0.03, 0, 0.07), size=(0.3, 0.4, 0.05)),
        medium=td.Medium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[box, gap_x, gap_z],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=2,
                    dl_min_from_gap_width=True,
                    interior_disjoint_geometries=False,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    assert any(np.isclose(sim.grid.boundaries.x, gap_z.geometry.center[0], atol=1e-4))
    assert any(np.isclose(sim.grid.boundaries.z, gap_x.geometry.center[2], atol=1e-4))
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test gaps near pec/pmc
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert any(np.isclose(sim.grid.boundaries.x, -expected_grid_line))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert any(np.isclose(sim.grid.boundaries.z, expected_grid_line))
    assert any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test limited size of layer spec
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.5, 0.2, 0.5),
                    center=(0.5, 0, -0.5),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # pretty much zero size layer spec
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.05, 0.2, 0.05),
                    center=(0.05, 0, 0.05),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=2,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert not any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert not any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # layer spec is outside
    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.pml(),
            z=td.Boundary.pmc(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(0.01, 0.2, 0.01),
                    center=(10.0, 0, 10.0),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    expected_grid_line = box.geometry.size[0] / 2 + (sim_size[0] - box.geometry.size[0]) / 4
    assert not any(np.isclose(sim.grid.boundaries.x, expected_grid_line))
    assert not any(np.isclose(sim.grid.boundaries.x, -expected_grid_line, atol=1e-2))

    expected_grid_line = box.geometry.size[2] / 2 + (sim_size[2] - box.geometry.size[2]) / 4
    assert not any(np.isclose(sim.grid.boundaries.z, expected_grid_line, atol=1e-2))
    assert not any(np.isclose(sim.grid.boundaries.z, -expected_grid_line))

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    # test gap near periodic
    sim = td.Simulation(
        size=sim_size,
        center=(0.025, -0.25, 0),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.pec(),
            z=td.Boundary.periodic(),
        ),
        structures=[box],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()

    assert any(np.isclose(sim.grid.boundaries.x, 0.5, atol=1e-4))
    assert any(np.isclose(sim.grid.boundaries.z, -0.5, atol=1e-4))

    # test a thin strip near periodic
    strip = td.Structure(
        geometry=td.Box(center=(0, 0.5, 0), size=(0.2, 0.05, 0.6)),
        medium=td.PECMedium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pec(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pmc(),
        ),
        structures=[strip],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=0,
                    size=(0.2, td.inf, td.inf),
                    corner_snapping=False,
                    corner_refinement=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=10,
            wavelength=1,
        ),
        run_time=1e-15,
    )

    assert any(np.isclose(sim.grid.boundaries.y, strip.geometry.center[1], atol=1e-4))
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(x=0, ax=ax)
    # sim.plot_grid(x=0, ax=ax)
    # plt.show()


def test_gap_meshing_skip_small_gap():
    """When the gap is very small, make sure it's skipped."""

    f0 = 7e9

    mm = 1000  # Conversion mm to micron
    H = 0.8 * mm  # Substrate thickness
    T = 0.035 * mm  # Metal thickness

    # Resonator dimensions
    MA, MB, MC, MD = (3.9 * mm, 7.1 * mm, 3.1 * mm, 2.3 * mm)
    ME, MF, MG, MH = (0.6 * mm, 0.2 * mm, 1.2 * mm, 0.5 * mm)
    MJ, MK, MM, MN = (4.8 * mm, 0.3 * mm, 0.1 * mm, 0.7 * mm)
    MP, MQ, MR, MS = (0.1 * mm, 0.7 * mm, 0.4 * mm, 0.3 * mm)
    Lsub, Wsub = (2 * MC + MH, 2 * (MH + MK + MB))

    geom_patch = td.Box.from_bounds(
        rmin=(-MA / 2, MH / 2 + MK, 0), rmax=(MA / 2, MH / 2 + MK + MB, T)
    )
    geom_hole1 = td.Box.from_bounds(
        rmin=(-MH / 2 - MN - MF - ME, MH / 2 + MK + MS, 0),
        rmax=(-MH / 2 - MN - MF, MH / 2 + MK + MS + MG, T),
    )
    geom_hole5 = td.Box.from_bounds(
        rmin=(-MA / 2 + 1.5 * MF, MH / 2 + MK + MS + MG + MQ, 0),
        rmax=(-MA / 2 + 1.5 * MF + MM, MH / 2 + MK + MB - MP, T),
    )
    geom_hole6 = geom_hole5.translated(-2 * geom_hole5.center[0], 0, 0)
    geom_hole7 = td.Box.from_bounds(
        rmin=(-MH / 2 - MN, MH / 2 + MK, 0), rmax=(MH / 2 + MN, MH / 2 + MD, T)
    )
    for hole in [geom_hole1, geom_hole5, geom_hole6, geom_hole7]:
        geom_patch -= hole

    x0, y0, z0 = geom_patch.bounding_box.center
    struct_patch = td.Structure(geometry=geom_patch, medium=td.PEC)

    # Add padding
    padding = td.C_0 / f0 / 2
    sim_LX = Lsub + padding
    sim_LY = Wsub + padding
    sim_LZ = H + padding

    # Layer refinement on resonator
    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[struct_patch],
        min_steps_along_axis=1,
        corner_refinement=td.GridRefinement(dl=T, num_cells=2),
        dl_min_from_gap_width=True,
    )

    # Define overall grid spec
    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f0,
        min_steps_per_wvl=12,
        layer_refinement_specs=[lr_spec],
    )

    # Define simulation object
    sim = td.Simulation(
        center=(x0, y0, z0),
        size=(sim_LX, sim_LY, sim_LZ),
        structures=[struct_patch],
        grid_spec=grid_spec,
        run_time=1e-9,
    )
    assert sim.grid_info["min_grid_size"] > 20


def test_gap_meshing_tiny_nearly_parallel():
    """Test that tiny and nearly parallel features are handled properly. That is,
    even though they are ignored, they are still taken into account when removing
    reentry features."""

    sim_size = (1, 1, 1)

    diff = td.Structure(
        geometry=td.PolySlab(
            slab_bounds=[-0.2, 0.2],
            axis=1,
            vertices=[
                (-0.43, -0.43),
                (0.4, -0.35),
                (0 - 1e-14, 0.05 - 1e-4),
                (0 + 1e-14, 0.05 + 1e-4),
                (-0.35, 0.4),
            ],
        ),
        medium=td.PECMedium(),
    )

    sim = td.Simulation(
        size=sim_size,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        structures=[diff],
        grid_spec=td.GridSpec.auto(
            layer_refinement_specs=[
                td.LayerRefinementSpec(
                    axis=1,
                    size=(td.inf, 0.2, td.inf),
                    corner_finder=None,
                    gap_meshing_iters=1,
                    dl_min_from_gap_width=True,
                )
            ],
            min_steps_per_wvl=7,
            wavelength=1,
        ),
        run_time=1e-15,
    )
    # _, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sim.plot(y=0, ax=ax)
    # sim.plot_grid(y=0, ax=ax)
    # plt.show()
    assert sim.grid_info["min_grid_size"] > 0.05


def test_gap_meshing_dl_min_warning():
    """Test that warning is displayed when dl_min_from_gaps is very small relative to lateral grid size."""
    from ..utils import AssertLogStr

    wavelength = 1.0
    # Create a very small gap that will trigger the warning
    # For a grid with min_steps_per_wvl=10 and wavelength=1, the grid size will be around 0.1
    # We need gap_width such that DL_MIN_FROM_GAPS_FRACTION * gap_width < GAP_REFINEMENT_WARNING_THRESH * min_lateral_grid_size
    # gap_width < (0.1 * 0.1) / 0.45 ≈ 0.022
    small_gap_width = 0.01  # Very small gap that will trigger warning

    # Create two parallel PEC strips with a small gap between them
    # Make structures smaller to avoid edge warnings (size 0.3 instead of 0.5 in y-direction)
    strip1 = td.Structure(
        geometry=td.Box(center=(-small_gap_width / 2 - 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    strip2 = td.Structure(
        geometry=td.Box(center=(small_gap_width / 2 + 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    # Create layer refinement spec with gap meshing enabled
    layer_spec = td.LayerRefinementSpec(
        axis=2,  # z-axis
        size=(td.inf, td.inf, 0.2),
        center=(0, 0, 0),
        gap_meshing_iters=1,
        dl_min_from_gap_width=True,
        corner_snapping=False,
        corner_refinement=None,
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength,
        min_steps_per_wvl=10,  # This will create grid cells of ~0.1 size
        layer_refinement_specs=[layer_spec],
    )

    # Test that warning IS displayed for very small gap
    # Use AssertLogStr to filter out edge warnings and only check for our specific warning
    with AssertLogStr("WARNING", contains_str="detected a very small gap width"):
        sim = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1, strip2],
            grid_spec=grid_spec,
            run_time=1e-15,
        )

    # Now test with a larger gap that should NOT trigger warning
    # gap_width should be large enough: gap_width > (GAP_REFINEMENT_WARNING_THRESH * min_lateral_grid_size) / DL_MIN_FROM_GAPS_FRACTION
    # For min_lateral_grid_size ~ 0.1: gap_width > (0.1 * 0.1) / 0.45 ≈ 0.022
    large_gap_width = 0.05  # Large enough to not trigger warning

    strip1_large = td.Structure(
        geometry=td.Box(center=(-large_gap_width / 2 - 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    strip2_large = td.Structure(
        geometry=td.Box(center=(large_gap_width / 2 + 0.05, 0, 0), size=(0.1, 0.3, 0.2)),
        medium=td.PECMedium(),
    )

    # Test that warning is NOT displayed for larger gap
    # Use AssertLogStr to exclude edge warnings and check that our specific warning is not present
    with AssertLogStr("WARNING", excludes_str="detected a very small gap width"):
        sim_large = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1_large, strip2_large],
            grid_spec=grid_spec,
            run_time=1e-15,
        )

    # Test with dl_min_from_gap_width=False - should not warn even with small gap
    layer_spec_no_gap = layer_spec.updated_copy(dl_min_from_gap_width=False)
    grid_spec_no_gap = grid_spec.updated_copy(layer_refinement_specs=[layer_spec_no_gap])

    with AssertLogStr("WARNING", excludes_str="detected a very small gap width"):
        sim_no_gap = td.Simulation(
            size=(1, 1, 0.2),
            structures=[strip1, strip2],
            grid_spec=grid_spec_no_gap,
            run_time=1e-15,
        )
