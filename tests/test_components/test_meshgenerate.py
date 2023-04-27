"""Tests generating meshes."""
import numpy as np
import warnings
import pytest

import tidy3d as td
from tidy3d.constants import fp_eps

from tidy3d.components.grid.mesher import GradedMesher

np.random.seed(4)

MESHER = GradedMesher()
np.random.seed(4)


def validate_dl_multiple_interval(
    dl_list,
    max_scale,
    max_dl_list,
    len_interval_list,
    is_periodic,
):
    """Validate dl_list"""

    # in each interval
    num_intervals = len(len_interval_list)
    right_dl = np.roll(max_dl_list, shift=-1)
    left_dl = np.roll(max_dl_list, shift=1)
    if not is_periodic:
        right_dl[-1] = max_dl_list[-1]
        left_dl[0] = max_dl_list[0]

    left_dl *= max_scale
    right_dl *= max_scale

    for i in range(num_intervals):
        validate_dl_in_interval(
            dl_list[i], max_scale, left_dl[i], right_dl[i], max_dl_list[i], len_interval_list[i]
        )

    dl_list = np.concatenate(dl_list)
    ratio_f = np.all(dl_list[1:] / dl_list[:-1] <= max_scale + fp_eps)
    ratio_b = np.all(dl_list[1:] / dl_list[:-1] >= 1 / (max_scale + fp_eps))
    assert (ratio_f and ratio_b) == True
    assert np.min(dl_list) >= 0.5 * np.min(max_dl_list)


def validate_dl_in_interval(
    dl_list,
    max_scale,
    left_neighbor_dl,
    right_neighbor_dl,
    max_dl,
    len_interval,
):
    """Validate dl_list"""
    ratio_f = np.all(dl_list[1:] / dl_list[:-1] <= max_scale + fp_eps)
    ratio_b = np.all(dl_list[1:] / dl_list[:-1] >= 1 / (max_scale + fp_eps))
    assert (ratio_f and ratio_b) == True

    left_dl = min(max_dl, left_neighbor_dl)
    right_dl = min(max_dl, right_neighbor_dl)

    assert dl_list[0] <= left_dl + fp_eps
    assert dl_list[-1] <= right_dl + fp_eps
    assert np.max(dl_list) <= max_dl + fp_eps
    assert np.isclose(np.sum(dl_list), len_interval, rtol=fp_eps)


def test_uniform_grid_in_interval():
    """Uniform mesh in an interval"""

    for i in range(100):
        len_interval = 10.0 - np.random.random(1)[0]
        # max_scale = 1, but left_dl != right_dl
        left_dl = np.random.random(1)[0]
        right_dl = np.random.random(1)[0]
        max_dl = np.random.random(1)[0]
        max_scale = 1
        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert np.any(dl - dl[0]) == False
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # max_scale > 1, but left_dl = right_dl
        left_dl = np.random.random(1)[0]
        right_dl = left_dl
        max_scale = 1 + np.random.random(1)[0]
        max_dl = left_dl
        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert np.any(dl - dl[0]) == False
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # single pixel
        left_dl = np.random.random(1)[0] + len_interval
        right_dl = np.random.random(1)[0] + len_interval
        max_scale = 1 + np.random.random(1)[0]
        max_dl = left_dl
        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert len(dl) == 1
        assert dl[0] == len_interval


def test_asending_grid_in_interval():
    """Nonuniform mesh in an interval from small to large"""

    # # sufficient remaining part, can be inserted
    len_interval = 1.3
    max_scale = 2
    left_dl = 0.3
    right_dl = 1.0
    max_dl = right_dl

    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # remaining part not sufficient to insert, but will not
    # violate max_scale by repearting 1st step
    len_interval = 1.0
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # scaling
    max_scale = 1.1
    len_interval = 0.2 * (1 - max_scale**3) / (1 - max_scale) + 0.14
    left_dl = 0.2
    right_dl = 1.0
    max_dl = right_dl
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.random(1)[0]
        right_dl = 10
        max_dl = 10

        N_step = 1 + np.log(max_dl / left_dl) / np.log(max_scale)
        N_step *= 0.49 + np.random.random(1)[0] * 0.5
        N_step = int(np.floor(N_step))
        len_interval = left_dl * max_scale * (1 - max_scale**N_step) / (1 - max_scale)
        len_interval *= np.random.random(1)[0]

        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # opposite direction
        left_dl, right_dl = right_dl, left_dl
        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_plateau_grid_in_interval():
    """Nonuniform mesh in an interval from small to large to plateau"""

    # # zero pixel for plateau, still asending
    len_interval = 1.0
    max_scale = 2
    left_dl = 0.3
    right_dl = 10
    max_dl = 0.6
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # # sufficient remaining part, can be inserted
    len_interval = 1.9
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.random(1)[0]
        right_dl = 10
        max_dl = 2 + np.random.random(1)[0] * 2

        N_step = 1 + np.log(max_dl / left_dl) / np.log(max_scale)
        N_step = int(np.floor(N_step))
        len_interval = left_dl * max_scale * (1 - max_scale**N_step) / (1 - max_scale)
        len_interval += max_dl * np.random.randint(1, 100)

        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)
        # print(left_dl*max_scale)
        # print(max_dl)
        # print(dl)

        # opposite direction
        left_dl, right_dl = right_dl, left_dl
        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_plateau_desending_grid_in_interval():
    """Nonuniform mesh in an interval from small to plateau to small"""

    max_scale = 2
    left_dl = 0.1
    right_dl = 0.3
    max_dl = 0.5
    len_interval = 1.51
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.random(1)[0]
        right_dl = np.random.random(1)[0]
        max_dl = 2 + np.random.random(1)[0] * 2

        N_left_step = 1 + np.log(max_dl / left_dl) / np.log(max_scale)
        N_right_step = 1 + np.log(max_dl / right_dl) / np.log(max_scale)
        N_left_step = int(np.floor(N_left_step))
        N_right_step = int(np.floor(N_right_step))
        len_interval = left_dl * max_scale * (1 - max_scale**N_left_step) / (1 - max_scale)
        len_interval += right_dl * max_scale * (1 - max_scale**N_right_step) / (1 - max_scale)
        len_interval += max_dl * (1 + np.random.randint(1, 100))

        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_desending_grid_in_interval():
    """Nonuniform mesh in an interval from small to plateau to small"""

    max_scale = 2
    left_dl = 0.1
    right_dl = 0.3
    max_dl = 1
    len_interval = 3.2
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    max_scale = 2
    left_dl = 0.3
    right_dl = 0.4
    max_dl = 1
    len_interval = 0.8
    dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    # print(dl)

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.random(1)[0]
        right_dl = np.random.random(1)[0]
        max_dl = 2 + np.random.random(1)[0] * 2

        N_left_step = 1 + np.log(max_dl / left_dl) / np.log(max_scale)
        N_right_step = 1 + np.log(max_dl / right_dl) / np.log(max_scale)
        N_left_step = int(np.floor(N_left_step))
        N_right_step = int(np.floor(N_right_step))
        len_interval = left_dl * max_scale * (1 - max_scale**N_left_step) / (1 - max_scale)
        len_interval += right_dl * max_scale * (1 - max_scale**N_right_step) / (1 - max_scale)
        len_interval -= max_dl
        len_interval *= np.random.random(1)[0]

        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_grid_in_interval():
    """Nonuniform mesh in an interval"""

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.randint(1, 10) * np.random.random(1)[0]
        right_dl = np.random.randint(1, 10) * np.random.random(1)[0]
        max_dl = np.random.randint(1, 10) * np.random.random(1)[0]

        len_interval = np.random.randint(1, 100) * np.random.random(1)[0]

        dl = MESHER.make_grid_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_grid_analytic_refinement():

    max_dl_list = np.array([0.5, 0.5, 0.4, 0.1, 0.4])
    len_interval_list = np.array([2.0, 0.5, 0.2, 0.1, 0.3])
    max_scale = 1.5
    periodic = True
    left_dl, right_dl = MESHER.grid_multiple_interval_analy_refinement(
        max_dl_list, len_interval_list, max_scale, periodic
    )
    assert np.all(np.isclose(left_dl[1:], right_dl[:-1])) == True


def test_grid_refinement():

    max_dl_list = np.array([0.5, 0.4, 0.1, 0.4])
    len_interval_list = np.array([0.5, 1.2, 0.1, 1.3])
    max_scale = 1.5
    is_periodic = False
    dl_list = MESHER.make_grid_multiple_intervals(
        max_dl_list, len_interval_list, max_scale, is_periodic
    )
    # print(np.min(np.concatenate(dl_list))/np.min(max_dl_list))

    validate_dl_multiple_interval(
        dl_list,
        max_scale,
        max_dl_list,
        len_interval_list,
        is_periodic,
    )

    num_intervals = 100
    max_shrink = 1
    for i in range(50):
        max_dl_list = np.random.random(num_intervals)
        len_interval_list = np.random.random(num_intervals) * 10
        too_short_ind = len_interval_list < max_dl_list
        len_interval_list[too_short_ind] = max_dl_list[too_short_ind] * (1 + np.random.random(1)[0])
        max_scale = 1.1
        is_periodic = True
        dl_list = MESHER.make_grid_multiple_intervals(
            max_dl_list, len_interval_list, max_scale, is_periodic
        )
        shrink_local = np.min(np.concatenate(dl_list)) / np.min(max_dl_list)
        if shrink_local < max_shrink:
            max_shrink = shrink_local
        validate_dl_multiple_interval(
            dl_list,
            max_scale,
            max_dl_list,
            len_interval_list,
            is_periodic,
        )


WAVELENGTH = 2.9

BOX1 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(2, 2, 2)), medium=td.Medium(permittivity=9)
)
# covers BOX1 along x and y but not z, smaller permittivity
BOX2 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(200, 200, 1)), medium=td.Medium(permittivity=4)
)
# covers BOX1 along x only, smaller permittivity
BOX3 = td.Structure(
    geometry=td.Box(center=(0, 1.5, 0), size=(200, 4, 1)), medium=td.Medium(permittivity=4)
)
# fully covers one edge of BOX1
BOX4 = td.Structure(
    geometry=td.Box(center=(0, 1.01, 0), size=(200, 0.2, 2)), medium=td.Medium(permittivity=2)
)
# box made out of gold
GOLD = td.material_library["Au"]["JohnsonChristy1972"]
BOX5 = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 0.1)), medium=GOLD)
# fully covers BOX5, regular dielectric
BOX6 = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(1, 1, 0.2)), medium=td.Medium(permittivity=2)
)


def test_mesh_structure_covers():
    # Test that the BOX2 permittivity is used along z in the region where it fully covers BOX1
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX1, BOX2],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert sizes[len(sizes) // 2] > 0.1


def test_mesh_structure_partially_covers():
    # Test that the BOX3 permittivity is not used along z as it doesn't fully cover BOX1
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX1, BOX3],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert sizes[len(sizes) // 2] < 0.1


def test_mesh_structure_covers_boundary():
    # Test that there is no grid boundary along y at the BOX1 right side covered by BOX4
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX1, BOX4],
    )
    boundaries = sim.grid.boundaries.to_list[1]
    assert 1.0 not in boundaries


def test_mesh_high_index_background():
    # Test high-index background medium
    sim = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX1, BOX2],
        medium=td.Medium(permittivity=5**2),
    )
    sizes = sim.grid.sizes.to_list[2]
    assert sizes[0] < WAVELENGTH / 50


def test_mesh_high_index_background_override():
    # Test high-index background with override box
    sim = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[
                td.Structure(
                    geometry=td.Box(size=(td.inf, td.inf, td.inf)),
                    medium=td.Medium(permittivity=1),
                ),
                BOX1,
                BOX2,
            ],
        ),
        run_time=1e-13,
        structures=[BOX1, BOX2],
        medium=td.Medium(permittivity=5**2),
    )
    sizes = sim.grid.sizes.to_list[2]
    assert np.isclose(sizes[0], WAVELENGTH / 10)


def test_mesh_direct_override():
    """test td.MeshOverrideStructure"""

    # default override takes effect along one axis
    for axis in range(3):
        dl = [None] * 3
        dl[axis] = 0.05

        override_fine = td.MeshOverrideStructure(
            geometry=td.Box(size=(1, 1, 1)),
            dl=dl,
        )

        sim = td.Simulation(
            size=(3, 3, 3),
            grid_spec=td.GridSpec.auto(
                wavelength=WAVELENGTH,
                override_structures=[override_fine],
            ),
            run_time=1e-13,
            structures=[BOX1],
        )
        assert np.all(sim.grid.sizes.to_list[(axis + 1) % 3] > 0.09)
        assert np.all(sim.grid.sizes.to_list[(axis + 2) % 3] > 0.09)
        sizes = sim.grid.sizes.to_list[axis]
        assert np.isclose(sizes[len(sizes) // 2], 0.05)

    # default override takes effect along two axes
    for axis in range(3):
        dl = [0.05] * 3
        dl[axis] = None

        override_fine = td.MeshOverrideStructure(
            geometry=td.Box(size=(1, 1, 1)),
            dl=dl,
        )

        sim = td.Simulation(
            size=(3, 3, 3),
            grid_spec=td.GridSpec.auto(
                wavelength=WAVELENGTH,
                override_structures=[override_fine],
            ),
            run_time=1e-13,
            structures=[BOX1],
        )
        assert np.all(sim.grid.sizes.to_list[axis] > 0.09)
        sizes = sim.grid.sizes.to_list[(axis + 1) % 3]
        assert np.isclose(sizes[len(sizes) // 2], 0.05)
        sizes = sim.grid.sizes.to_list[(axis + 2) % 3]
        assert np.isclose(sizes[len(sizes) // 2], 0.05)

    # Over all three axes
    override_fine = td.MeshOverrideStructure(
        geometry=td.Box(size=(1, 1, 1)),
        dl=[0.05] * 3,
    )

    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[override_fine],
        ),
        run_time=1e-13,
        structures=[BOX1],
    )
    for axis in range(3):
        sizes = sim.grid.sizes.to_list[axis]
        assert np.isclose(sizes[len(sizes) // 2], 0.05)

    # default override has no effect when coarser than enclosing structure
    override_coarse = override_fine.copy(update={"dl": [0.2] * 3})
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[override_coarse],
        ),
        run_time=1e-13,
        structures=[BOX1],
    )
    for axis in range(3):
        sizes = sim.grid.sizes.to_list[axis]
        assert sizes[len(sizes) // 2] < 0.1

    # however, when enforced, override takes effect again
    override_coarse_enforce = override_coarse.copy(update={"enforce": True})
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[override_coarse_enforce],
        ),
        run_time=1e-13,
        structures=[BOX1],
    )
    for axis in range(3):
        sizes = sim.grid.sizes.to_list[axis]
        assert sizes[len(sizes) // 2] > 0.15


def test_mesh_multiple_direct_override_and_global_min():
    """Test multiple td.MeshOverrideStructure objects.
    Let's consider three override structures: enforced + default + enforced;
    The two enforced structure overlap.

    We also test dl_min here.
    """

    override_enforce1 = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, -1, 1), size=(0.4, 0.4, 3)),
        dl=[None, None, 0.13],
        enforce=True,
    )

    override_enforce2 = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, -1, 0), size=(0.3, 0.3, 1)), dl=[0.15] * 3, enforce=True
    )

    override_default = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 0, 0), size=(1.5, 1.5, 1.5)),
        dl=[0.05] * 3,
    )

    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[override_enforce1, override_default, override_enforce2],
        ),
        run_time=1e-13,
        structures=[BOX1],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert max(sizes) > 0.14
    assert min(sizes) <= 0.05
    assert sizes[-1] > 0.12

    # now let's set a dl_min
    sim = td.Simulation(
        size=(3, 3, 3),
        grid_spec=td.GridSpec.auto(
            wavelength=WAVELENGTH,
            override_structures=[override_enforce1, override_default, override_enforce2],
            dl_min=0.1,
        ),
        run_time=1e-13,
        structures=[BOX1],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert max(sizes) > 0.14
    assert min(sizes) >= 0.08
    assert sizes[-1] > 0.12


def test_mesh_gold_slab():
    # Test meshing of a slab with large negative permittivity
    gold_step = WAVELENGTH / 10 / np.sqrt(np.abs(GOLD.eps_model(td.C_0 / WAVELENGTH).real))

    sim = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX5],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert np.amin(sizes) < gold_step

    # Test that the minimum step is overridden if the gold slab is covered
    # This includes checking that only one of the dielectric box boundaries along z is added
    sim = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[BOX5, BOX6],
    )
    sizes = sim.grid.sizes.to_list[2]
    assert np.amin(sizes) > BOX6.geometry.size[2]


@pytest.mark.timeout(3.0)
def test_mesher_timeout():
    """Test that the mesh generation is fast."""
    np.random.seed(4)
    num_boxes = 500
    box_scale = 5
    sim_size = 5
    n_max = 5
    mediums = [td.Medium(permittivity=n**2) for n in (1 + (n_max - 1) * np.random.rand(100))]

    boxes = []
    for i in range(num_boxes):
        center = sim_size * (np.random.rand(3) - 0.5)
        center[0] = 0
        size = np.abs(box_scale * np.random.randn(3))
        n = 1 + (n_max - 1) * np.random.rand(1)
        box = td.Structure(
            geometry=td.Box(center=center.tolist(), size=size.tolist()),
            medium=mediums[np.random.randint(0, 100)],
        )
        boxes.append(box)

    sim = td.Simulation(
        size=(sim_size,) * 3,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=6),
        run_time=1e-13,
        structures=boxes,
        sources=[
            td.PointDipole(
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
                size=(0, 0, 0),
                polarization="Ex",
            )
        ],
    )

    grid = sim.grid


def test_shapely_strtree_warnings():

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = GradedMesher().parse_structures(
            axis=2,
            structures=[BOX1, BOX2],
            wavelength=1.0,
            min_steps_per_wvl=6,
            dl_min=1.0,
        )


def test_anisotropic_material_meshing():
    """Make sure the largest propagation index defines refinement in all directions."""
    perm_diag = [3, 2, 1]
    cond_diag = [0.2, 0.15, 0.1]

    box = td.Box(
        center=(0, 0, 0),
        size=(1, 2, 3),
    )

    box_iso = td.Structure(
        geometry=box,
        medium=td.Medium(permittivity=perm_diag[0], conductivity=cond_diag[0]),
    )

    box_diag = td.Structure(
        geometry=box,
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=perm_diag[0], conductivity=cond_diag[0]),
            yy=td.Medium(permittivity=perm_diag[1], conductivity=cond_diag[1]),
            zz=td.Medium(permittivity=perm_diag[2], conductivity=cond_diag[2]),
        ),
    )

    box_full = td.Structure(
        geometry=box,
        medium=td.FullyAnisotropicMedium(
            permittivity=np.diag(perm_diag),
            conductivity=np.diag(cond_diag),
        ),
    )

    sim_iso = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[box_iso],
    )

    sim_diag = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[box_diag],
    )

    sim_full = td.Simulation(
        size=(3, 3, 6),
        grid_spec=td.GridSpec.auto(wavelength=WAVELENGTH),
        run_time=1e-13,
        structures=[box_full],
    )

    for dim in range(3):
        assert np.allclose(sim_iso.grid.sizes.to_list[dim], sim_diag.grid.sizes.to_list[dim])
        assert np.allclose(sim_iso.grid.sizes.to_list[dim], sim_full.grid.sizes.to_list[dim])
