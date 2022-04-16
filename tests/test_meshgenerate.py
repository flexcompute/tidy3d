import numpy as np

import tidy3d as td
from tidy3d.constants import fp_eps

from tidy3d.components import AutoMeshSpec as mesh


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

    # assert(np.min(dl_list)>=0.5*np.min(max_dl_list))


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


def test_uniform_mesh_in_interval():
    """Uniform mesh in an interval"""

    for i in range(100):
        len_interval = 10.0 - np.random.random(1)[0]
        # max_scale = 1, but left_dl != right_dl
        left_dl = np.random.random(1)[0]
        right_dl = np.random.random(1)[0]
        max_dl = np.random.random(1)[0]
        max_scale = 1
        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert np.any(dl - dl[0]) == False
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # max_scale > 1, but left_dl = right_dl
        left_dl = np.random.random(1)[0]
        right_dl = left_dl
        max_scale = 1 + np.random.random(1)[0]
        max_dl = left_dl
        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert np.any(dl - dl[0]) == False
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # single pixel
        left_dl = np.random.random(1)[0] + len_interval
        right_dl = np.random.random(1)[0] + len_interval
        max_scale = 1 + np.random.random(1)[0]
        max_dl = left_dl
        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        assert len(dl) == 1
        assert dl[0] == len_interval


def test_asending_mesh_in_interval():
    """Nonuniform mesh in an interval from small to large"""

    # # sufficient remaining part, can be inserted
    len_interval = 1.3
    max_scale = 2
    left_dl = 0.3
    right_dl = 1.0
    max_dl = right_dl

    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # remaining part not sufficient to insert, but will not
    # violate max_scale by repearting 1st step
    len_interval = 1.0
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # scaling
    max_scale = 1.1
    len_interval = 0.2 * (1 - max_scale**3) / (1 - max_scale) + 0.14
    left_dl = 0.2
    right_dl = 1.0
    max_dl = right_dl
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
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

        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

        # opposite direction
        left_dl, right_dl = right_dl, left_dl
        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_plateau_mesh_in_interval():
    """Nonuniform mesh in an interval from small to large to plateau"""

    # # zero pixel for plateau, still asending
    len_interval = 1.0
    max_scale = 2
    left_dl = 0.3
    right_dl = 10
    max_dl = 0.6
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    # # sufficient remaining part, can be inserted
    len_interval = 1.9
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
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

        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)
        # print(left_dl*max_scale)
        # print(max_dl)
        # print(dl)

        # opposite direction
        left_dl, right_dl = right_dl, left_dl
        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_plateau_desending_mesh_in_interval():
    """Nonuniform mesh in an interval from small to plateau to small"""

    max_scale = 2
    left_dl = 0.1
    right_dl = 0.3
    max_dl = 0.5
    len_interval = 1.51
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
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

        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_asending_desending_mesh_in_interval():
    """Nonuniform mesh in an interval from small to plateau to small"""

    max_scale = 2
    left_dl = 0.1
    right_dl = 0.3
    max_dl = 1
    len_interval = 3.2
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
    validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)

    max_scale = 2
    left_dl = 0.3
    right_dl = 0.4
    max_dl = 1
    len_interval = 0.8
    dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
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

        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_mesh_in_interval():
    """Nonuniform mesh in an interval"""

    # randoms
    for i in range(100):
        max_scale = 1 + np.random.random(1)[0]
        left_dl = np.random.randint(1, 10) * np.random.random(1)[0]
        right_dl = np.random.randint(1, 10) * np.random.random(1)[0]
        max_dl = np.random.randint(1, 10) * np.random.random(1)[0]

        len_interval = np.random.randint(1, 100) * np.random.random(1)[0]

        dl = mesh._make_mesh_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)
        validate_dl_in_interval(dl, max_scale, left_dl, right_dl, max_dl, len_interval)


def test_mesh_analytic_refinement():

    max_dl_list = np.array([0.5, 0.5, 0.4, 0.1, 0.4])
    len_interval_list = np.array([2.0, 0.5, 0.2, 0.1, 0.3])
    max_scale = 1.5
    periodic = True
    left_dl, right_dl = mesh._mesh_multiple_interval_analy_refinement(
        max_dl_list, len_interval_list, max_scale, periodic
    )
    assert np.all(np.isclose(left_dl[1:], right_dl[:-1])) == True


def test_mesh_refinement():

    max_dl_list = np.array([0.5, 0.4, 0.1, 0.4])
    len_interval_list = np.array([0.5, 1.2, 0.1, 1.3])
    max_scale = 1.5
    is_periodic = False
    dl_list = mesh._make_mesh_multiple_intervals(
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
        dl_list = mesh._make_mesh_multiple_intervals(
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


#     # print(max_shrink)
