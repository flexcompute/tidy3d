"""Tests mode objects."""
import pytest
import pydantic as pydantic
import numpy as np
import tidy3d as td


def test_modes():
    _ = td.ModeSpec(num_modes=2)
    _ = td.ModeSpec(num_modes=1, target_neff=1.0)

    options = [None, "lowest", "highest", "central"]
    for opt in options:
        _ = td.ModeSpec(num_modes=3, track_freq=opt)

    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq="middle")
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq=4)


def test_bend_axis_not_given():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(bend_radius=1.0, bend_axis=None)


def test_glancing_incidence():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(angle_theta=np.pi / 2)


def test_group_index_step_validation():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(group_index_step=1.0)

    ms = td.ModeSpec(group_index_step=True)
    assert ms.group_index_step == td.components.mode.GROUP_INDEX_STEP

    ms = td.ModeSpec(group_index_step=False)
    assert ms.group_index_step is False
    assert not ms.group_index_step > 0
