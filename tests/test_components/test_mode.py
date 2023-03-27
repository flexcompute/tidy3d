"""Tests mode objects."""
import pytest
import pydantic
import numpy as np
import tidy3d as td
from tidy3d.exceptions import SetupError


def test_modes():

    m = td.ModeSpec(num_modes=2)
    m = td.ModeSpec(num_modes=1, target_neff=1.0)

    options = [None, "lowest", "highest", "central"]
    for opt in options:
        m = td.ModeSpec(num_modes=3, track_freq=opt)

    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.ModeSpec(num_modes=3, track_freq="middle")
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.ModeSpec(num_modes=3, track_freq=4)


def test_bend_axis_not_given():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(bend_radius=1.0, bend_axis=None)


def test_glancing_incidence():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(angle_theta=np.pi / 2)
