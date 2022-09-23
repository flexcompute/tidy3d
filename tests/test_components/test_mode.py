"""Tests mode objects."""
import pytest
import numpy as np
import tidy3d as td
from tidy3d.log import SetupError


def test_modes():

    m = td.ModeSpec(num_modes=2)
    m = td.ModeSpec(num_modes=1, target_neff=1.0)


def test_bend_axis_not_given():
    with pytest.raises(SetupError):
        _ = td.ModeSpec(bend_radius=1.0, bend_axis=None)


def test_glancing_incidence():
    with pytest.raises(SetupError):
        _ = td.ModeSpec(angle_theta=np.pi / 2)
