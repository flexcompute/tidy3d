"""Tests the base model."""
from typing import Dict, Union, List
import pytest
import numpy as np
import pydantic

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.exceptions import ValidationError, SetupError, Tidy3dKeyError


M = td.Medium()


def test_shallow_copy():
    with pytest.raises(ValueError):
        _ = M.copy(deep=False)


def test_help():
    M.help()


def test_negative_infinity():
    class TestModel(Tidy3dBaseModel):
        z: float

    T = TestModel(z="-Infinity")
    assert np.isneginf(T.z)


def test_comparisons():
    M2 = td.Medium(permittivity=3)
    M > M2
    M < M2
    M <= M2
    M >= M2
    M == M2


def _test_version():
    """ensure there's a version in simulation"""

    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
    )
    path = "tests/tmp/simulation.json"
    sim.to_file("tests/tmp/simulation.json")
    with open(path, "r") as f:
        s = f.read()
        assert '"version": ' in s


def test_deep_copy():
    """Make sure deep copying works as expceted with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    # s_shallow = s.copy(deep=False)
    # with shallow copy, these should be the same objects
    # assert id(s.geometry) == id(s_shallow.geometry)
    # assert id(s.medium) == id(s_shallow.medium)

    s_deep = s.copy(deep=True)

    # with deep copy, these should be different objects
    assert id(s.geometry) != id(s_deep.geometry)
    assert id(s.medium) != id(s_deep.medium)

    # default should be deep
    s_default = s.copy()
    assert id(s.geometry) != id(s_default.geometry)
    assert id(s.medium) != id(s_default.medium)

    # make sure other kwargs work, here we update the geometry to a sphere and shallow copy medium
    # s_kwargs = s.copy(deep=False, update=dict(geometry=Sphere(radius=1.0)))
    # assert id(s.medium) == id(s_kwargs.medium)
    # assert id(s.geometry) != id(s_kwargs.geometry)

    # behavior of modifying attributes
    s_default = s.copy(update=dict(geometry=td.Sphere(radius=1.0)))
    assert id(s.geometry) != id(s_default.geometry)

    # s_shallow = s.copy(deep=False, update=dict(geometry=Sphere(radius=1.0)))
    # assert id(s.geometry) != id(s_shallow.geometry)

    # behavior of modifying attributes of attributes
    new_geometry = s.geometry.copy(update=dict(size=(2, 2, 2)))
    s_default = s.copy(update=dict(geometry=new_geometry))
    assert id(s.geometry) != id(s_default.geometry)

    # s_shallow = s.copy(deep=False)
    # new_geometry = s.geometry.copy(update=dict(size=(2,2,2)))
    # s_shallow = s_shallow.copy(update=dict(geometry=new_geometry))
    # assert id(s.geometry) == id(s_shallow.geometry)


def test_updated_copy():
    """Make sure updated copying shortcut works as expceted with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    b2 = b.updated_copy(size=(2, 2, 2))
    m2 = m.updated_copy(permittivity=2)
    s2 = s.updated_copy(medium=m2, geometry=b2)
    assert s2.geometry == b2
    assert s2.medium == m2
    s3 = s.updated_copy(**{"medium": m2, "geometry": b2})
    assert s3 == s2
