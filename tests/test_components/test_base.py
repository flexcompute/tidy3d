"""Tests the base model."""

import pytest
import numpy as np

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel


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


def _test_version(tmp_path):
    """ensure there's a version in simulation"""

    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
    )
    path = str(tmp_path / "simulation.json")
    sim.to_file(path)
    with open(path) as f:
        s = f.read()
        assert '"version": ' in s


def test_deep_copy():
    """Make sure deep copying works as expected with defaults."""
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
    """Make sure updated copying shortcut works as expected with defaults."""
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


def test_updated_copy_path():
    """Make sure updated copying shortcut works as expected with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    index = 12
    structures = (index + 1) * [s]

    sim = td.Simulation(
        size=(4, 4, 4),
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=structures,
    )

    # works as expected
    new_size = (2, 2, 2)
    sim2 = sim.updated_copy(size=new_size, path=f"structures/{index}/geometry")
    assert sim2.structures[index].geometry.size != sim.structures[index].geometry.size
    assert sim2.structures[index].geometry.size == new_size

    # wrong integer index
    with pytest.raises(ValueError):
        sim2 = sim.updated_copy(size=new_size, path="structures/blah/geometry")

    # try with medium for good measure
    new_permittivity = 2.0
    sim3 = sim.updated_copy(permittivity=new_permittivity, path=f"structures/{index}/medium")
    assert sim3.structures[index].medium.permittivity == new_permittivity
    assert sim3.structures[index].medium.permittivity != sim.structures[index].medium.permittivity

    # wrong field name
    with pytest.raises(AttributeError):
        sim3 = sim.updated_copy(
            permittivity=new_permittivity, path=f"structures/{index}/not_a_field"
        )

    # forgot path
    with pytest.raises(ValueError):
        assert sim == sim.updated_copy(permittivity=2.0)

    assert sim.updated_copy(size=(6, 6, 6)) == sim.updated_copy(size=(6, 6, 6), path=None)


def test_equality():
    # test freqs / arraylike
    mnt1 = td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="1")
    mnt2 = td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="1")

    assert mnt1 == mnt2


def test_special_characters_in_name():
    """Test error if special characters are in a component's name."""
    with pytest.raises(ValueError):
        mnt = td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="mnt/flux")
