"""Tests GeometryGroup container."""

import pytest
import pydantic
import numpy as np

import tidy3d as td
from tidy3d.log import ValidationError


def make_geo_group():
    """Make a generic Geometry Group."""
    boxes = [td.Box(size=(1, 1, 1), center=(i, 0, 0)) for i in range(-5, 5)]
    return td.GeometryGroup(geometries=boxes)


def test_initialize():
    """make sure you can construct one."""
    geo_group = make_geo_group()


def test_structure():
    """make sure you can construct a structure using GeometryGroup."""

    geo_group = make_geo_group()
    structure = td.Structure(geometry=geo_group, medium=td.Medium())


def test_methods():
    """Tests the geometry methods of geo group."""

    geo_group = make_geo_group()
    geo_group.inside(0, 1, 2)
    geo_group.inside(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    geo_group.intersections(y=0)
    geo_group.intersects(td.Box(size=(1, 1, 1)))
    rmin, rmax = geo_group.bounds


def test_empty():
    """dont allow empty geometry list."""

    with pytest.raises(ValidationError):
        geo_group = td.GeometryGroup(geometries=[])
