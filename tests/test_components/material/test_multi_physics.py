from __future__ import annotations

import copy

import pytest

import tidy3d as td


@pytest.fixture
def dummy_optical():
    return td.Medium(permittivity=1.0)


def test_delegated_attributes_work(dummy_optical):
    mp = td.MultiPhysicsMedium(optical=dummy_optical)

    # delegated names resolve
    assert mp.is_pec is dummy_optical.is_pec
    assert mp.is_pmc is dummy_optical.is_pmc
    assert mp._eps_plot == dummy_optical._eps_plot
    assert mp.viz_spec == dummy_optical.viz_spec

    # deepcopy still succeeds because __deepcopy__ is ignored
    copy.deepcopy(mp)


def test_delegated_attribute_without_optical_raises():
    mp_no_opt = td.MultiPhysicsMedium(optical=None)

    with pytest.raises(AttributeError, match=r"optical medium is 'None'"):
        _ = mp_no_opt.is_pec

    with pytest.raises(AttributeError, match=r"optical medium is 'None'"):
        _ = mp_no_opt.is_pmc


def test_has_cached_props(dummy_optical):
    mp = td.MultiPhysicsMedium(optical=dummy_optical)
    mp._cached_properties


def test_unknown_attribute_error(dummy_optical):
    mp = td.MultiPhysicsMedium(optical=dummy_optical)
    with pytest.raises(AttributeError, match=r"Did you mean to access the attribute of one"):
        _ = mp.not_a_real_attribute
