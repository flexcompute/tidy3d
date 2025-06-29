"""Tests objects shared by multiple components."""

from __future__ import annotations

import pytest

from tidy3d.components.utils import pop_axis_and_swap, unpop_axis_and_swap


@pytest.mark.parametrize("transpose", [True, False])
def test_pop_axis_and_swap(transpose):
    for axis in range(3):
        coords = (1, 2, 3)
        Lz, (Lx, Ly) = pop_axis_and_swap(coords, axis=axis, transpose=transpose)
        _coords = unpop_axis_and_swap(Lz, (Lx, Ly), axis=axis, transpose=transpose)
        assert all(c == _c for (c, _c) in zip(coords, _coords))
        _Lz, (_Lx, _Ly) = pop_axis_and_swap(_coords, axis=axis, transpose=transpose)
        assert Lz == _Lz
        assert Lx == _Lx
        assert Ly == _Ly
