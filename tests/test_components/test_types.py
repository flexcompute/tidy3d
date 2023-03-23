"""Tests type definitions."""
import pytest
import tidy3d as td
from tidy3d.components.types import ArrayLike, Complex, constrained_array
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.exceptions import ValidationError
import numpy as np


def _test_validate_array_like():
    class S(Tidy3dBaseModel):
        f: ArrayLike[float, 2]

    s = S(f=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(pydantic.ValidationError):
        s = S(f=np.array([1.0, 2.0, 3.0]))


def test_schemas():
    class S(Tidy3dBaseModel):
        f: ArrayLike
        ca: constrained_array(ndim=1, dtype=complex)
        c: Complex

    # TODO: unexpected behavior, if list with more than one element, it fails.
    s = S(f=[13], c=1 + 1j, ca=1 + 1j)
    S.schema()


def test_array_like():
    class MyClass(Tidy3dBaseModel):

        a: ArrayLike = None  # can be any array-like thing
        b: constrained_array(ndim=2) = None  # must be 2D
        c: constrained_array(dtype=float) = None  # must be float-like
        d: constrained_array(ndim=1, dtype=complex) = None  # 1D complex
        e: ArrayLike

    my_obj = MyClass(
        a=1.0 + 2j,
        b=np.array([[1.0, 2.0]]),
        c=[1, 3.0],
        d=[1.0],
        e=[[[[1.0]]]],
    )

    assert np.all(my_obj.a == [1.0 + 2j])  # scalars converted to list of len 1
    assert np.all(my_obj.b == [1.0, 2.0])  # numpy arrays converted tolist()
    assert np.all(my_obj.c == [1.0, 3.0])  # converted to float
    assert np.all(my_obj.d == [1.0 + 0.0j])  # converted to complex

    my_obj.json()


def test_hash():
    class MyClass(Tidy3dBaseModel):

        a: ArrayLike
        b: constrained_array(ndim=1)

    c = MyClass(a=[1.0], b=[2.0, 1.0])
    hash(c)
