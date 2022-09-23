"""Tests type definitions."""
import pytest
import tidy3d as td
from tidy3d.components.types import ArrayLike, Complex
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.log import ValidationError
import numpy as np


def _test_validate_array_like():
    class S(Tidy3dBaseModel):
        f: ArrayLike[float, 2]

    s = S(f=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(ValidationError):
        s = S(f=np.array([1.0, 2.0, 3.0]))


def test_schemas():
    class S(Tidy3dBaseModel):
        f: ArrayLike[float, 1]
        c: Complex

    # TODO: unexpected behavior, if list with more than one element, it fails.
    s = S(f=[13], c=1 + 1j)
    S.schema()
