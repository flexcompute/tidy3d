import tidy3d as td
import numpy as np

from tidy3d.components.source import ScalarField


def test_scalar_field():

    xs = (1, 2, 3)
    ys = (4, 5, 6, 7)
    zs = (5, 6, 7, 8, 9)

    values = np.random.random((3, 4, 5)).tolist()

    s = ScalarField(x=xs, y=ys, z=zs, values=values)
