#!/usr/bin/env -S poetry run python
# ruff: noqa: F401
import itertools

import autograd.numpy as anp
import numpy as np
import xarray as xr
from autograd.test_util import check_grads

from tidy3d.components.data.data_array import DataArray
from tidy3d.plugins.autograd.primitives import interpn


def check_interp_allclose(coords, values, xi, yi):
    xr_data_array = xr.DataArray(values, coords=coords, dims=("x", "y"))
    xr_interp = xr_data_array.interp(x=xi, y=yi)
    tidy3d_data_array = DataArray(values, coords=coords)
    tidy3d_interp = tidy3d_data_array.interp(x=xi, y=yi)

    my_interp = interpn((coords["x"], coords["y"]), values, (xi, yi))

    print("xarray vs tidy3d:", np.allclose(xr_interp.values, tidy3d_interp.values))
    print("xarray vs interp:", np.allclose(xr_interp.values, my_interp))
    print("tidy3d vs interp:", np.allclose(tidy3d_interp.values, my_interp))


def main():
    coords = {"x": np.linspace(-1, 1, 11), "y": np.linspace(0, 10, 11)}
    values = np.random.random((coords["x"].size, coords["y"].size))
    xi = np.linspace(-0.9, 0.5, 9)
    yi = np.linspace(0.5, 9.5, 13)

    check_interp_allclose(coords, values, xi, yi)

    print("Test grads nearest...", end=" ")
    check_grads(
        lambda x: interpn((coords["x"], coords["y"]), x, (xi, yi), method="nearest"),
        modes=["fwd", "rev"],
        order=3,
    )(values)
    print("ok!")

    print("Test grads linear...", end=" ")
    check_grads(
        lambda x: interpn((coords["x"], coords["y"]), x, (xi, yi), method="linear"),
        modes=["fwd", "rev"],
        order=3,
    )(values)
    print("ok!")


if __name__ == "__main__":
    main()
