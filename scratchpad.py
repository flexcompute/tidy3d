#!/usr/bin/env -S poetry run python
# ruff: noqa: F401

from time import perf_counter

import autograd.numpy as anp
import numpy as np
import xarray as xr
import xarray.testing as xrt
from autograd import grad
from autograd.test_util import check_grads

from tidy3d.components.autograd.derivative_utils import integrate_within_bounds
from tidy3d.components.data.data_array import DataArray

rng = np.random.default_rng(456342522)

data = rng.random((40, 50, 10))

coords = {
    "x": np.linspace(0, 1, data.shape[0]),
    "y": np.linspace(0, 1, data.shape[1]),
    "z": np.linspace(0, 1, data.shape[2]),
}


def f(x):
    da = DataArray(x, coords=coords)
    # da = da._ag_interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.8])
    # da = da._ag_interp(x=xr.DataArray([0.5, 0.6], dims=["x"]), y=[0.5, 0.6])
    # da = da._ag_interp(x=[0.5, 0.6], y=[0.5, 0.6])
    # da = da._ag_interp(x=0.5)
    da = da.interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6])
    return anp.mean(da).item()


# da = DataArray(data, coords=coords)
# da_ag = da._ag_interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6])
# da = DataArray(data, coords=coords)
# da_xr = da.interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6])

# xrt.assert_allclose(da_ag, da_xr)
# exit()

# print(f(data))
# print(grad(f)(data))
check_grads(f, modes=["rev"], order=1)(data)
exit()
# print(da._ag_interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6]))
print(da._ag_interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6]))
# print(da.interp(x=xr.DataArray([0.5, 0.6], dims=["index"]), y=[0.5, 0.6]))
exit()

# subset_coords = {
#     "x": np.linspace(0.1, 0.9, 4),
#     "y": np.linspace(0.1, 0.9, 5),
# }

# coords = {dim: da.coords[dim].data for dim in da.coords.dims}
# coords.update(subset_coords)

# updated_data_array = da.assign_coords(coords)
