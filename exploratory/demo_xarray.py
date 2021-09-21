import xarray as xr
import numpy as np

from typing import Tuple
from tidy3d import Monitor, Mesh, Box, GeometryObject, Bound


mon = Monitor(
    geometry=Box(size=(1, 0, 1), center=(0, 1, 0)),
    store_values=("E", "H", "flux"),
)

mesh = Mesh(grid_step=(0.1, 0.1, 0.1))


def transpose_bounds(bounds: Bound) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    coord_min, coord_max = bounds
    x_min, y_min, z_min = coord_min
    x_max, y_max, z_max = coord_max
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def get_coords(mesh: Mesh, geo_object: GeometryObject):
    """creates a data container for a GeometryObject"""

    bounds = geo_object.geometry.bounds
    xbounds, ybounds, zbounds = transpose_bounds(bounds)
    dx, dy, dz = mesh.grid_step

    xs = np.arange(xbounds[0], xbounds[1] + 1e-10, dx)
    ys = np.arange(ybounds[0], ybounds[1] + 1e-10, dy)
    zs = np.arange(zbounds[0], zbounds[1] + 1e-10, dz)

    return xs, ys, zs


def initialize_monitor_data(mesh: Mesh, monitor: Monitor) -> xr.Dataset:

    xs, ys, zs = coords = get_coords(mesh, monitor)
    coord_dict = {"x": xs, "y": ys, "z": zs}
    data_shape = (len(xs), len(ys), len(zs))

    data_vars = {}
    for store_val in monitor.store_values:

        numpy_data = np.empty(data_shape, dtype=float)

        data_array = xr.DataArray(data=numpy_data, coords=coord_dict)

        data_vars[store_val] = data_array

    dataset = xr.Dataset(data_vars, attrs={"propagated": False})

    return dataset


dataset = initialize_monitor_data(mesh, mon)
