import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator


def make_interpolator(xs, ys, zs, fn):

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    values = fn(xx, yy, zz)
    values = values.flatten()
    pts = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)
    return LinearNDInterpolator(pts, values)


Nx, Ny, Nz = 13, 14, 15
xs = np.linspace(-10, 10, Nx)
ys = np.linspace(-10, 10, Ny)
zs = np.linspace(-10, 10, Nz)

interpolator = make_interpolator(xs, ys, zs, lambda x, y, z: np.abs(x) + np.abs(y) + np.abs(z))

print(interpolator(1.0, -1.0, 1.0))
# should be around array(3)
