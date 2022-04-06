""" Coordinate transformations.

The Jacobian of a transformation from coordinates r = (x, y, z) into coordinates
r' = (u, v, w) is defined as J_ij = dr'_i/dr_j. Here, z and w are the propagation axes in the
original and transformed planes, respectively, and the coords are only provided in (x, y) and
transformed to (u, v). The Yee grid positions also have to be taken into account. The Jacobian
for the transformation of eps and E is evaluated at the r' positions of E-field components.
Similarly, the jacobian for mu and H is evaluated at the r' positions of H-field components.
Currently, the half-step offset in w is ignored, which should be a pretty good approximation."""

import numpy as np


def radial_transform(coords, radius, bend_axis):
    """Compute the new coordinates and the Jacobian of a polar coordinate transformation. After
    offsetting the plane such that its center is a distance of ``radius`` away from the center of
    curvature, we have, e.g. for ``bend_axis=='y'``:

        u = (x**2 + z**2)
        v = y
        w = R acos(x / u)

    These are all evaluated at z = 0 below.

    Parameters
    ----------
    coords : tuple
        A tuple of two arrays of size Nx + 1, Ny + 1, respectively.
    radius : float
        Radius of the bend.
    bend_axis : 0 or 1
        Axis normal to the bend plane.

    Returns
    -------
    new_coords: tuple
        Transformed coordinates, same shape as ``coords``.
    jac_e: np.ndarrray
        Jacobian of the transformation at the E-field positions, shape ``(3, 3, Nx * Ny)``.
    jac_h: np.ndarrray
        Jacobian of the transformation at the H-field positions, shape ``(3, 3, Nx * Ny)``.
    k_to_kp: np.ndarray
        A matrix of shape (3, 3) that transforms the k-vector from the original coordinates to the
        transformed ones.
    """

    Nx, Ny = coords[0].size - 1, coords[1].size - 1
    norm_axis = 0 if bend_axis == 1 else 1

    # Center the new coordinates such that the radius is at the center of the plane
    u = coords[0] + (norm_axis == 0) * (radius - coords[0][Nx // 2])
    v = coords[1] + (norm_axis == 1) * (radius - coords[1][Ny // 2])
    new_coords = (u, v)

    """The only nontrivial derivative is dwdz and it only depends on the coordinate in the
    norm_axis direction (orthogonal to both bend_axis and z). We need to compute that derivative
    at the En and Hn positions.
    """
    dwdz_e = radius / new_coords[norm_axis][:-1]
    dwdz_h = radius / (new_coords[norm_axis][:-1] + new_coords[norm_axis][1:]) * 2

    jac_e = np.zeros((3, 3, Nx, Ny))
    jac_e[0, 0, :, :] = 1
    jac_e[1, 1, :, :] = 1
    jac_e[2, 2, :, :] = np.expand_dims(dwdz_e, axis=bend_axis)
    jac_e = jac_e.reshape((3, 3, -1))

    jac_h = np.zeros((3, 3, Nx, Ny))
    jac_h[0, 0, :, :] = 1
    jac_h[1, 1, :, :] = 1
    jac_h[2, 2, :, :] = np.expand_dims(dwdz_h, axis=bend_axis)
    jac_h = jac_h.reshape((3, 3, -1))

    return new_coords, jac_e, jac_h


def angled_transform(coords, angle_theta, angle_phi):
    """Compute the new coordinates and the Jacobian for a transformation that "straightens"
    an angled waveguide such that it is translationally invariant in w. The transformation is
    u = x - tan(angle) * z

    Parameters
    ----------
    coords : tuple
        A tuple of two arrays of size Nx + 1, Ny + 1, respectively.
    angle_theta : float, optional
        (radian) Polar angle from the normal axis.
    angle_phi : float, optional
        (radian) Azimuth angle in the plane orthogonal to the normal axis.

    Returns
    -------
    new_coords: tuple
        Transformed coordinates, same shape as ``coords``.
    jac_e: np.ndarrray
        Jacobian of the transformation at the E-field positions, shape ``(3, 3, Nx * Ny)``.
    jac_h: np.ndarrray
        Jacobian of the transformation at the H-field positions, shape ``(3, 3, Nx * Ny)``.
    """

    Nx, Ny = coords[0].size - 1, coords[1].size - 1

    # The new coordinates are exactly the same at z = 0
    new_coords = (np.copy(c) for c in coords)

    # The only nontrivial derivatives are dudz, dvdz and they are constant everywhere
    jac = np.zeros((3, 3, Nx * Ny))
    jac[0, 0, :] = 1
    jac[1, 1, :] = 1
    jac[2, 2, :] = 1
    jac[0, 2, :] = -np.tan(angle_theta) * np.cos(angle_phi)
    jac[1, 2, :] = -np.tan(angle_theta) * np.sin(angle_phi)

    return new_coords, jac, jac
