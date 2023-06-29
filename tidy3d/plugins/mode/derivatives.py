""" Finite-difference derivatives and PML absorption operators expressed as sparse matrices. """

import numpy as np
import scipy.sparse as sp

from ...constants import EPSILON_0, ETA_0


def make_dxf(dls, shape, pmc):
    """Forward derivative in x."""
    Nx, Ny = shape
    if Nx == 1:
        return sp.csr_matrix((Ny, Ny))
    dxf = sp.csr_matrix(sp.diags([-1, 1], [0, 1], shape=(Nx, Nx)))
    if not pmc:
        dxf[0, 0] = 0.0
    dxf = sp.diags(1 / dls).dot(dxf)
    dxf = sp.kron(dxf, sp.eye(Ny))
    return dxf


def make_dxb(dls, shape, pmc):
    """Backward derivative in x."""
    Nx, Ny = shape
    if Nx == 1:
        return sp.csr_matrix((Ny, Ny))
    dxb = sp.csr_matrix(sp.diags([1, -1], [0, -1], shape=(Nx, Nx)))
    if pmc:
        dxb[0, 0] = 2.0
    else:
        dxb[0, 0] = 0.0
    dxb = sp.diags(1 / dls).dot(dxb)
    dxb = sp.kron(dxb, sp.eye(Ny))
    return dxb


def make_dyf(dls, shape, pmc):
    """Forward derivative in y."""
    Nx, Ny = shape
    if Ny == 1:
        return sp.csr_matrix((Nx, Nx))
    dyf = sp.csr_matrix(sp.diags([-1, 1], [0, 1], shape=(Ny, Ny)))
    if not pmc:
        dyf[0, 0] = 0.0
    dyf = sp.diags(1 / dls).dot(dyf)
    dyf = sp.kron(sp.eye(Nx), dyf)
    return dyf


def make_dyb(dls, shape, pmc):
    """Backward derivative in y."""
    Nx, Ny = shape
    if Ny == 1:
        return sp.csr_matrix((Nx, Nx))
    dyb = sp.csr_matrix(sp.diags([1, -1], [0, -1], shape=(Ny, Ny)))
    if pmc:
        dyb[0, 0] = 2.0
    else:
        dyb[0, 0] = 0.0
    dyb = sp.diags(1 / dls).dot(dyb)
    dyb = sp.kron(sp.eye(Nx), dyb)
    return dyb


def create_d_matrices(shape, dlf, dlb, dmin_pmc=(False, False)):
    """Make the derivative matrices without PML. If dmin_pmc is True, the
    'backward' derivative in that dimension will be set to implement PMC
    boundary, otherwise it will be set to PEC."""

    dxf = make_dxf(dlf[0], shape, dmin_pmc[0])
    dxb = make_dxb(dlb[0], shape, dmin_pmc[0])
    dyf = make_dyf(dlf[1], shape, dmin_pmc[1])
    dyb = make_dyb(dlb[1], shape, dmin_pmc[1])

    return (dxf, dxb, dyf, dyb)


# pylint:disable=too-many-locals, too-many-arguments
def create_s_matrices(omega, shape, npml, dlf, dlb, dmin_pml=(True, True)):
    """Makes the 'S-matrices'. When dotted with derivative matrices, they add
    PML. If dmin_pml is set to False, PML will not be applied on the "bottom"
    side of the domain."""

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    nx_pml, ny_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dlf[0], Nx, nx_pml, dmin_pml[0])
    s_vector_x_b = create_sfactor("b", omega, dlb[0], Nx, nx_pml, dmin_pml[0])
    s_vector_y_f = create_sfactor("f", omega, dlf[1], Ny, ny_pml, dmin_pml[1])
    s_vector_y_b = create_sfactor("b", omega, dlb[1], Ny, ny_pml, dmin_pml[1])

    # Fill the 2d space with layers of appropriate s-factors
    sx_f_2d = np.zeros(shape, dtype=np.complex128)
    sx_b_2d = np.zeros(shape, dtype=np.complex128)
    sy_f_2d = np.zeros(shape, dtype=np.complex128)
    sy_b_2d = np.zeros(shape, dtype=np.complex128)

    # Insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(Ny):
        sx_f_2d[:, i] = 1 / s_vector_x_f
        sx_b_2d[:, i] = 1 / s_vector_x_b
    for i in range(Nx):
        sy_f_2d[i, :] = 1 / s_vector_y_f
        sy_b_2d[i, :] = 1 / s_vector_y_b

    # Reshape the 2d s-factors into a 1D s-vecay
    sx_f_vec = sx_f_2d.flatten()
    sx_b_vec = sx_b_2d.flatten()
    sy_f_vec = sy_f_2d.flatten()
    sy_b_vec = sy_b_2d.flatten()

    # Construct the 1D total s-vector into a diagonal matrix
    sx_f = sp.spdiags(sx_f_vec, 0, N, N)
    sx_b = sp.spdiags(sx_b_vec, 0, N, N)
    sy_f = sp.spdiags(sy_f_vec, 0, N, N)
    sy_b = sp.spdiags(sy_b_vec, 0, N, N)

    return sx_f, sx_b, sy_f, sy_b


# pylint:disable=too-many-arguments
def create_sfactor(direction, omega, dls, N, n_pml, dmin_pml):
    """Creates the S-factor cross section needed in the S-matrices"""

    # For no PNL, this should just be identity matrix.
    if n_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # Otherwise, get different profiles for forward and reverse derivatives.
    if direction == "f":
        return create_sfactor_f(omega, dls, N, n_pml, dmin_pml)
    if direction == "b":
        return create_sfactor_b(omega, dls, N, n_pml, dmin_pml)

    raise ValueError(f"Direction value {direction} not recognized")


def create_sfactor_f(omega, dls, N, n_pml, dmin_pml):
    """S-factor profile applied after forward derivative matrix, i.e. applied to H-field
    locations."""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= n_pml - 1 and dmin_pml:
            sfactor_array[i] = s_value(dls[0], (n_pml - i - 0.5) / n_pml, omega)
        elif i >= N - n_pml:
            sfactor_array[i] = s_value(dls[-1], (i - (N - n_pml) + 0.5) / n_pml, omega)
    return sfactor_array


def create_sfactor_b(omega, dls, N, n_pml, dmin_pml):
    """S-factor profile applied after backward derivative matrix, i.e. applied to E-field
    locations."""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i < n_pml and dmin_pml:
            sfactor_array[i] = s_value(dls[0], (n_pml - i) / n_pml, omega)
        elif i > N - n_pml:
            sfactor_array[i] = s_value(dls[-1], (i - (N - n_pml)) / n_pml, omega)
    return sfactor_array


def sig_w(dl, step, sorder=3):
    """Fictional conductivity, note that these values might need tuning"""
    sig_max = 0.8 * (sorder + 1) / (ETA_0 * dl)
    return sig_max * step**sorder


def s_value(dl, step, omega):
    """S-value to use in the S-matrices"""
    # print(step)
    return 1 - 1j * sig_w(dl, step) / (omega * EPSILON_0)
