"""Finite-difference derivatives and PML absorption operators expressed as sparse matrices."""

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


def create_d_matrices(shape, dls, dmin_pmc=(False, False)):
    """Make the derivative matrices without PML. If dmin_pmc is True, the
    'backward' derivative in that dimension will be set to implement PMC
    boundary, otherwise it will be set to PEC."""

    dlf, dlb = dls
    dxf = make_dxf(dlf[0], shape, dmin_pmc[0])
    dxb = make_dxb(dlb[0], shape, dmin_pmc[0])
    dyf = make_dyf(dlf[1], shape, dmin_pmc[1])
    dyb = make_dyb(dlb[1], shape, dmin_pmc[1])

    return (dxf, dxb, dyf, dyb)


def create_s_matrices(omega, shape, npml, dls, eps_tensor, mu_tensor, dmin_pml=(True, True)):
    """Makes the 'S-matrices'. When dotted with derivative matrices, they add
    PML. If dmin_pml is set to False, PML will not be applied on the "bottom"
    side of the domain."""

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    nx_pml, ny_pml = npml
    # forward and backward (primal and dual) grid steps
    dlf, dlb = dls

    # Average speed (relative to C_0) in directions (xminus, xplus, yminus, yplus)
    avg_speed = average_relative_speed(Nx, Ny, npml, eps_tensor, mu_tensor)

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dlf[0], Nx, nx_pml, dmin_pml[0], avg_speed[:2])
    s_vector_x_b = create_sfactor("b", omega, dlb[0], Nx, nx_pml, dmin_pml[0], avg_speed[:2])
    s_vector_y_f = create_sfactor("f", omega, dlf[1], Ny, ny_pml, dmin_pml[1], avg_speed[2:])
    s_vector_y_b = create_sfactor("b", omega, dlb[1], Ny, ny_pml, dmin_pml[1], avg_speed[2:])

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


def average_relative_speed(Nx, Ny, npml, eps_tensor, mu_tensor):
    """Compute the relative speed of light in the four pml regions by averaging the diagonal
    elements of the relative epsilon and mu within the pml region."""

    def relative_mean(tensor):
        """Mean for relative parameters. If an empty array just return 1."""
        if tensor.size == 0:
            return 1.0
        return np.mean(tensor)

    def pml_average_allsides(tensor):
        """Average ``tensor`` in the PML regions on all four sides. Returns the average values in
        order (xminus, xplus, yminus, yplus)."""

        # convert to shape (3, N) and then to (3, Nx, Ny)
        tensor_diag = np.stack([tensor[i, i, :] for i in range(3)])
        tensor_diag = tensor_diag.reshape((3, Nx, Ny))

        avg_xminus = relative_mean(tensor_diag[:, : npml[0], :])
        avg_xplus = relative_mean(tensor_diag[:, Nx - npml[0] + 1 :, :])
        avg_yminus = relative_mean(tensor_diag[:, :, : npml[1]])
        avg_yplus = relative_mean(tensor_diag[:, :, Ny - npml[1] + 1 :])
        return np.array([avg_xminus, avg_xplus, avg_yminus, avg_yplus])

    eps_avg = pml_average_allsides(eps_tensor)
    mu_avg = pml_average_allsides(mu_tensor)
    return 1 / np.sqrt(eps_avg * mu_avg)


def create_sfactor(direction, omega, dls, N, n_pml, dmin_pml, avg_speed):
    """Creates the S-factor cross section needed in the S-matrices"""

    # For no PNL, this should just be identity matrix.
    if n_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # Otherwise, get different profiles for forward and reverse derivatives.
    if direction == "f":
        return create_sfactor_f(omega, dls, N, n_pml, dmin_pml, avg_speed)
    if direction == "b":
        return create_sfactor_b(omega, dls, N, n_pml, dmin_pml, avg_speed)

    raise ValueError(f"Direction value {direction} not recognized")


def create_sfactor_f(omega, dls, N, n_pml, dmin_pml, avg_speed=(1, 1)):
    """S-factor profile applied after forward derivative matrix, i.e. applied to H-field
    locations."""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= n_pml - 1 and dmin_pml:
            sfactor_array[i] = s_value(dls[0], (n_pml - i - 0.5) / n_pml, omega, avg_speed[0])
        elif i >= N - n_pml:
            sfactor_array[i] = s_value(
                dls[-1], (i - (N - n_pml) + 0.5) / n_pml, omega, avg_speed[1]
            )
    return sfactor_array


def create_sfactor_b(omega, dls, N, n_pml, dmin_pml, avg_speed=(1, 1)):
    """S-factor profile applied after backward derivative matrix, i.e. applied to E-field
    locations."""
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i < n_pml and dmin_pml:
            sfactor_array[i] = s_value(dls[0], (n_pml - i) / n_pml, omega, avg_speed[0])
        elif i > N - n_pml:
            sfactor_array[i] = s_value(dls[-1], (i - (N - n_pml)) / n_pml, omega, avg_speed[1])
    return sfactor_array


def s_value(
    dl: float,
    step: int,
    omega: float,
    avg_speed: float,
    sigma_max: float = 2,
    kappa_min: float = 1,
    kappa_max: float = 3,
    order: int = 3,
):
    """S-value to use in the S-matrices.
    We use coordinate stretching formulation such that
        s(x) = kappa(x) + 1j * sigma(x) / (omega * EPSILON_0)
        kappa(x) = kappa_min + (kappa_max - kappa_min) * (x / d) ** order
        sigma(x) = sigma_max * (x / d) ** order
    where x is the position along the PML assumed to begin at 0; d is the total PML thickness;
    order is the polynomial order of the PML profile; and sigma_max is in units of
    ``avg_speed / eta_0 / dl``, with dl the mesh step and ``avg_speed`` the averaged
    speed of the wave relative to C_0 in the PML regions.

    Here x / d is given by ``step``, which has a (half-)integer value depending on whether we're
    doing forward or backward derivatives (E or H field locations along the PML normal direction).

    TODO: expose the parameters to the user.
    """

    if not sigma_max:
        # Old default value
        sigma_max = 0.8 * (order + 1)

    kappa = kappa_min + (kappa_max - kappa_min) * step**order
    sigma = sigma_max * avg_speed / (ETA_0 * dl) * step**order
    return kappa + 1j * sigma / (omega * EPSILON_0)
