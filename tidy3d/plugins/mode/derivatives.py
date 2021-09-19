import numpy as np
import scipy.sparse as sp

from ..constants import EPSILON_0, ETA_0

def make_Dxf(dLs, shape):
    """ Forward derivative in x. """
    Nx, Ny = shape
    if Nx == 1:
        return sp.csr_matrix((Ny, Ny))
    Dxf = sp.diags([-1, 1], [0, 1], shape=(Nx, Nx))
    Dxf = sp.diags(1 / dLs).dot(Dxf)
    Dxf = sp.kron(Dxf, sp.eye(Ny))
    return Dxf

def make_Dxb(dLs, shape, pmc):
    """ Backward derivative in x. """
    Nx, Ny = shape
    if Nx == 1:
        return sp.csr_matrix((Ny, Ny))
    Dxb = sp.diags([1, -1], [0, -1], shape=(Nx, Nx))
    if pmc == True:
        Dxb = sp.csr_matrix(Dxb)
        Dxb[0, 0] = 2.
    Dxb = sp.diags(1 / dLs).dot(Dxb)
    Dxb = sp.kron(Dxb, sp.eye(Ny))
    return Dxb

def make_Dyf(dLs, shape):
    """ Forward derivative in y. """
    Nx, Ny = shape
    if Ny == 1:
        return sp.csr_matrix((Nx, Nx))
    Dyf = sp.diags([-1, 1], [0, 1], shape=(Ny, Ny))
    Dyf = sp.diags(1 / dLs).dot(Dyf)
    Dyf = sp.kron(sp.eye(Nx), Dyf)
    return Dyf

def make_Dyb(dLs, shape, pmc):
    """ Backward derivative in y. """
    Nx, Ny = shape
    if Ny == 1:
        return sp.csr_matrix((Nx, Nx))
    Dyb = sp.diags([1, -1], [0, -1], shape=(Ny, Ny))
    if pmc == True:
        Dyb = sp.csr_matrix(Dyb)
        Dyb[0, 0] = 2.
    Dyb = sp.diags(1 / dLs).dot(Dyb)
    Dyb = sp.kron(sp.eye(Nx), Dyb)
    return Dyb

def create_D_matrices(shape, dLf, dLb, dmin_pmc=(False, False)):
    """Make the derivative matrices without PML. If dmin_pmc is True, the 
    'backward' derivative in that dimension will be set to implement PMC
    symmetry."""

    Dxf = make_Dxf(dLf[0], shape)
    Dxb = make_Dxb(dLb[0], shape, dmin_pmc[0])
    Dyf = make_Dyf(dLf[1], shape)
    Dyb = make_Dyb(dLb[1], shape, dmin_pmc[1])

    return (Dxf, Dxb, Dyf, Dyb)

def create_S_matrices(omega, shape, npml, dLf, dLb, dmin_pml=(True, True)):
    """Makes the 'S-matrices'. When dotted with derivative matrices, they add 
    PML. If dmin_pml is set to False, PML will not be applied on the "bottom"
    side of the domain. """

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    Nx_pml, Ny_pml = npml    

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dLf[0], Nx, Nx_pml, dmin_pml[0])
    s_vector_x_b = create_sfactor('b', omega, dLb[0], Nx, Nx_pml, dmin_pml[0])
    s_vector_y_f = create_sfactor('f', omega, dLf[1], Ny, Ny_pml, dmin_pml[1])
    s_vector_y_b = create_sfactor('b', omega, dLb[1], Ny, Ny_pml, dmin_pml[1])

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(shape, dtype=np.complex128)
    Sx_b_2D = np.zeros(shape, dtype=np.complex128)
    Sy_f_2D = np.zeros(shape, dtype=np.complex128)
    Sy_b_2D = np.zeros(shape, dtype=np.complex128)

    # Insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b
    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()

    # Construct the 1D total s-vector into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b

def create_sfactor(direction, omega, dLs, N, N_pml, dmin_pml):
    """ Creates the S-factor cross section needed in the S-matrices """

    # For no PNL, this should just be identity matrix.
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # Otherwise, get different profiles for forward and reverse derivatives.
    if direction == 'f':
        return create_sfactor_f(omega, dLs, N, N_pml, dmin_pml)
    elif direction == 'b':
        return create_sfactor_b(omega, dLs, N, N_pml, dmin_pml)
    else:
        raise ValueError("Direction value {} not recognized".format(direction))

def create_sfactor_f(omega, dLs, N, N_pml, dmin_pml):
    """ S-factor profile for forward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml and dmin_pml:
            sfactor_array[i] = s_value(dLs[0], (N_pml - i + 0.5)/N_pml, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dLs[-1], (i - (N - N_pml) - 0.5)/N_pml, omega)
    return sfactor_array

def create_sfactor_b(omega, dLs, N, N_pml, dmin_pml):
    """ S-factor profile for backward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml and dmin_pml:
            sfactor_array[i] = s_value(dLs[0], (N_pml - i + 1)/N_pml, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dLs[-1], (i - (N - N_pml) - 1)/N_pml, omega)
    return sfactor_array

def sig_w(dL, step, sorder=3):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = 0.8 * (sorder + 1) / (ETA_0 * dL)
    return sig_max * step**sorder

def s_value(dL, step, omega):
    """ S-value to use in the S-matrices """
    # print(step)
    return 1 - 1j * sig_w(dL, step) / (omega * EPSILON_0)