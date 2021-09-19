import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from ..constants import EPSILON_0, ETA_0, C_0, MU_0, fp_eps, pec_val
from .derivatives import create_D_matrices as D_mats
from .derivatives import create_S_matrices as S_mats
from .Mode import Mode

def compute_modes(eps_cross, freq, mesh_step, pml_layers, num_modes=1,
    target_neff=None, symmetries=(0, 0), coords=None):
    """Solve for the modes of a waveguide cross section.
    
    Parameters
    ----------
    eps_cross : array_like or tuple of array_like
        Either a single 2D array defining the relative permittivity in the 
        cross-section, or three 2D arrays defining the permittivity at the Ex, 
        Ey, and Ez locations of the Yee cell, respectively.
    freq : float
        (Hertz) Frequency at which the eigenmodes are computed.
    mesh_step : list or tuple of float
        (micron) Step size in x, y and z. The mesh step in z is currently 
        unused, but it could be needed if numerical dispersion is to be taken 
        into account.
    pml_layers : list or tuple of int
        Number of pml layers in x and y.
    num_modes : int, optional
        Number of modes to be computed.
    target_neff : None or float, optional
        Look for modes closest to target_neff. If ``None``, modes with the
        largest effective index are returned.
    symmetries : array_like, optional
        Array of two integers defining reflection symmetry to be applied
        at the xmin and the ymin locations. Note then that this assumes that
        ``eps_cross`` is only supplied in the quadrants in which it is needed
        and *not* in their symmetric counterparts. Each element can be ``0``
        (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or ``-1`` (odd, i.e. 
        'PEC' symmetry).
    coords : List of array_like or None, optional
        If provided, overrides ``mesh_step``, and must be a list of two arrays
        with size one larger than the corresponding axis of ``eps_cross`.
        Defines a non-uniform Cartesian grid on which the modes are computed.
    
    Returns
    -------
    list of dict
        A list of all the computed modes. Each entry is a dictionary with the 
        real and imaginary part of the effective index of the waveguide and 
        all the E and H field components.
    
    Raises
    ------
    RuntimeError
        Description
    ValueError
        Description
    """
    omega = 2*np.pi*freq
    k0 = omega / C_0

    try:
        if isinstance(eps_cross, np.ndarray):
            eps_xx, eps_yy, eps_zz = [np.copy(eps_cross)]*3
        elif len(eps_cross)==3:
            eps_xx, eps_yy, eps_zz = [np.copy(e) for e in eps_cross]
        else:
            raise ValueError
    except Exception as e:
        printf("Wrong input to mode solver pemittivity!")
        raise(e)

    Nx, Ny = eps_xx.shape
    N = eps_xx.size

    if coords is None:
        coords_x = [mesh_step[0] * np.arange(Nx + 1)]
        coords_y = [mesh_step[1] * np.arange(Ny + 1)]
        coords = [coords_x, coords_y]
    else:
        if coords[0].size != Nx + 1 or coords[1].size != Ny + 1:
            raise ValueError("Mismatch between 'coords' and 'esp_cross' shapes.")

    """ The forward derivative matrices already impose PEC boundary at the 
    xmax and ymax interfaces. Here, we also impose PEC boundaries on the
    xmin and ymin interfaces through the permittivity at those positions,
    unless a PMC symmetry is specifically requested. The PMC symmetry is
    imposed by modifying the backward derivative matrices."""
    dmin_pmc = [False, False]
    if symmetries[0] != 1:
        # PEC at the xmin edge
        eps_yy[0, :] = pec_val
        eps_zz[0, :] = pec_val
    else:
        # Modify the backwards x derivative
        dmin_pmc[0] = True

    if Ny > 1:
        if symmetries[1] != 1:
            # PEC at the ymin edge
            eps_xx[:, 0] = pec_val
            eps_zz[:, 0] = pec_val
        else:
            # Modify the backwards y derivative
            dmin_pmc[1] = True

    # Primal grid steps for E-field derivatives
    dLf = [c[1:] - c[:-1] for c in coords]
    # Dual grid steps for H-field derivatives
    dLtmp = [(dL[:-1] + dL[1:]) / 2 for dL in dLf]
    dLb = [np.hstack((d1[0], d2)) for d1, d2 in zip(dLf, dLtmp)]
    
    # Derivative matrices with PEC boundaries at the far end and optional pmc at the near end
    Dmats = D_mats((Nx, Ny), dLf, dLb, dmin_pmc)

    # PML matrices; do not impose PML on the bottom when symmetry present
    dmin_pml = np.array(symmetries) == 0
    Smats = S_mats(omega, (Nx, Ny), pml_layers, dLf, dLb, dmin_pml)

    # Add the PML on top of the derivatives
    SDmats = [Smat.dot(Dmat) for Smat, Dmat in zip(Smats, Dmats)]

    # Normalize by k0 to match the EM-possible notation
    Dxf, Dxb, Dyf, Dyb = [mat/k0 for mat in SDmats]

    # Compute matrix for diagonalization
    inv_eps_zz = sp.spdiags(1/eps_zz.flatten(), [0], N, N)
    P11 = -Dxf.dot(inv_eps_zz).dot(Dyb)
    P12 = Dxf.dot(inv_eps_zz).dot(Dxb) + sp.eye(N)
    P21 = -Dyf.dot(inv_eps_zz).dot(Dyb) - sp.eye(N)
    P22 = Dyf.dot(inv_eps_zz).dot(Dxb)
    Q11 = -Dxb.dot(Dyf)
    Q12 = Dxb.dot(Dxf) + sp.spdiags(eps_yy.flatten(), [0], N, N)
    Q21 = -Dyb.dot(Dyf) - sp.spdiags(eps_xx.flatten(), [0], N, N)
    Q22 = Dyb.dot(Dxf)

    Pmat = sp.bmat([[P11, P12], [P21, P22]])
    Qmat = sp.bmat([[Q11, Q12], [Q21, Q22]])
    A = Pmat.dot(Qmat)
    
    if target_neff is None:
        n_max = np.sqrt(np.max(eps_cross))
        guess_value = -n_max**2
    else:
        guess_value = -target_neff**2

    vals, vecs = solver_eigs(A, num_modes, guess_value=guess_value)
    vre, vim = -np.real(vals), -np.imag(vals)

    # Sort by descending real part
    sort_inds = np.argsort(vre)[::-1]
    vre = vre[sort_inds]
    vim = vim[sort_inds]
    vecs = vecs[:, sort_inds]

    # Real and imaginary part of the effective index
    neff_tmp = np.sqrt(vre/2 + np.sqrt(vre**2 + vim**2)/2)
    keff = vim/2/(neff_tmp + 1e-10)

    # Correct formula taking numerical dispersion into account
    # neff = 2/mesh_step[2]*np.arcsin((neff_tmp + 1j*keff)*mesh_step[2]/2)
    neff = neff_tmp

    # Field components from eigenvectors
    Ex = vecs[:N, :]
    Ey = vecs[N:, :]

    # Get the other field components; normalize according to CEM
    Hs = -Qmat.dot(vecs)/(neff + 1j*keff)[np.newaxis, :]/ETA_0
    Hx = Hs[:N, :]
    Hy = Hs[N:, :]

    Hz = Dxf.dot(Ey) - Dyf.dot(Ex)
    Ez = inv_eps_zz.dot((Dxb.dot(Hy) - Dyb.dot(Hx)))

    # Store all the information about the modes.
    modes = []
    for im in range(num_modes):
        E = np.array([Ex[:, im].reshape(Nx, Ny),
                        Ey[:, im].reshape(Nx, Ny),
                        Ez[:, im].reshape(Nx, Ny)])
        H = np.array([Hx[:, im].reshape(Nx, Ny),
                        Hy[:, im].reshape(Nx, Ny),
                        Hz[:, im].reshape(Nx, Ny)])
        modes.append(Mode(E, H, neff[im], keff[im]))

    if vals.size == 0:
        raise RuntimeError("Could not find any eigenmodes for this waveguide")

    return modes

def solver_eigs(A, num_modes, guess_value=1.0):
    """ Find ``num_modes`` eigenmodes of ``A`` cloest to ``guess_value``.

    Parameters
    ----------
    A : scipy.sparse matrix
        Square matrix for diagonalization.
    num_modes : int
        Number of eigenmodes to compute.
    guess_value : float, optional
    """

    values, vectors = spl.eigs(A, k=num_modes, sigma=guess_value, tol=fp_eps)
    return values, vectors