"""Mode solver for propagating EM modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import EpsSpecType, ModeSolverType, Numpy
from tidy3d.constants import C_0, ETA_0, fp_eps, pec_val

from .derivatives import create_d_matrices as d_mats
from .derivatives import create_s_matrices as s_mats
from .transforms import angled_transform, radial_transform

# Consider vec to be complex if norm(vec.imag)/norm(vec) > TOL_COMPLEX
TOL_COMPLEX = 1e-10
# Tolerance for eigs
TOL_EIGS = fp_eps / 10
# Relative tolerance of complex effective index for finding candidate degenerate modes
TOL_DEGENERATE_CANDIDATE = 1e3 * TOL_EIGS
# Tolerance used to determine whether possible degenerate modes need manual orthogonalization
TOL_NEEDS_ORTHOGONALIZATION = 1e2 * TOL_EIGS
# Tolerance for deciding on the matrix to be diagonal or tensorial
TOL_TENSORIAL = 1e-6
# shift target neff by this value, both rel and abs, whichever results in larger shift
TARGET_SHIFT = 10 * fp_eps
# Preconditioner: "Jacobi" or "Material" based
PRECONDITIONER = "Material"
# Good conductor permittivity cut-off value. Let it be as large as possible so long as not causing overflow in
# double precision. This value is very heuristic.
GOOD_CONDUCTOR_CUT_OFF = 1e70

if TYPE_CHECKING:
    from scipy import sparse as sp

# Consider a material to be good conductor if |ep| (or |mu|) > GOOD_CONDUCTOR_THRESHOLD * |pec_val|
GOOD_CONDUCTOR_THRESHOLD = 0.9


class EigSolver(Tidy3dBaseModel):
    """Interface for computing eigenvalues given permittivity and mode spec.
    It's a collection of static methods.
    """

    @classmethod
    def compute_modes(
        cls,
        eps_cross,
        coords,
        freq,
        mode_spec,
        precision,
        mu_cross=None,
        split_curl_scaling=None,
        symmetry=(0, 0),
        direction="+",
        solver_basis_fields=None,
        plane_center: Optional[tuple[float, float]] = None,
    ) -> tuple[Numpy, Numpy, EpsSpecType]:
        """
        Solve for the modes of a waveguide cross-section.

        Parameters
        ----------
        eps_cross : array_like or tuple of array_like
            Either a single 2D array defining the relative permittivity in the cross-section,
            or nine 2D arrays defining the permittivity at the Ex, Ey, and Ez locations
            of the Yee grid in the order xx, xy, xz, yx, yy, yz, zx, zy, zz.
        coords : List[Numpy]
            Two 1D arrays with each with size one larger than the corresponding axis of
            ``eps_cross``.
            Defines a (potentially non-uniform) Cartesian grid on which the modes are computed.
        freq : float
            (Hertz) Frequency at which the eigenmodes are computed.
        mode_spec : ModeSpec
            ``ModeSpec`` object containing specifications of the mode solver.
        precision : Literal["single", "double"]
            Apply "single" or "double" precision in mode solver.
        mu_cross : array_like or tuple of array_like
            Either a single 2D array defining the relative permeability in the cross-section,
            or nine 2D arrays defining the permeability at the Hx, Hy, and Hz locations
            of the Yee grid in the order xx, xy, xz, yx, yy, yz, zx, zy, zz.
            Set to 1 if `None`.
        split_curl_scaling : tuple of array_like
            Split curl coefficient to Curl E. Three 2D arrays defining the scaling factor
            at the Ex, Ey, and Ez locations of the Yee grid in the order xx, yy, zz.
            Following Benkler's approach, we formulate it as the following:
            1) during mode solver: eps_cross -> eps_corss / scaling, so eigenvector is E * scaling
            2) in postprocessing: apply scaling^-1 to eigenvector to obtain E
        direction : Union["+", "-"]
            Direction of mode propagation.
        solver_basis_fields
            If provided, solve for modes in this basis.
        plane_center
            The center of the mode plane along the tangential axes of the global simulation. Used
            in case of bend modes to offset the coordinates correctly w.r.t. the bend radius, which
            is assumed to refer to the distance from the bend center to the mode plane center.

        Returns
        -------
        Tuple[Numpy, Numpy, str]
            The first array gives the E and H fields for all modes, the second one gives the complex
            effective index. The last variable describes permittivity characterization on the mode
            solver's plane ("diagonal", "tensorial_real", or "tensorial_complex").
        """

        num_modes = mode_spec.num_modes
        bend_radius = mode_spec.bend_radius
        bend_axis = mode_spec.bend_axis
        angle_theta = mode_spec.angle_theta
        angle_phi = mode_spec.angle_phi
        omega = 2 * np.pi * freq
        k0 = omega / C_0
        enable_incidence_matrices = False  # Experimental feature, always off for now

        eps_formated = cls.format_medium_data(eps_cross)
        eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz = eps_formated

        mu_formated = None
        if mu_cross is not None:
            mu_formated = cls.format_medium_data(mu_cross)

        Nxy = eps_xx.shape
        Nx, Ny = Nxy
        N = eps_xx.size

        if len(coords[0]) != Nx + 1 or len(coords[1]) != Ny + 1:
            raise ValueError("Mismatch between 'coords' and 'esp_cross' shapes.")
        new_coords = [np.copy(c) for c in coords]

        """We work with full tensorial epsilon in mu to handle the most general cases that can
        be introduced by coordinate transformations. In the solver, we distinguish the case when
        these tensors are still diagonal, in which case the matrix for diagonalization has shape
        (2N, 2N), and the full tensorial case, in which case it has shape (4N, 4N)."""
        eps_tensor = np.zeros((3, 3, N), dtype=np.complex128)
        mu_tensor = np.zeros((3, 3, N), dtype=np.complex128)
        identity_tensor = np.zeros((3, 3, N), dtype=np.complex128)
        for row, eps_row in enumerate(
            [[eps_xx, eps_xy, eps_xz], [eps_yx, eps_yy, eps_yz], [eps_zx, eps_zy, eps_zz]]
        ):
            identity_tensor[row, row, :] = 1.0
            for col, eps in enumerate(eps_row):
                if split_curl_scaling is not None and col == row:
                    outside_pec = ~np.isclose(split_curl_scaling[col], 0)
                    eps[outside_pec] /= split_curl_scaling[col][outside_pec]

                eps_tensor[row, col, :] = eps.ravel()

        if mu_formated is not None:
            mu_xx, mu_xy, mu_xz, mu_yx, mu_yy, mu_yz, mu_zx, mu_zy, mu_zz = mu_formated
            for row, mu_row in enumerate(
                [[mu_xx, mu_xy, mu_xz], [mu_yx, mu_yy, mu_yz], [mu_zx, mu_zy, mu_zz]]
            ):
                for col, mu in enumerate(mu_row):
                    mu_tensor[row, col, :] = mu.ravel()
        else:
            mu_tensor = np.copy(identity_tensor)

        # Get Jacobian of all coordinate transformations. Initialize as identity (same as mu so far)
        jac_e = np.real(np.copy(identity_tensor))
        jac_h = np.real(np.copy(identity_tensor))

        if np.abs(angle_theta) > 0:
            new_coords, jac_e, jac_h = angled_transform(new_coords, angle_theta, angle_phi)

        if bend_radius is not None:
            if plane_center is None:
                raise ValueError(
                    "When using a nonzero 'bend_radius', the 'plane_center' must also "
                    "be provided to correctly offset the coordinates."
                )
            new_coords, jac_e_tmp, jac_h_tmp = radial_transform(
                new_coords, bend_radius, bend_axis, plane_center
            )
            jac_e = np.einsum("ij...,jp...->ip...", jac_e_tmp, jac_e)
            jac_h = np.einsum("ij...,jp...->ip...", jac_h_tmp, jac_h)

        """We also need to keep track of the transformation of the k-vector. This is
        the eigenvalue of the momentum operator assuming some sort of translational invariance and is
        different from just the transformation of the derivative operator. For example, in a bent
        waveguide, there is strictly speaking no k-vector in the original coordinates as the system
        is not translationally invariant there. However, if we define kz = R k_phi, then the
        effective index approaches that for a straight-waveguide in the limit of infinite radius.
        Since we use w = R phi in the radial_transform, there is nothing else needed in the k transform.
        For the angled_transform, the transformation between k-vectors follows from writing the field as
        E' exp(i k_p w) in transformed coordinates, and identifying this with
        E exp(i k_x x + i k_y y + i k_z z) in the original ones."""
        kxy = np.cos(angle_theta) ** 2
        kz = np.cos(angle_theta) * np.sin(angle_theta)
        kp_to_k = np.array([kxy * np.sin(angle_phi), kxy * np.cos(angle_phi), kz])

        # Transform epsilon and mu
        jac_e_det = np.linalg.det(np.moveaxis(jac_e, [0, 1], [-2, -1]))
        jac_h_det = np.linalg.det(np.moveaxis(jac_h, [0, 1], [-2, -1]))
        eps_tensor = np.einsum("ij...,jp...->ip...", jac_e, eps_tensor)  # J.dot(eps)
        eps_tensor = np.einsum("ij...,pj...->ip...", eps_tensor, jac_e)  # (J.dot(eps)).dot(J.T)
        eps_tensor /= jac_e_det
        mu_tensor = np.einsum("ij...,jp...->ip...", jac_h, mu_tensor)
        mu_tensor = np.einsum("ij...,pj...->ip...", mu_tensor, jac_h)
        mu_tensor /= jac_h_det

        # # Uncomment block to force eps and mu to be translationally invariant into the PML.
        # # This may be important for bends as the jacobian transformation breaks the invariance, but
        # # tests reveal that it has little effect.
        # eps_tensor = cls.make_pml_invariant(Nxy, eps_tensor, mode_spec.num_pml)
        # mu_tensor = cls.make_pml_invariant(Nxy, mu_tensor, mode_spec.num_pml)

        """ Boundaries are imposed through the derivative matrices. The forward derivative matrices
        always impose PEC boundary at the xmax and ymax interfaces, and on the xmin and ymin
        interfaces unless PMC symmetry is present. If so, the PMC boundary is imposed through the
        backward derivative matrices."""
        dmin_pmc = [sym == 1 for sym in symmetry]

        # Primal grid steps for E-field derivatives
        dl_f = [new_cs[1:] - new_cs[:-1] for new_cs in new_coords]
        # Dual grid steps for H-field derivatives
        dl_tmp = [(dl[:-1] + dl[1:]) / 2 for dl in dl_f]
        dl_b = [np.hstack((d1[0], d2)) for d1, d2 in zip(dl_f, dl_tmp)]
        dls = (dl_f, dl_b)

        # Derivative matrices with PEC boundaries by default and optional PMC at the near end
        der_mats_tmp = d_mats(Nxy, dls, dmin_pmc)

        # PML matrices; do not impose PML on the bottom when symmetry present
        dmin_pml = np.array(symmetry) == 0
        pml_mats = s_mats(omega, Nxy, mode_spec.num_pml, dls, eps_tensor, mu_tensor, dmin_pml)

        # Add the PML on top of the derivatives; normalize by k0 to match the EM-possible notation
        der_mats = [Smat.dot(Dmat) / k0 for Smat, Dmat in zip(pml_mats, der_mats_tmp)]

        # Determine initial guess value for the solver in transformed coordinates
        if mode_spec.target_neff is None:
            eps_physical = np.array(eps_cross)
            eps_physical = eps_physical[np.abs(eps_physical) < np.abs(pec_val)]
            n_max = np.sqrt(np.max(np.abs(eps_physical)))
            target = n_max
        else:
            target = mode_spec.target_neff
        target_neff_p = target / np.linalg.norm(kp_to_k)

        # shift target_neff slightly to avoid cases where the shiftted matrix is exactly singular
        if abs(TARGET_SHIFT) > abs(target_neff_p * TARGET_SHIFT):
            target_neff_p += TARGET_SHIFT
        else:
            target_neff_p *= 1 + TARGET_SHIFT

        # preprocess solver_basis_fields
        basis_E = None
        if solver_basis_fields is not None:
            basis_E = solver_basis_fields[:3, ...]
            try:
                basis_E = basis_E.reshape((3, Nx * Ny, num_modes))
            except ValueError:
                raise ValueError(
                    "Shape mismatch between 'basis_fields' and requested mode data. "
                    "Make sure the mode solvers are set up the same, and that the "
                    "basis mode solver data has 'colocate=False'."
                ) from None
            if split_curl_scaling is not None:
                basis_E = cls.split_curl_field_postprocess_inverse(split_curl_scaling, basis_E)
            jac_e_inv = np.moveaxis(
                np.linalg.inv(np.moveaxis(jac_e, [0, 1], [-2, -1])), [-2, -1], [0, 1]
            )
            basis_E = np.sum(jac_e_inv[..., None] * basis_E[:, None, ...], axis=0)

        # Solve for the modes
        E, H, neff, keff, eps_spec = cls.solver_em(
            Nx,
            Ny,
            eps_tensor,
            mu_tensor,
            der_mats,
            num_modes,
            target_neff_p,
            precision,
            direction,
            enable_incidence_matrices,
            basis_E=basis_E,
            dls=dls,
            dmin_pmc=dmin_pmc,
        )

        # Transform back to original axes, E = J^T E'
        E = np.sum(jac_e[..., None] * E[:, None, ...], axis=0)
        if split_curl_scaling is not None:
            E = cls.split_curl_field_postprocess(split_curl_scaling, E)
        E = E.reshape((3, Nx, Ny, 1, num_modes))
        H = np.sum(jac_h[..., None] * H[:, None, ...], axis=0)
        H = H.reshape((3, Nx, Ny, 1, num_modes))

        fields = np.stack((E, H), axis=0)

        neff = neff * np.linalg.norm(kp_to_k)
        keff = keff * np.linalg.norm(kp_to_k)

        if mode_spec.precision == "single":
            # Recast to single precision which may have changed due to earlier manipulations
            fields = fields.astype(np.complex64)

        return fields, neff + 1j * keff, eps_spec

    @classmethod
    def solver_em(
        cls,
        Nx,
        Ny,
        eps_tensor,
        mu_tensor,
        der_mats,
        num_modes,
        neff_guess,
        mat_precision,
        direction,
        enable_incidence_matrices,
        basis_E,
        dls,
        dmin_pmc=None,
    ):
        """Solve for the electromagnetic modes of a system defined by in-plane permittivity and
        permeability and assuming translational invariance in the normal direction.

        Parameters
        ----------
        Nx : int
            Number of grids along x-direction.
        Ny : int
            Number of grids along y-direction.
        eps_tensor : np.ndarray
            Shape (3, 3, N), the permittivity tensor at every point in the plane.
        mu_tensor : np.ndarray
            Shape (3, 3, N), the permittivity tensor at every point in the plane.
        der_mats : List[scipy.sparse.csr_matrix]
            The sparse derivative matrices dxf, dxb, dyf, dyb, including the PML.
        num_modes : int
            Number of modes to solve for.
        neff_guess : float
            Initial guess for the effective index.
        mat_precision : Union['single', 'double']
            Single or double-point precision in eigensolver.
        direction : Union["+", "-"]
            Direction of mode propagation.
        basis_E: np.ndarray
            Basis for mode solving.
        dls: Tuple[List[np.ndarray], List[np.ndarray]]
            Primal and dual grid steps along each of the two tangential dimensions.
        dmin_pmc: List[bool]
            List of booleans indicating whether to apply PMC at the near end of the grid.

        Returns
        -------
        E : np.ndarray
            Electric field of the eigenmodes, shape (3, N, num_modes).
        H : np.ndarray
            Magnetic field of the eigenmodes, shape (3, N, num_modes).
        neff : np.ndarray
            Real part of the effective index, shape (num_modes, ).
        keff : np.ndarray
            Imaginary part of the effective index, shape (num_modes, ).
        eps_spec : Union["diagonal", "tensorial_real", "tensorial_complex"]
            Permittivity characterization on the mode solver's plane.
        """

        # In the matrices P and Q below, they contain terms ``epsilon_parallel`` or
        # ``mu_parallel``, and also a term proportional to 1/(k0 * dl)**2. To make sure
        # that permittivity of good conductor is visible in low-frequency/high resolution, pec_val should be
        # scaled by a factor max(1, max(1/k0 dl) **2).
        pec_scaling = max(1, max([np.max(abs(f)) for f in der_mats]) ** 2)
        pec_scaled_val = min(GOOD_CONDUCTOR_CUT_OFF, pec_scaling * abs(pec_val))

        # use a high-conductivity model for locations associated with a good conductor
        def conductivity_model_for_good_conductor(
            eps, threshold=GOOD_CONDUCTOR_THRESHOLD * pec_val
        ):
            """Entries associated with 'eps' are converted to a high-conductivity model."""
            eps = eps.astype(complex)
            eps[np.abs(eps) >= abs(threshold)] = 1 + 1j * pec_scaled_val
            return eps

        eps_tensor = conductivity_model_for_good_conductor(eps_tensor)
        mu_tensor = conductivity_model_for_good_conductor(mu_tensor)

        # Determine if ``eps`` and ``mu`` are diagonal or tensorial
        off_diagonals = (np.ones((3, 3)) - np.eye(3)).astype(bool)
        eps_offd = np.abs(eps_tensor[off_diagonals])
        mu_offd = np.abs(mu_tensor[off_diagonals])
        is_tensorial = np.any(eps_offd > TOL_TENSORIAL) or np.any(mu_offd > TOL_TENSORIAL)

        # Determine if ``eps`` and ``mu`` represent reciprocal media
        is_reciprocal = True
        if is_tensorial:
            is_reciprocal = cls._check_reciprocity(
                eps_tensor, TOL_TENSORIAL
            ) and cls._check_reciprocity(mu_tensor, TOL_TENSORIAL)

        # initial vector for eigensolver in correct data type
        vec_init = cls.set_initial_vec(Nx, Ny, is_tensorial=is_tensorial)

        # call solver: base kwargs shared by diagonal and tensorial solvers
        base_kwargs = {
            "eps": eps_tensor,
            "mu": mu_tensor,
            "der_mats": der_mats,
            "num_modes": num_modes,
            "neff_guess": neff_guess,
            "vec_init": vec_init,
            "mat_precision": mat_precision,
        }

        if basis_E is not None and is_tensorial:
            raise RuntimeError(
                "Tensorial eps not yet supported in relative mode solver "
                "(with basis fields provided)."
            )

        # Determine if epsilon has complex values (used to select real vs complex tensorial solver)
        is_eps_complex = cls.isinstance_complex(eps_tensor)

        if not is_tensorial:
            eps_spec = "diagonal"
            E, H, neff, keff = cls.solver_diagonal(
                **base_kwargs,
                enable_incidence_matrices=enable_incidence_matrices,
                basis_E=basis_E,
            )
            if direction == "-":
                H[0] *= -1
                H[1] *= -1
                E[2] *= -1

        elif not is_eps_complex:
            eps_spec = "tensorial_real"
            E, H, neff, keff = cls.solver_tensorial(
                **base_kwargs,
                direction="+",
                dls=dls,
                Nxy=(Nx, Ny),
                dmin_pmc=dmin_pmc,
            )
            if direction == "-":
                E = np.conj(E)
                H = -np.conj(H)

        else:
            eps_spec = "tensorial_complex"
            E, H, neff, keff = cls.solver_tensorial(
                **base_kwargs,
                direction=direction,
                dls=dls,
                Nxy=(Nx, Ny),
                dmin_pmc=dmin_pmc,
            )

        # We only can orthogonalize reciprocal modes in this manner.
        if is_reciprocal:
            dl_f, dl_b = dls
            # Normalize all modes so self-dot equals 1.0 and find any self-orthogonal modes
            all_modes = list(range(num_modes))
            original_shape = E.shape
            E = E.reshape((3, Nx, Ny, num_modes))
            H = H.reshape((3, Nx, Ny, num_modes))
            E, H, self_orthogonal_modes = cls._normalize_modes(
                E, H, dl_f, dl_b, all_modes, TOL_NEEDS_ORTHOGONALIZATION
            )
            # Identify and orthogonalize degenerate modes, excluding self-orthogonal modes
            E, H = cls._orthogonalize_degenerate_modes(
                E, H, neff, keff, dl_f, dl_b, self_orthogonal_modes
            )
            E = E.reshape(original_shape)
            H = H.reshape(original_shape)

        return E, H, neff, keff, eps_spec

    @classmethod
    def solver_diagonal(
        cls,
        eps,
        mu,
        der_mats,
        num_modes,
        neff_guess,
        vec_init,
        mat_precision,
        enable_incidence_matrices,
        basis_E,
    ):
        """EM eigenmode solver assuming ``eps`` and ``mu`` are diagonal everywhere."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        # code associated with these options is included below in case it's useful in the future
        enable_preconditioner = False
        analyze_conditioning = False
        _threshold = 0.9 * np.abs(pec_val)

        def incidence_matrix_for_pec(eps_vec, threshold=_threshold):
            """Incidence matrix indicating non-PEC entries associated with 'eps_vec'."""
            nnz = eps_vec[np.abs(eps_vec) < threshold]
            eps_nz = eps_vec.copy()
            eps_nz[np.abs(eps_vec) >= threshold] = 0
            rows = np.arange(0, len(nnz))
            cols = np.argwhere(eps_nz).flatten()
            dnz = sp.csr_matrix(([1] * len(nnz), (rows, cols)), shape=(len(rows), len(eps_vec)))
            return dnz

        mode_solver_type = "diagonal"
        N = eps.shape[-1]

        # Unpack eps, mu and derivatives
        eps_xx = eps[0, 0, :]
        eps_yy = eps[1, 1, :]
        eps_zz = eps[2, 2, :]
        mu_xx = mu[0, 0, :]
        mu_yy = mu[1, 1, :]
        mu_zz = mu[2, 2, :]
        dxf, dxb, dyf, dyb = der_mats

        if any(
            cls.mode_plane_contain_good_conductor(i)
            for i in [eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz]
        ):
            enable_preconditioner = True

        # Compute the matrix for diagonalization
        inv_eps_zz = sp.spdiags(1 / eps_zz, [0], N, N)
        inv_mu_zz = sp.spdiags(1 / mu_zz, [0], N, N)

        if enable_incidence_matrices:
            dnz_xx, dnz_yy, dnz_zz = (incidence_matrix_for_pec(i) for i in [eps_xx, eps_yy, eps_zz])
            dnz = sp.block_diag((dnz_xx, dnz_yy), format="csr")
            inv_eps_zz = (dnz_zz.T) * dnz_zz * inv_eps_zz * (dnz_zz.T) * dnz_zz

        # P = p_mu + p_partial
        # Q = q_ep + q_partial
        # Note that p_partial @ q_partial = 0, so that PQ = p_mu @ Q + p_partial @ q_ep
        p_mu = sp.bmat(
            [[None, sp.spdiags(mu_yy, [0], N, N)], [sp.spdiags(-mu_xx, [0], N, N), None]]
        )
        p_partial = sp.bmat(
            [
                [-dxf.dot(inv_eps_zz).dot(dyb), dxf.dot(inv_eps_zz).dot(dxb)],
                [-dyf.dot(inv_eps_zz).dot(dyb), dyf.dot(inv_eps_zz).dot(dxb)],
            ]
        )
        q_ep = sp.bmat(
            [[None, sp.spdiags(eps_yy, [0], N, N)], [sp.spdiags(-eps_xx, [0], N, N), None]]
        )
        q_partial = sp.bmat(
            [
                [-dxb.dot(inv_mu_zz).dot(dyf), dxb.dot(inv_mu_zz).dot(dxf)],
                [-dyb.dot(inv_mu_zz).dot(dyf), dyb.dot(inv_mu_zz).dot(dxf)],
            ]
        )

        # pmat = p_mu + p_partial  # no need to assemble pmat, as it is not used anywhere
        qmat = q_ep + q_partial
        mat = p_mu @ qmat + p_partial @ q_ep

        # Cast matrix to target data type
        mat_dtype = cls.matrix_data_type(eps, mu, der_mats, mat_precision, is_tensorial=False)
        mat = cls.type_conversion(mat, mat_dtype)

        # Casting starting vector to target data type
        vec_init = cls.type_conversion(vec_init, mat_dtype)

        # Starting eigenvalue guess in target data type
        eig_guess = cls.type_conversion(np.array([-(neff_guess**2)]), mat_dtype)[0]

        if enable_incidence_matrices:
            mat = dnz * mat * dnz.T
            vec_init = dnz * vec_init

        # Denote the original eigenvalue problem as Ax = lambda x,
        # with left and right preconditioners, we solve for the following eigenvalue problem,
        # L A R y = lambda LR y, where x = R y
        precon_left = None
        precon_right = None
        generalized_M = None  # matrix in the generalized eigenvalue problem
        if enable_preconditioner:
            if PRECONDITIONER == "Jacobi":
                precon_right = sp.diags(1 / mat.diagonal())

            elif PRECONDITIONER == "Material":

                def conditional_inverted_vec(eps_vec, threshold=1):
                    """Returns a diagonal sparse matrix whose i-th element in the diagonal
                    is |eps_i|^-1 if |eps_i|>threshold, and |eps_i| otherwise.
                    """
                    abs_vec = np.abs(eps_vec)
                    return sp.spdiags(
                        np.where(abs_vec > threshold, 1.0 / abs_vec, abs_vec), [0], N, N
                    )

                precon_left = sp.bmat(
                    [
                        [conditional_inverted_vec(mu_yy), None],
                        [None, conditional_inverted_vec(mu_xx)],
                    ]
                )
                precon_right = sp.bmat(
                    [
                        [conditional_inverted_vec(eps_xx), None],
                        [None, conditional_inverted_vec(eps_yy)],
                    ]
                )
            generalized_M = precon_right
            mat = mat @ precon_right
            if precon_left is not None:
                generalized_M = precon_left @ generalized_M
                mat = precon_left @ mat

        # Trim small values in single precision case
        if mat_precision == "single":
            cls.trim_small_values(mat, tol=fp_eps)

        if analyze_conditioning:
            aca = mat.conjugate().T * mat
            aac = mat * mat.conjugate().T
            diff = aca - aac
            print(
                f"inf-norm: A*A: {spl.norm(aca, ord=np.inf)}, AA*: {spl.norm(aac, ord=np.inf)}, nonnormality: {spl.norm(diff, ord=np.inf)}, relative nonnormality: {spl.norm(diff, ord=np.inf) / spl.norm(aca, ord=np.inf)}"
            )
            print(
                f"fro-norm: A*A: {spl.norm(aca, ord='fro')}, AA*: {spl.norm(aac, ord='fro')}, nonnormality: {spl.norm(diff, ord='fro')}, relative nonnormality: {spl.norm(diff, ord='fro') / spl.norm(aca, ord='fro')}"
            )

        # preprocess basis modes
        basis_vecs = None
        if basis_E is not None:
            basis_Ex = basis_E[0, ...]
            basis_Ey = basis_E[1, ...]
            basis_vecs = np.concatenate((basis_Ex, basis_Ey), axis=0)

            # if enable_preconditioner:
            #    basis_vecs = (1 / precon) * basis_vecs

            # if enable_incidence_matrices:
            #    basis_vecs = dnz * basis_vecs

        # Call the eigensolver. The eigenvalues are -(neff + 1j * keff)**2
        if basis_E is None:
            vals, vecs = cls.solver_eigs(
                mat,
                num_modes,
                vec_init,
                guess_value=eig_guess,
                mode_solver_type=mode_solver_type,
                M=generalized_M,
            )
        else:
            vals, vecs = cls.solver_eigs_relative(
                mat,
                num_modes,
                vec_init,
                guess_value=eig_guess,
                mode_solver_type=mode_solver_type,
                M=generalized_M,
                basis_vecs=basis_vecs,
            )
        neff, keff = cls.eigs_to_effective_index(vals, mode_solver_type)

        # Sort by descending neff
        sort_inds = np.argsort(neff)[::-1]
        neff = neff[sort_inds]
        keff = keff[sort_inds]

        E, H = None, None
        if basis_E is None:
            if precon_right is not None:
                vecs = precon_right * vecs

            if enable_incidence_matrices:
                vecs = dnz.T * vecs

        vecs = vecs[:, sort_inds]

        # Field components from eigenvectors
        Ex = vecs[:N, :]
        Ey = vecs[N:, :]

        # Get the other field components
        h_field = qmat.dot(vecs)
        Hx = h_field[:N, :] / (1j * neff - keff)
        Hy = h_field[N:, :] / (1j * neff - keff)
        Hz = inv_mu_zz.dot(dxf.dot(Ey) - dyf.dot(Ex))
        # Ez = inv_eps_zz.dot(dxb.dot(Hy) - dyb.dot(Hx))

        # Ez = -inv_eps_zz * div^H J H_xy, while Hxy = vals^-1 * qmat * Exy;
        # Note that div^H J q_partial = 0, so Ez = -vals^-1 inv_eps_zz * div^H J q_ep Exy
        h_partial_field = q_ep.dot(vecs) / (1j * neff - keff)
        Ez = inv_eps_zz.dot(dxb.dot(h_partial_field[N:, :]) - dyb.dot(h_partial_field[:N, :]))

        # Bundle up
        E = np.stack((Ex, Ey, Ez), axis=0)
        H = np.stack((Hx, Hy, Hz), axis=0)

        # Return to standard H field units (see CEM notes for H normalization used in solver)
        H *= -1j / ETA_0

        return E, H, neff, keff

    @classmethod
    def matrix_data_type(cls, eps, mu, der_mats, mat_precision, is_tensorial):
        """Determine data type that should be used for the matrix for diagonalization."""
        mat_dtype = np.float32
        # In tensorial case, even though the matrix can be real, the
        # expected eigenvalue is purely imaginary. So for now we enforce
        # the matrix to be complex type so that it will look for the right eigenvalues.
        if is_tensorial:
            mat_dtype = np.complex128 if mat_precision == "double" else np.complex64
        else:
            # 1) check if complex or not
            complex_solver = (
                cls.isinstance_complex(eps)
                or cls.isinstance_complex(mu)
                or np.any([cls.isinstance_complex(f) for f in der_mats])
            )
            # 2) determine precision
            if complex_solver:
                mat_dtype = np.complex128 if mat_precision == "double" else np.complex64
            else:
                if mat_precision == "double":
                    mat_dtype = np.float64

        return mat_dtype

    @classmethod
    def trim_small_values(cls, mat: sp.csr_matrix, tol: float) -> sp.csr_matrix:
        """Eliminate elements of matrix ``mat`` for which ``abs(element) / abs(max_element) < tol``,
        or ``np.abs(mat_data) < tol``. This operates in-place on mat so there is no return.
        """
        max_element = np.amax(np.abs(mat))
        mat.data *= np.logical_or(np.abs(mat.data) / max_element > tol, np.abs(mat.data) > tol)
        mat.eliminate_zeros()
        return mat

    @staticmethod
    def _check_reciprocity(material_tensor: np.ndarray, tol: float) -> bool:
        """Check if material tensor is symmetric (reciprocal): tensor[i,j] ≈ tensor[j,i]."""
        diff = np.abs(material_tensor - material_tensor.transpose(1, 0, 2))
        max_error = np.max(diff)
        return max_error <= tol

    @classmethod
    def solver_tensorial(
        cls,
        eps,
        mu,
        der_mats,
        num_modes,
        neff_guess,
        vec_init,
        mat_precision,
        direction,
        dls,
        Nxy=None,
        dmin_pmc=None,
    ):
        """EM eigenmode solver assuming ``eps`` or ``mu`` have off-diagonal elements."""
        import scipy.sparse as sp

        mode_solver_type = "tensorial"
        N = eps.shape[-1]
        dxf, dxb, dyf, dyb = der_mats

        # Compute all blocks of the matrix for diagonalization
        inv_eps_zz = sp.spdiags(1 / eps[2, 2, :], [0], N, N)
        inv_mu_zz = sp.spdiags(1 / mu[2, 2, :], [0], N, N)
        axax = -dxf.dot(sp.spdiags(eps[2, 0, :] / eps[2, 2, :], [0], N, N)) - sp.spdiags(
            mu[1, 2, :] / mu[2, 2, :], [0], N, N
        ).dot(dyf)
        axay = -dxf.dot(sp.spdiags(eps[2, 1, :] / eps[2, 2, :], [0], N, N)) + sp.spdiags(
            mu[1, 2, :] / mu[2, 2, :], [0], N, N
        ).dot(dxf)
        axbx = -dxf.dot(inv_eps_zz).dot(dyb) + sp.spdiags(
            mu[1, 0, :] - mu[1, 2, :] * mu[2, 0, :] / mu[2, 2, :], [0], N, N
        )
        axby = dxf.dot(inv_eps_zz).dot(dxb) + sp.spdiags(
            mu[1, 1, :] - mu[1, 2, :] * mu[2, 1, :] / mu[2, 2, :], [0], N, N
        )
        ayax = -dyf.dot(sp.spdiags(eps[2, 0, :] / eps[2, 2, :], [0], N, N)) + sp.spdiags(
            mu[0, 2, :] / mu[2, 2, :], [0], N, N
        ).dot(dyf)
        ayay = -dyf.dot(sp.spdiags(eps[2, 1, :] / eps[2, 2, :], [0], N, N)) - sp.spdiags(
            mu[0, 2, :] / mu[2, 2, :], [0], N, N
        ).dot(dxf)
        aybx = -dyf.dot(inv_eps_zz).dot(dyb) + sp.spdiags(
            -mu[0, 0, :] + mu[0, 2, :] * mu[2, 0, :] / mu[2, 2, :], [0], N, N
        )
        ayby = dyf.dot(inv_eps_zz).dot(dxb) + sp.spdiags(
            -mu[0, 1, :] + mu[0, 2, :] * mu[2, 1, :] / mu[2, 2, :], [0], N, N
        )
        bxbx = -dxb.dot(sp.spdiags(mu[2, 0, :] / mu[2, 2, :], [0], N, N)) - sp.spdiags(
            eps[1, 2, :] / eps[2, 2, :], [0], N, N
        ).dot(dyb)
        bxby = -dxb.dot(sp.spdiags(mu[2, 1, :] / mu[2, 2, :], [0], N, N)) + sp.spdiags(
            eps[1, 2, :] / eps[2, 2, :], [0], N, N
        ).dot(dxb)
        bxax = -dxb.dot(inv_mu_zz).dot(dyf) + sp.spdiags(
            eps[1, 0, :] - eps[1, 2, :] * eps[2, 0, :] / eps[2, 2, :], [0], N, N
        )
        bxay = dxb.dot(inv_mu_zz).dot(dxf) + sp.spdiags(
            eps[1, 1, :] - eps[1, 2, :] * eps[2, 1, :] / eps[2, 2, :], [0], N, N
        )
        bybx = -dyb.dot(sp.spdiags(mu[2, 0, :] / mu[2, 2, :], [0], N, N)) + sp.spdiags(
            eps[0, 2, :] / eps[2, 2, :], [0], N, N
        ).dot(dyb)
        byby = -dyb.dot(sp.spdiags(mu[2, 1, :] / mu[2, 2, :], [0], N, N)) - sp.spdiags(
            eps[0, 2, :] / eps[2, 2, :], [0], N, N
        ).dot(dxb)
        byax = -dyb.dot(inv_mu_zz).dot(dyf) + sp.spdiags(
            -eps[0, 0, :] + eps[0, 2, :] * eps[2, 0, :] / eps[2, 2, :], [0], N, N
        )
        byay = dyb.dot(inv_mu_zz).dot(dxf) + sp.spdiags(
            -eps[0, 1, :] + eps[0, 2, :] * eps[2, 1, :] / eps[2, 2, :], [0], N, N
        )

        mat = sp.bmat(
            [
                [axax, axay, axbx, axby],
                [ayax, ayay, aybx, ayby],
                [bxax, bxay, bxbx, bxby],
                [byax, byay, bybx, byby],
            ]
        )

        # The eigenvalues for the matrix above are 1j * (neff + 1j * keff)
        # Multiply the matrix by -1j, so that eigenvalues are (neff + 1j * keff)
        mat *= -1j

        # change matrix sign for backward direction
        if direction == "-":
            mat *= -1

        # Cast matrix to target data type
        mat_dtype = cls.matrix_data_type(eps, mu, der_mats, mat_precision, is_tensorial=True)
        mat = cls.type_conversion(mat, mat_dtype)

        # Trim small values in single precision case
        if mat_precision == "single":
            cls.trim_small_values(mat, tol=fp_eps)

        # Casting starting vector to target data type
        vec_init = cls.type_conversion(vec_init, mat_dtype)

        # Starting eigenvalue guess in target data type
        eig_guess = cls.type_conversion(np.array([neff_guess]), mat_dtype)[0]

        # Call the eigensolver.
        vals, vecs = cls.solver_eigs(
            mat,
            num_modes,
            vec_init,
            guess_value=eig_guess,
            mode_solver_type=mode_solver_type,
        )
        neff, keff = cls.eigs_to_effective_index(vals, mode_solver_type)
        # Sort by descending real part
        sort_inds = np.argsort(neff)[::-1]
        neff = neff[sort_inds]
        keff = keff[sort_inds]
        vecs = vecs[:, sort_inds]

        # Field components from eigenvectors
        Ex = vecs[:N, :]
        Ey = vecs[N : 2 * N, :]
        Hx = vecs[2 * N : 3 * N, :]
        Hy = vecs[3 * N :, :]

        # Get the other field components
        hxy_term = (-mu[2, 0, :] * Hx.T - mu[2, 1, :] * Hy.T).T
        Hz = inv_mu_zz.dot(dxf.dot(Ey) - dyf.dot(Ex) + hxy_term)
        exy_term = (-eps[2, 0, :] * Ex.T - eps[2, 1, :] * Ey.T).T
        Ez = inv_eps_zz.dot(dxb.dot(Hy) - dyb.dot(Hx) + exy_term)

        # Bundle up
        E = np.stack((Ex, Ey, Ez), axis=0)
        H = np.stack((Hx, Hy, Hz), axis=0)

        # Return to standard H field units (see CEM notes for H normalization used in solver)
        # The minus sign here is suspicious, need to check how modes are used in Mode objects
        H *= -1j / ETA_0

        return E, H, neff, keff

    @classmethod
    def solver_eigs(
        cls,
        mat,
        num_modes,
        vec_init,
        guess_value=1.0,
        M=None,
        **kwargs: Any,
    ):
        """Find ``num_modes`` eigenmodes of ``mat`` cloest to ``guess_value``.

        Parameters
        ----------
        mat : scipy.sparse matrix
            Square matrix for diagonalization.
        num_modes : int
            Number of eigenmodes to compute.
        guess_value : float, optional
        """
        import scipy.sparse.linalg as spl

        values, vectors = spl.eigs(
            mat, k=num_modes, sigma=guess_value, tol=TOL_EIGS, v0=vec_init, M=M
        )

        # for i, eig_i in enumerate(values):
        #     vec = vectors[:, i]
        #     rhs = vec
        #     if M is not None:
        #         rhs = M @ rhs
        #     eig_from_vec = (vec.T @ (mat @ vec)) / (vec.T @ rhs)
        #     residue = np.linalg.norm(mat @ vec - eig_i * rhs) / np.linalg.norm(vec)
        #     print(
        #         f"{i}-th eigenvalue: {eig_i}, referred from eigenvectors: {eig_from_vec}, relative residual: {residue}."
        #     )
        return values, vectors

    @classmethod
    def solver_eigs_relative(
        cls,
        mat,
        num_modes,
        vec_init,
        guess_value=1.0,
        M=None,
        basis_vecs=None,
        **kwargs: Any,
    ):
        """Find ``num_modes`` eigenmodes of ``mat`` cloest to ``guess_value``.

        Parameters
        ----------
        mat : scipy.sparse matrix
            Square matrix for diagonalization.
        num_modes : int
            Number of eigenmodes to compute.
        guess_value : float, optional
        """
        import scipy.linalg as linalg

        basis, _ = np.linalg.qr(basis_vecs)
        mat_basis = np.conj(basis.T) @ mat @ basis
        values, coeffs = linalg.eig(mat_basis)
        vectors = None
        vectors = basis @ coeffs
        return values, vectors

    @classmethod
    def isinstance_complex(cls, vec_or_mat, tol=TOL_COMPLEX):
        """Check if a numpy array or scipy.sparse.csr_matrix has complex component by looking at
        norm(x.imag)/norm(x)>TOL_COMPLEX

        Parameters
        ----------
        vec_or_mat : Union[np.ndarray, sp.csr_matrix]
        """
        import scipy.sparse.linalg as spl
        from scipy.sparse import csr_matrix

        if isinstance(vec_or_mat, np.ndarray):
            return np.linalg.norm(vec_or_mat.imag) / (np.linalg.norm(vec_or_mat) + fp_eps) > tol
        if isinstance(vec_or_mat, csr_matrix):
            mat_norm = spl.norm(vec_or_mat)
            mat_imag_norm = spl.norm(vec_or_mat.imag)
            return mat_imag_norm / (mat_norm + fp_eps) > tol
        raise RuntimeError(
            f"Variable type should be either numpy array or scipy.sparse.csr_matrix, got {type(vec_or_mat)}."
        )

    @classmethod
    def type_conversion(cls, vec_or_mat, new_dtype):
        """Convert vec_or_mat to new_type.

        Parameters
        ----------
        vec_or_mat : Union[np.ndarray, sp.csr_matrix]
            vec or mat to be converted.
        new_dtype : Union[np.complex128, np.complex64, np.float64, np.float32]
            Final type of vec or mat

        Returns
        -------
        converted_vec_or_mat : Union[np.ndarray, sp.csr_matrix]
        """

        if new_dtype in {np.complex128, np.complex64}:
            return vec_or_mat.astype(new_dtype)
        if new_dtype in {np.float64, np.float32}:
            converted_vec_or_mat = vec_or_mat.real
            return converted_vec_or_mat.astype(new_dtype)

        raise RuntimeError("Unsupported new_type.")

    @classmethod
    def set_initial_vec(cls, Nx, Ny, is_tensorial=False):
        """Set initial vector for eigs:
        1) The field at x=0 and y=0 boundaries are set to 0. This should be
        the case for PEC boundaries, but wouldn't hurt for non-PEC boundary;
        2) The vector is np.complex128 by default, and will be converted to
        appropriate type afterwards.

        Parameters
        ----------
        Nx : int
            Number of grids along x-direction.
        Ny : int
            Number of grids along y-direction.
        is_tensorial : bool
            diagonal or tensorial eigenvalue problem.
        """

        # The size of the vector is len_multiplier * Nx * Ny
        len_multiplier = 2
        if is_tensorial:
            len_multiplier *= 2

        # Initialize the vector
        size = (Nx, Ny, len_multiplier)
        rng = np.random.default_rng(0)
        vec_init = rng.random(size) + 1j * rng.random(size)

        # Set values at the boundary to be 0
        if Nx > 1:
            vec_init[0, :, :] = 0
        if Ny > 1:
            vec_init[:, 0, :] = 0

        # Concatenate the vector appropriately
        vec_init = np.vstack(vec_init)
        return vec_init.flatten("F")

    @classmethod
    def eigs_to_effective_index(cls, eig_list: Numpy, mode_solver_type: ModeSolverType):
        """Convert obtained eigenvalues to n_eff and k_eff.

        Parameters
        ----------
        eig_list : Numpy
            Array of eigenvalues
        mode_solver_type : ModeSolverType
            The type of mode solver problems

        Returns
        -------
        Tuple[Numpy, Numpy]
            n_eff and k_eff
        """
        if eig_list.size == 0:
            raise RuntimeError("Could not find any eigenmodes for this waveguide.")

        # for tensorial type, it's simply (neff + 1j * keff)
        if mode_solver_type == "tensorial":
            return np.real(eig_list), np.imag(eig_list)

        # for diagonal type, eigenvalues are -(neff + 1j * keff)**2
        if mode_solver_type == "diagonal":
            sqrt_eig_list = np.emath.sqrt(-eig_list + 0j)
            return np.real(sqrt_eig_list), np.imag(sqrt_eig_list)

        raise RuntimeError(f"Unidentified 'mode_solver_type={mode_solver_type}'.")

    @staticmethod
    def format_medium_data(mat_data):
        """
        mat_data can be either permittivity or permeability. It's either a single 2D array
        defining the relative property in the cross-section, or nine 2D arrays defining
        the property at the E(H)x, E(H)y, and E(H)z locations of the Yee grid in the order
        xx, xy, xz, yx, yy, yz, zx, zy, zz.
        """
        if isinstance(mat_data, Numpy):
            return (mat_data[i, :, :] for i in range(9))
        if len(mat_data) == 9:
            return (np.copy(e) for e in mat_data)
        raise ValueError("Wrong input to mode solver pemittivity/permeability!")

    @staticmethod
    def split_curl_field_postprocess(split_curl, E):
        """E has the shape (3, N, num_modes)"""
        _, Nx, Ny = split_curl.shape
        field_shape = E.shape

        # set a dummy value of split curl inside PEC to avoid division by 0 warning (it's 0/0, since
        # E field inside PEC is also 0); then by the end, zero out E inside PEC again just to be safe.
        outside_pec = ~np.isclose(split_curl, 0)
        split_curl_scaling = np.copy(split_curl)
        split_curl_scaling[~outside_pec] = 1.0

        E = E.reshape(3, Nx, Ny, field_shape[-1])
        E /= split_curl_scaling[:, :, :, np.newaxis]
        E *= outside_pec[:, :, :, np.newaxis]
        E = E.reshape(field_shape)
        return E

    @staticmethod
    def make_pml_invariant(Nxy, tensor, num_pml):
        """For a given epsilon or mu tensor of shape ``(3, 3, Nx, Ny)``, and ``num_pml`` pml layers
        along ``x`` and ``y``, make all the tensor values in the PML equal by replicating the first
        pixel into the PML."""

        Nx, Ny = Nxy
        new_ten = tensor.reshape((3, 3, Nx, Ny))
        new_ten[:, :, : num_pml[0], :] = new_ten[:, :, num_pml[0], :][:, :, None, :]
        new_ten[:, :, Nx - num_pml[0] + 1 :, :] = new_ten[:, :, Nx - num_pml[0], :][:, :, None, :]
        new_ten[:, :, :, : num_pml[1]] = new_ten[:, :, :, num_pml[1]][:, :, :, None]
        new_ten[:, :, :, Ny - num_pml[1] + 1 :] = new_ten[:, :, :, Ny - num_pml[1]][:, :, :, None]
        return new_ten.reshape((3, 3, -1))

    @staticmethod
    def split_curl_field_postprocess_inverse(split_curl, E) -> None:
        """E has the shape (3, N, num_modes)"""
        raise RuntimeError("Split curl not yet implemented for relative mode solver.")

    @staticmethod
    def mode_plane_contain_good_conductor(material_response) -> bool:
        """Find out if epsilon on the modal plane contain good conductors whose permittivity
        or permeability value is very large.
        """
        if material_response is None:
            return False
        return np.any(np.abs(material_response) > GOOD_CONDUCTOR_THRESHOLD * np.abs(pec_val))

    @staticmethod
    def _identify_degenerate_modes(
        n_complex: np.ndarray,
        tol: float,
    ) -> list[set[int]]:
        """Inspects the n_complex of modes to find sets of degenerate modes."""
        num_modes = len(n_complex)
        ungrouped = set(range(num_modes))
        degenerate_sets = []

        while ungrouped:
            # Start a new group with an ungrouped mode
            seed = ungrouped.pop()
            current_set = {seed}

            # Find all ungrouped modes similar to the seed
            to_remove = set()
            for mode_idx in ungrouped:
                if np.isclose(n_complex[mode_idx], n_complex[seed], rtol=tol, atol=tol):
                    current_set.add(mode_idx)
                    to_remove.add(mode_idx)

            # Remove grouped modes from ungrouped set
            ungrouped -= to_remove

            # Only keep groups with more than one mode
            if len(current_set) >= 2:
                degenerate_sets.append(current_set)

        return degenerate_sets

    @staticmethod
    def _identify_degenerate_modes_with_dot(
        degenerate_sets: list[set[int]],
        E: np.ndarray,
        H: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        tol: float,
    ) -> list[set[int]]:
        """Inspects the biorthogonality condition between modes to filter sets of degenerate modes.

        For each set of modes identified as degenerate by neff similarity, this function computes
        the overlap matrix and uses connected components to find which modes are truly degenerate
        (have significant mutual overlap).

        Returns only groups with >= 2 modes that have significant overlap.
        """
        import scipy.sparse as sp

        filtered_degenerate_sets = []

        for degenerate_group in degenerate_sets:
            # Convert set to list for indexing
            mode_indices = sorted(degenerate_group)

            # S is a symmetric overlap matrix of the degenerate modes
            S = EigSolver._outer_dot(E, H, dl_primal, dl_dual, mode_indices)

            # Create adjacency matrix
            # True if overlap is significant, False if orthogonal
            # Zero out diagonal so a mode doesn't connect to itself
            adjacency = (np.abs(S) > tol).astype(int)
            np.fill_diagonal(adjacency, 0)

            # Find connected components
            # labels: array indicating which group each mode belongs to
            _, labels = sp.csgraph.connected_components(sp.csr_matrix(adjacency), directed=False)

            # Group modes by their connected component label
            groups = {}
            for label_idx, group_id in enumerate(labels):
                mode_index = mode_indices[label_idx]
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(mode_index)

            # Convert groups dict to list of sets, keeping only groups with >= 2 modes
            for group_modes in groups.values():
                if len(group_modes) >= 2:
                    filtered_degenerate_sets.append(set(group_modes))

        return filtered_degenerate_sets

    @staticmethod
    def _dot(
        E: np.ndarray,
        H: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        mode_1: int,
        mode_2: int,
    ) -> complex:
        """Dot product based on the bi-orthogonality relationship between E and H."""
        # Extract field components
        Ex = E[0, ...]
        Ey = E[1, ...]
        Hx = H[0, ...]
        Hy = H[1, ...]

        # Make the differential area elements
        Ex_Hy_dS = np.outer(dl_primal[0], dl_dual[1])
        Ey_Hx_dS = np.outer(dl_dual[0], dl_primal[1])

        term1 = Ex[..., mode_1] * Hy[..., mode_2] + Ex[..., mode_2] * Hy[..., mode_1]
        term1 *= Ex_Hy_dS
        term2 = Ey[..., mode_1] * Hx[..., mode_2] + Ey[..., mode_2] * Hx[..., mode_1]
        term2 *= Ey_Hx_dS
        return (1 / 4) * np.sum(term1 - term2)

    @staticmethod
    def _cauchy_schwarz_dot_bound(
        E: np.ndarray,
        H: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        mode_idx: int,
    ) -> complex:
        """Calculates the upper bound for the self-dot (overlap) using the Cauchy-Schwarz inequality.

        Hard to predict how the normalization of the eigenvectors from the linear algebra solver
        translates to normalization in terms of the EM overlap integral,
        so we use this upper bound to determine whether a mode should be considered as self-orthogonal.

        Sketch of derivation for the upper bound:
        | S (E x H) . z dA |               (Absolute value of the bi-orthogonality overlap integral)
        <=  S |(E x H) . z| dA             (Triangle Inequality)
        <=  S |E| * |H|     dA             (Cross Product Magnitude)
        <= sqrt(S |E|^2) * sqrt(S |H|^2)   (Cauchy-Schwarz)
        """
        # Extract all field components
        Ex = E[0, ...]
        Ey = E[1, ...]
        Ez = E[2, ...]
        Hx = H[0, ...]
        Hy = H[1, ...]
        Hz = H[2, ...]

        # Make the differential area elements
        Ex_Hy_dS = np.outer(dl_primal[0], dl_dual[1])
        Ey_Hx_dS = np.outer(dl_dual[0], dl_primal[1])
        Ez_dS = np.outer(dl_dual[0], dl_dual[1])
        Hz_dS = np.outer(dl_primal[0], dl_primal[1])

        E_int = np.sum(Ex[..., mode_idx] * Ex[..., mode_idx].conj() * Ex_Hy_dS)
        E_int += np.sum(Ey[..., mode_idx] * Ey[..., mode_idx].conj() * Ey_Hx_dS)
        E_int += np.sum(Ez[..., mode_idx] * Ez[..., mode_idx].conj() * Ez_dS)
        E_int = E_int / 2

        H_int = np.sum(Hx[..., mode_idx] * Hx[..., mode_idx].conj() * Ey_Hx_dS)
        H_int += np.sum(Hy[..., mode_idx] * Hy[..., mode_idx].conj() * Ex_Hy_dS)
        H_int += np.sum(Hz[..., mode_idx] * Hz[..., mode_idx].conj() * Hz_dS)
        H_int = H_int / 2

        return np.sqrt(E_int * H_int)

    @staticmethod
    def _outer_dot(
        E: np.ndarray,
        H: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        mode_indices: set[int],
    ) -> np.ndarray:
        """Vectorized modal overlap matrix calculation for a set of modes.

        This overlap is based on the bi-orthogonality relationship between E and H.
        Returns a matrix S where S[i, j] is the overlap between mode i and mode j.

        Note: The overlap matrix is symmetric, so we only compute the upper triangle
        (including diagonal) and fill the lower triangle by symmetry.
        """
        # Convert set to sorted list for consistent indexing
        mode_list = sorted(mode_indices)
        n_modes = len(mode_list)

        # Extract field components
        Ex_sel = E[0][..., mode_list]  # (Nx, Ny, 1, n)
        Ey_sel = E[1][..., mode_list]
        Hx_sel = H[0][..., mode_list]
        Hy_sel = H[1][..., mode_list]

        # Make the differential area elements
        Ex_Hy_dS = np.outer(dl_primal[0], dl_dual[1])
        Ey_Hx_dS = np.outer(dl_dual[0], dl_primal[1])

        # Initialize output matrix
        dtype = Ex_sel.dtype
        S = np.zeros((n_modes, n_modes), dtype=dtype)

        # Only compute upper triangle (including diagonal)
        for i in range(n_modes):
            # Vectorize over j >= i
            j_indices = np.arange(i, n_modes)

            # Extract mode i fields
            Ex_i = Ex_sel[..., i : i + 1]  # (Nx, Ny, 1, 1)
            Ey_i = Ey_sel[..., i : i + 1]
            Hy_i = Hy_sel[..., i : i + 1]
            Hx_i = Hx_sel[..., i : i + 1]

            # Extract mode j fields (vectorized for j >= i)
            Ex_j = Ex_sel[..., j_indices]  # (Nx, Ny, 1, n_j)
            Ey_j = Ey_sel[..., j_indices]
            Hy_j = Hy_sel[..., j_indices]
            Hx_j = Hx_sel[..., j_indices]

            # Compute term1: (Ex[i] * Hy[j] + Ex[j] * Hy[i]) * dS
            term1 = (Ex_i * Hy_j + Ex_j * Hy_i) * Ex_Hy_dS[..., np.newaxis]

            # Compute term2: (Ey[i] * Hx[j] + Ey[j] * Hx[i]) * dS
            term2 = (Ey_i * Hx_j + Ey_j * Hx_i) * Ey_Hx_dS[..., np.newaxis]

            # Sum over spatial dimensions to get S[i, j] for j >= i
            S[i, j_indices] = (1 / 4) * np.sum(term1 - term2, axis=(0, 1))

        # Fill lower triangle by symmetry
        S = S + S.T - np.diag(np.diag(S))

        return S

    @staticmethod
    def _normalize_modes(
        E: np.ndarray,
        H: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        mode_indices: list[int],
        self_orthogonal_tol: float,
    ) -> tuple[np.ndarray, np.ndarray, set[int]]:
        """Normalize modes so that their self-dot equals 1.0.

        Returns:
            Tuple of (E, H, self_orthogonal_modes) where self_orthogonal_modes is a set of mode indices
            that are self-orthogonal and cannot be normalized.
        """
        self_orthogonal_modes = set()
        for mode_idx in mode_indices:
            self_dot = EigSolver._dot(E, H, dl_primal, dl_dual, mode_idx, mode_idx)
            self_dot_upper_bound = EigSolver._cauchy_schwarz_dot_bound(
                E, H, dl_primal, dl_dual, mode_idx
            )
            # Check if mode is self-orthogonal or otherwise not suitable for orthogonalization procedure
            if (
                np.abs(self_dot) < self_dot_upper_bound * self_orthogonal_tol
                or np.any(np.isnan(E[..., mode_idx]))
                or np.any(np.isnan(H[..., mode_idx]))
            ):
                self_orthogonal_modes.add(mode_idx)
                continue
            norm_factor = 1.0 / np.sqrt(self_dot)
            E[..., mode_idx] *= norm_factor
            H[..., mode_idx] *= norm_factor

        return E, H, self_orthogonal_modes

    @classmethod
    def _orthogonalize_degenerate_modes(
        cls,
        E: np.ndarray,
        H: np.ndarray,
        neff: np.ndarray,
        keff: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
        self_orthogonal_modes: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify and orthogonalize degenerate modes, excluding self-orthogonal modes."""
        # Identify and orthogonalize candidate degenerate modes with a coarser tolerance
        degenerate_modes = cls._identify_degenerate_modes(
            neff + keff * 1j, TOL_DEGENERATE_CANDIDATE
        )

        # Remove self-orthogonal modes from degenerate mode sets since they cannot be orthogonalized
        degenerate_modes = [
            mode_set - self_orthogonal_modes
            for mode_set in degenerate_modes
            if len(mode_set - self_orthogonal_modes)
            > 1  # Only keep sets with at least 2 valid modes
        ]

        # Orthogonalize candidate degenerate modes that result in an overlap greater than
        # the required tolerance
        filtered_degenerate_modes = cls._identify_degenerate_modes_with_dot(
            degenerate_modes, E, H, dl_primal, dl_dual, TOL_NEEDS_ORTHOGONALIZATION
        )

        E, H = cls._make_orthogonal_basis_for_degenerate_modes(
            filtered_degenerate_modes, E, H, dl_primal, dl_dual
        )

        return E, H

    @staticmethod
    def _make_orthogonal_basis_for_degenerate_modes(
        degenerate_mode_sets: list[set[int]],
        E_vec: np.ndarray,
        H_vec: np.ndarray,
        dl_primal: np.ndarray,
        dl_dual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensures that groups of degenerate modes are orthogonal, which is not guaranteed for eigenvectors with degenerate eigenvalues."""
        dtype = E_vec.dtype
        for degenerate_group in degenerate_mode_sets:
            # Convert set to list for indexing
            mode_indices = sorted(degenerate_group)

            # S is a symmetric overlap matrix of the degenerate modes.
            S = EigSolver._outer_dot(E_vec, H_vec, dl_primal, dl_dual, mode_indices)

            # 1. Diagonalize
            # eigenvalues (w) and right eigenvectors (v)
            eigvals, eigvecs = np.linalg.eig(S)
            # 2. Normalize Eigenvectors using UNCONJUGATED dot product
            # Standard np.linalg.eig normalizes so v.conj().T @ v = 1 (Hermitian)
            # We need v.T @ v = 1 (Complex Symmetric)
            for i in range(eigvecs.shape[1]):
                vec = eigvecs[:, i]
                # Calculate complex self-dot (no conjugation)
                self_dot = np.dot(vec, vec)

                # Avoid division by zero
                if np.isclose(self_dot, 0, atol=np.finfo(dtype).tiny):
                    raise ValueError(f"Eigenvector {i} is self-orthogonal. Cannot orthogonalize.")

                scale_factor = np.sqrt(self_dot)
                eigvecs[:, i] = vec / scale_factor
            # 3. Calculate Inverse Square Root of Eigenvalues
            lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
            # 4. Construct W
            # Since V is complex orthogonal (V.T @ V = I), V_inv = V.T
            # W = V * Lambda^(-1/2) * V.T
            W = eigvecs @ lambda_inv_sqrt @ eigvecs.T
            # S_new will be identity
            # S_new = W.T @ S @ W
            E_vec[..., mode_indices] = E_vec[..., mode_indices] @ W
            H_vec[..., mode_indices] = H_vec[..., mode_indices] @ W

        return E_vec, H_vec


def compute_modes(*args: Any, **kwargs: Any) -> tuple[Numpy, Numpy, str]:
    """A wrapper around ``EigSolver.compute_modes``, which is used in :class:`.ModeSolver`."""
    return EigSolver.compute_modes(*args, **kwargs)
