"""Fit PoleResidue Dispersion models to optical NK data"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from rich.progress import Progress
from pydantic import Field, validator, PositiveInt, NonNegativeFloat, PositiveFloat
import scipy

from .fit import DispersionFitter
from ...log import log
from ...components.base import Tidy3dBaseModel, cached_property
from ...components.medium import PoleResidue
from ...components.types import ArrayFloat1D, ArrayComplex1D, ArrayFloat2D, ArrayComplex2D
from ...constants import HBAR
from ...exceptions import ValidationError

# numerical tolerance for pole relocation for fast fitter
TOL = 1e-8
# numerical cutoff for passivity testing
CUTOFF = np.finfo(np.float32).eps
# range for checking and enforcing passivity, in addition to extrema method
PASSIVITY_MIN = -10
PASSIVITY_MAX = 4
PASSIVITY_NUM = 1000
# parameters for passivity optimization
PASSIVITY_NUM_ITERS_DEFAULT = 50
SLSQP_CONSTRAINT_SCALE_DEFAULT = 1e35
# min value of the rms for default weights calculated based on Re[eps], Im[eps]
RMS_MIN = 0.1

DEFAULT_MAX_POLES = 5
DEFAULT_NUM_ITERS = 20
DEFAULT_TOLERANCE_RMS = 1e-5

# this avoids divide by zero errors with lossless poles
SCALE_FACTOR = 1.01


class AdvancedFastFitterParam(Tidy3dBaseModel):
    """Advanced fast fitter parameters."""

    loss_bounds: Tuple[float, float] = Field(
        (0, np.inf),
        title="Loss bounds",
        description="Bounds (lower, upper) on Im[eps]. Default corresponds to only passivity. "
        "A lower bound of 0 or greater ensures passivity. To fit a gain medium without "
        "additional constraints, use ``loss_bounds=(-np.inf, np.inf)``. "
        "Increasing the lower bound could help with simulation stability. "
        "A finite upper bound may be helpful when fitting lossless materials. "
        "In this case, consider also increasing the weight for fitting the imaginary part.",
    )
    weights: Tuple[NonNegativeFloat, NonNegativeFloat] = Field(
        None,
        title="Weights",
        description="Weights (real, imag) in objective function for fitting. The weights "
        "are applied to the real and imaginary parts of the permittivity epsilon. The weights "
        "will be rescaled together so they average to 1. If ``None``, the weights are calculated "
        "according to the typical value of the real and imaginary part, so that the relative error "
        "in the real and imaginary part of the fit should be comparable. "
        "More precisely, the RMS value ``rms`` of the real and imaginary parts are "
        "calculated, and the default weights are 1 / max(``rms``, ``RMS_MIN``). "
        "Changing this can be helpful if fitting either the real or imaginary part is "
        "more important than the other.",
    )
    show_progress: bool = Field(
        True,
        title="Show progress bar",
        description="Whether to show progress bar during fitter run.",
    )
    show_unweighted_rms: bool = Field(
        False,
        title="Show unweighted RMS",
        description="Whether to show unweighted RMS error in addition to the default weighted "
        'RMS error. Requires ``td.config.logging_level = "INFO"``.',
    )
    relaxed: Optional[bool] = Field(
        None,
        title="Relaxed",
        description="Whether to use relaxed fitting algorithm, which "
        "has better pole relocation properties. If ``None``, will try both original and relaxed "
        "algorithms.",
    )
    smooth: Optional[bool] = Field(
        None,
        title="Smooth",
        description="Whether to use real starting poles, which can help when fitting smooth data. "
        "If ``None``, will try both real and complex starting poles.",
    )
    logspacing: Optional[bool] = Field(
        None,
        title="Log spacing",
        description="Whether to space the poles logarithmically. "
        "If ``None``, will try both log and linear spacing.",
    )

    # more technical parameters
    num_iters: PositiveInt = Field(
        DEFAULT_NUM_ITERS,
        title="Number of iterations",
        description="Number of iterations of the fitting algorithm. Make this smaller to "
        "speed up fitter, or make it larger to try to improve fit.",
    )
    passivity_num_iters: PositiveInt = Field(
        PASSIVITY_NUM_ITERS_DEFAULT,
        title="Number of loss bounds enforcement iterations",
        description="Number of loss bounds enforcement iterations of the fitting algorithm. "
        "Make this smaller to speed up fitter. There will be a warning if this value "
        "is too small. To fit a gain medium, use the ``loss_bounds`` parameter instead.",
    )
    slsqp_constraint_scale: PositiveFloat = Field(
        SLSQP_CONSTRAINT_SCALE_DEFAULT,
        title="Scale factor for SLSQP",
        description="Passivity constraint is weighted relative to fit quality by this factor, "
        "before running passivity optimization using the SLSQP algorithm. "
        "There will be a warning if this value is too small.",
    )

    @validator("loss_bounds", always=True)
    def _max_loss_geq_min_loss(cls, val):
        """Must have max_loss >= min_loss."""
        if val[0] > val[1]:
            raise ValidationError(
                "The loss lower bound cannot be larger than the loss upper bound."
            )
        return val

    @validator("weights", always=True)
    def _weights_average_to_one(cls, val):
        """Weights must average to one."""
        if val is None:
            return None
        avg = (val[0] + val[1]) / 2
        new_val = (val[0] / avg, val[1] / avg)
        return new_val


# pylint: disable=too-many-public-methods
class FastFitterData(AdvancedFastFitterParam):
    """Data class for internal use while running fitter."""

    omega: ArrayComplex1D = Field(
        ..., title="Angular frequencies in eV", description="Angular frequencies in eV"
    )
    eps: ArrayComplex1D = Field(..., title="Permittivity", description="Permittivity to fit")

    optimize_eps_inf: bool = Field(
        None, title="Optimize eps_inf", description="Whether to optimize ``eps_inf``."
    )

    num_poles: PositiveInt = Field(None, title="Number of poles", description="Number of poles")
    eps_inf: float = Field(
        None,
        title="eps_inf",
        description="Value of ``eps_inf``.",
    )
    poles: ArrayComplex1D = Field(
        None, title="Pole frequencies in eV", description="Pole frequencies in eV"
    )
    residues: ArrayComplex1D = Field(None, title="Residues in eV", description="Residues in eV")

    passivity_optimized: Optional[bool] = Field(
        False,
        title="Passivity optimized",
        description="Whether the fit was optimized to enforce passivity. If None, "
        "then passivity optimization did not terminate; "
        "consider increasing ``AdvancedFastFitterParam.passivity_num_iters``.",
    )
    passivity_num_iters_too_small: bool = Field(
        False,
        title="Passivity num iters too small",
        description="If this is True, consider increasing "
        "``AdvancedFastFitterParam.passivity_num_iters``.",
    )
    slsqp_constraint_scale_too_small: bool = Field(
        False,
        title="SLSQP constraint scale too small",
        description="The constraint is rescaled by ``slsqp_constraint_scale`` before "
        "running passivity optimization using the SLSQP algorithm. If this is ``True``, "
        "consider increasing ``AdvancedFastFitterParam.slsqp_constraint_scale``.",
    )

    @validator("eps_inf", always=True)
    def _eps_inf_geq_one(cls, val, values):
        """Must have eps_inf >= 1 unless it is being optimized.
        In the latter case, it will be made >= 1 later."""
        if values["optimize_eps_inf"] is False and val < 1:
            raise ValidationError("The value of 'eps_inf' must be at least 1.")
        return val

    @validator("poles", always=True)
    def _generate_initial_poles(cls, val, values):
        """Generate initial poles."""
        if not val is None:
            return val
        if values["logspacing"] is None or values["smooth"] is None:
            return None
        omega = values["omega"]
        num_poles = values["num_poles"]
        if values["logspacing"]:
            pole_range = np.logspace(
                np.log10(min(omega) / SCALE_FACTOR), np.log10(max(omega) * SCALE_FACTOR), num_poles
            )
        else:
            pole_range = np.linspace(
                min(omega) / SCALE_FACTOR, max(omega) * SCALE_FACTOR, num_poles
            )
        if values["smooth"]:
            poles = -pole_range
        else:
            poles = -pole_range / 100 + 1j * pole_range
        return poles

    @validator("residues", always=True)
    def _generate_initial_residues(cls, val, values):
        """Generate initial residues."""
        if not val is None:
            return val
        poles = values["poles"]
        if poles is None:
            return None
        return np.zeros(len(poles))

    @classmethod
    def initialize(
        cls,
        omega: ArrayFloat1D,
        eps: ArrayComplex1D,
        eps_inf: float,
        advanced_param: AdvancedFastFitterParam,
    ) -> FastFitterData:
        """Construct :class:`.FastFitterData` from :class:`.AdvancedFastFitterParam`."""
        weights = advanced_param.weights or cls.get_default_weights(eps)
        optimize_eps_inf = None if eps_inf is None else False
        data = FastFitterData(
            num_iters=advanced_param.num_iters,
            passivity_num_iters=advanced_param.passivity_num_iters,
            slsqp_constraint_scale=advanced_param.slsqp_constraint_scale,
            loss_bounds=advanced_param.loss_bounds,
            weights=weights,
            show_progress=advanced_param.show_progress,
            show_unweighted_rms=advanced_param.show_unweighted_rms,
            relaxed=advanced_param.relaxed,
            smooth=advanced_param.smooth,
            logspacing=advanced_param.logspacing,
            omega=omega,
            eps=eps,
            optimize_eps_inf=optimize_eps_inf,
            eps_inf=eps_inf or 1,
        )
        return data

    @cached_property
    def real_poles(self) -> ArrayFloat1D:
        """The real poles."""
        return self.poles[np.isreal(self.poles)]

    @cached_property
    def complex_poles(self) -> ArrayFloat1D:
        """The complex poles."""
        return self.poles[np.iscomplex(self.poles)]

    @classmethod
    def get_default_weights(cls, eps: ArrayComplex1D) -> Tuple[float, float]:
        """Default weights based on real and imaginary part of eps."""
        rms = np.array([np.sqrt(np.mean(x**2)) for x in (np.real(eps), np.imag(eps))])
        rms = np.maximum(RMS_MIN, rms)
        weights = [1 / val for val in rms]
        average = (weights[0] + weights[1]) / 2
        weights = [val / average for val in weights]
        return tuple(weights)

    @cached_property
    def pole_residue(self) -> PoleResidue:
        """Corresponding :class:`.PoleResidue` model."""
        if self.eps_inf is None or self.poles is None:
            return None
        return PoleResidue(
            eps_inf=self.eps_inf, poles=list(zip(self.poles / HBAR, self.residues / HBAR))
        )

    def evaluate(self, omega: float) -> complex:
        """Evaluate model at omega in eV."""
        eps = self.eps_inf
        for (pole, res) in zip(self.poles, self.residues):
            eps += -res / (1j * omega + pole) - np.conj(res) / (1j * omega + np.conj(pole))
        return eps

    @cached_property
    def values(self) -> ArrayComplex1D:
        """Evaluate model at sample frequencies."""
        return self.evaluate(self.omega)

    @cached_property
    def imag_extrema(self) -> Tuple[ArrayFloat1D, ArrayComplex1D]:
        """Extrema of imaginary part of eps."""

        # pylint: disable=too-many-locals
        def _extrema_loss_freq_finder(areal, aimag, creal, cimag):
            """For each pole, find frequencies for the extrema of Im[eps]"""

            a_square = areal**2 + aimag**2
            alpha = creal
            beta = creal * (areal**2 - aimag**2) + 2 * cimag * areal * aimag
            mus = 2 * (areal**2 - aimag**2)
            nus = a_square**2

            numerator = np.array([0])
            denominator = np.array([1])
            for i in range(len(creal)):
                numerator_i = np.array(
                    [alpha[i], 2 * beta[i], mus[i] * beta[i] - alpha[i] * nus[i]]
                )
                denominator_i = np.array(
                    [1, 2 * mus[i], 2 * nus[i] + mus[i] ** 2, 2 * mus[i] * nus[i], nus[i] ** 2]
                )
                # to avoid divergence, let's renormalize
                if np.abs(alpha[i]) > 1:
                    numerator_i /= alpha[i]
                    denominator_i /= alpha[i]

                # n/d + ni/di = (n*di+d*ni)/(d*di)
                n_di = np.polymul(numerator, denominator_i)
                d_ni = np.polymul(denominator, numerator_i)
                numerator = np.polyadd(n_di, d_ni)
                denominator = np.polymul(denominator, denominator_i)

            roots = np.sqrt(np.roots(numerator) + 0j)
            # cutoff to determine if it's a real number
            r_real = roots.real[np.abs(roots.imag) / (np.abs(roots) + CUTOFF) < CUTOFF]
            return r_real[r_real > 0]

        try:
            return _extrema_loss_freq_finder(
                self.poles.real,
                self.poles.imag,
                self.residues.real,
                self.residues.imag,
            )
        except np.linalg.LinAlgError:
            log.warning("'LinAlgError' in passivity checking, passivity not guaranteed.")
            return []

    @cached_property
    def loss_in_bounds_violations(self) -> ArrayFloat1D:
        """Return list of frequencies where model violates loss bounds."""
        # let's check a big range in addition to the imag_extrema
        range_omega = np.logspace(PASSIVITY_MIN, PASSIVITY_MAX, PASSIVITY_NUM)
        omega = np.concatenate((range_omega, self.imag_extrema))
        loss = self.evaluate(omega).imag
        bmin, bmax = self.loss_bounds
        violation_inds = np.where((loss < bmin) | (loss > bmax))
        return omega[violation_inds]

    @cached_property
    def loss_in_bounds(self) -> bool:
        """Whether model satisfies loss bounds."""
        return len(self.loss_in_bounds_violations) == 0

    @cached_property
    def sellmeier_passivity(self) -> bool:
        """Check passivity in the case of marginally stable poles, if loss_bounds[0] >= 0."""
        if self.loss_bounds[0] < 0:
            return True
        for pole, res in zip(self.poles, self.residues):
            if np.real(pole) == 0:
                if np.imag(pole) * np.imag(res) > 0:
                    return False

        return True

    @cached_property
    def rms_error(self) -> float:
        """RMS error."""
        if self.eps_inf is None or self.residues is None:
            return np.inf
        diff = self.values - self.eps
        square = (self.weights[0] * np.real(diff)) ** 2 + (self.weights[1] * np.imag(diff)) ** 2
        return np.sqrt(np.sum(square) / len(self.eps))

    @cached_property
    def unweighted_rms_error(self) -> float:
        """RMS error."""
        if self.eps_inf is None or self.residues is None:
            return np.inf
        diff = self.values - self.eps
        square = (np.real(diff)) ** 2 + (np.imag(diff)) ** 2
        return np.sqrt(np.sum(square) / len(self.eps))

    def pole_matrix_omega(self, omega: ArrayFloat1D) -> ArrayComplex2D:
        """A matrix used in the fitting algorithms containing the pole information."""
        size = len(self.real_poles) + 2 * len(self.complex_poles)
        pole_matrix = np.zeros((len(omega), size), dtype=complex)
        for i, pole in enumerate(self.real_poles):
            pole_matrix[:, i] = 1 / (1j * omega + pole) + 1 / (1j * omega + np.conj(pole))
        offset = len(self.real_poles)
        for i, pole in enumerate(self.complex_poles):
            pole_matrix[:, offset + 2 * i] = 1 / (1j * omega + pole) + 1 / (
                1j * omega + np.conj(pole)
            )
            pole_matrix[:, offset + 2 * i + 1] = 1j / (1j * omega + pole) - 1j / (
                1j * omega + np.conj(pole)
            )
        return pole_matrix

    @cached_property
    def pole_matrix(self) -> ArrayComplex2D:
        """A matrix used in the fitting algorithms containing the pole information."""
        return self.pole_matrix_omega(self.omega)

    def real_weighted_matrix(self, matrix: ArrayComplex2D) -> ArrayFloat2D:
        """Turn a complex matrix into a weighted real matrix."""
        return np.concatenate(
            (self.weights[0] * np.real(matrix), self.weights[1] * np.imag(matrix))
        )

    # pylint:disable=too-many-locals
    def iterate_poles(self) -> FastFitterData:
        """Perform a single iteration of the pole-updating procedure."""

        def compute_zeros(residues: ArrayComplex1D, d_tilde: float) -> ArrayComplex1D:
            """Compute the zeros from the residues."""
            size = len(self.real_poles) + 2 * len(self.complex_poles)
            a_matrix = np.zeros((size, size))
            b_vector = np.zeros(size)
            c_vector = np.zeros(size)
            for i, pole in enumerate(self.real_poles):
                a_matrix[i, i] = np.real(pole)
                b_vector[i] = 1
                c_vector[i] = np.real(residues[i])
            for i, pole in enumerate(self.complex_poles):
                offset = len(self.real_poles)
                a_matrix[
                    offset + 2 * i : offset + 2 * i + 2, offset + 2 * i : offset + 2 * i + 2
                ] = [[np.real(pole), np.imag(pole)], [-np.imag(pole), np.real(pole)]]
                b_vector[offset + 2 * i : offset + 2 * i + 2] = [2, 0]
                c_vector[offset + 2 * i : offset + 2 * i + 2] = [
                    np.real(residues[i]),
                    np.imag(residues[i]),
                ]

            zeros, _ = np.linalg.eig(a_matrix + np.outer(b_vector, c_vector) / d_tilde)

            return zeros.astype(complex)

        d_tilde = None if self.relaxed else 1
        for _ in range(2 if self.relaxed else 1):
            # build the matrices
            if self.optimize_eps_inf:
                poly_len = 1
                b_vector = np.zeros(len(self.eps), dtype=complex)
            else:
                poly_len = 0
                # fixed eps_inf enters into b_vector
                b_vector = -self.eps_inf * np.ones(len(self.eps), dtype=complex)

            a_matrix = np.hstack(
                (
                    self.pole_matrix,
                    np.ones((len(self.omega), poly_len)),
                    -self.eps[:, None] * self.pole_matrix,
                )
            )

            if d_tilde is None:
                nontriviality_weight = np.sqrt(np.sum(np.abs(self.eps) ** 2)) / len(self.omega)
                nontriviality_matrix = np.real(np.sum(self.pole_matrix, axis=0))
                nontriviality_matrix = np.concatenate(
                    (
                        np.zeros(len(nontriviality_matrix)),
                        np.zeros(poly_len),
                        nontriviality_matrix,
                        [len(self.omega)],
                    )
                )
                nontriviality_matrix *= nontriviality_weight
                a_matrix = np.hstack((a_matrix, -self.eps[:, None]))
            else:
                b_vector += d_tilde * self.eps

            a_matrix_real = self.real_weighted_matrix(a_matrix)
            b_vector_real = self.real_weighted_matrix(b_vector)

            if d_tilde is None:
                a_matrix_real = np.vstack((a_matrix_real, nontriviality_matrix))
                b_vector_real = np.concatenate(
                    (b_vector_real, [nontriviality_weight * len(self.omega)])
                )

            # solve the least squares problem
            x_vector = scipy.optimize.lsq_linear(a_matrix_real, b_vector_real).x

            # unpack the solution
            residues = np.zeros(len(self.poles), dtype=complex)
            size = len(self.real_poles) + 2 * len(self.complex_poles)
            offset0 = size + poly_len
            for i in range(len(self.real_poles)):
                residues[i] = x_vector[offset0 + i]
            offset = len(self.real_poles)
            for i in range(len(self.complex_poles)):
                residues[offset + i] = (
                    x_vector[offset0 + offset + 2 * i] + 1j * x_vector[offset0 + offset + 2 * i + 1]
                )

            if d_tilde is None:
                d_tilde = x_vector[-1]
                if abs(d_tilde) > TOL:
                    break
                d_tilde = TOL if d_tilde == 0 else TOL * np.sign(d_tilde)

        new_poles = compute_zeros(residues, d_tilde)
        # only keep one in each conjugate pair
        new_poles = new_poles[np.imag(new_poles) <= 0]
        # impose stability, negative real part
        new_poles[np.real(new_poles) > 0] = -1j * np.conj(1j * new_poles[np.real(new_poles) > 0])
        # impose minimum decay rate
        return self.updated_copy(poles=new_poles)

    def fit_residues(self) -> FastFitterData:
        """Fit residues."""
        # build the matrices
        if self.optimize_eps_inf:
            poly_len = 1
            b_vector = self.eps
        else:
            poly_len = 0
            b_vector = self.eps - self.eps_inf * np.ones(len(self.eps))

        a_matrix = np.hstack((self.pole_matrix, np.ones((len(self.omega), poly_len))))

        a_matrix_real = self.real_weighted_matrix(a_matrix)
        b_vector_real = self.real_weighted_matrix(b_vector)

        # solve the least squares problem
        bounds = (-np.inf * np.ones(a_matrix.shape[1]), np.inf * np.ones(a_matrix.shape[1]))
        bounds[0][-1] = 1  # eps_inf >= 1
        x_vector = scipy.optimize.lsq_linear(a_matrix_real, b_vector_real).x

        # unpack the solution
        residues = np.zeros(len(self.poles), dtype=complex)
        for i in range(len(self.real_poles)):
            residues[i] = x_vector[i]
        offset = len(self.real_poles)
        for i in range(len(self.complex_poles)):
            residues[offset + i] = x_vector[offset + 2 * i] + 1j * x_vector[offset + 2 * i + 1]

        if self.optimize_eps_inf:
            return self.updated_copy(residues=-residues, eps_inf=x_vector[-1])
        return self.updated_copy(residues=-residues)

    def iterate_fit(self) -> FastFitterData:
        """Perform a single fit to the data and return optimization result.

        Returns
        -------
        :class:`.FastFitterData`
            Result of single fit.
        """

        model = self.iterate_poles()
        model = model.fit_residues()

        return model

    # pylint:disable=too-many-locals
    def iterate_passivity(self, passivity_omega: ArrayFloat1D) -> Tuple[FastFitterData, int]:
        """Iterate passivity enforcement algorithm."""

        size = len(self.real_poles) + 2 * len(self.complex_poles)
        constraint_matrix = np.imag(self.pole_matrix_omega(passivity_omega))

        c_vector = np.imag(self.evaluate(passivity_omega))

        if self.loss_bounds[1] != np.inf:
            constraint_matrix = np.concatenate((constraint_matrix, -constraint_matrix))
            c_vector = np.concatenate(
                (c_vector - self.loss_bounds[0], self.loss_bounds[1] - c_vector)
            )

        a_matrix_real = self.real_weighted_matrix(self.pole_matrix)
        b_vector_real = self.real_weighted_matrix(self.values - self.eps)

        h_matrix = a_matrix_real.T @ a_matrix_real
        f_vector = a_matrix_real.T @ b_vector_real

        def loss(dx):
            return dx.T @ h_matrix @ dx / 2 - f_vector.T @ dx

        def jac(dx):
            return dx.T @ h_matrix - f_vector.T

        cons = {
            "type": "ineq",
            "fun": lambda dx: (c_vector - constraint_matrix @ dx) * self.slsqp_constraint_scale,
            "jac": lambda dx: -constraint_matrix * self.slsqp_constraint_scale,
        }
        opt = {"disp": False}

        x0 = np.zeros(size)
        err = np.amin(c_vector - constraint_matrix @ x0)
        result = scipy.optimize.minimize(
            loss, x0=x0, jac=jac, constraints=cons, method="SLSQP", options=opt
        )
        x_vector = result.x
        err = np.amin(c_vector - constraint_matrix @ x_vector)
        model = self
        if result.status == 0 and err < 0:
            model = self.updated_copy(slsqp_constraint_scale_too_small=True)
        residues = np.zeros(len(self.poles), dtype=complex)
        for i in range(len(self.real_poles)):
            residues[i] = x_vector[i]
        offset = len(self.real_poles)
        for i in range(len(self.complex_poles)):
            residues[offset + i] = x_vector[offset + 2 * i] + 1j * x_vector[offset + 2 * i + 1]
        return model.updated_copy(residues=np.array(self.residues) + residues), result.status

    def enforce_passivity(
        self,
    ) -> FastFitterData:
        """Try to enforce loss bounds."""
        if self.loss_in_bounds:
            return self

        model = self.updated_copy(passivity_optimized=True)
        violations = model.loss_in_bounds_violations
        range_omega = np.logspace(PASSIVITY_MIN, PASSIVITY_MAX, PASSIVITY_NUM)
        violations = np.unique(np.concatenate((violations, range_omega)))

        # only need one iteration since poles are fixed
        for _ in range(self.passivity_num_iters):
            model, status = model.iterate_passivity(violations)
            if model.loss_in_bounds or status != 0:
                return model
            new_violations = model.loss_in_bounds_violations
            violations = np.unique(np.concatenate((violations, new_violations)))

        model = model.updated_copy(passivity_num_iters_too_small=True)

        return model


class FastDispersionFitter(DispersionFitter):
    """Tool for fitting refractive index data to get a
    dispersive medium described by :class:`.PoleResidue` model."""

    def _fit_fixed_parameters(
        self, num_poles_range: Tuple[PositiveInt, PositiveInt], model: FastFitterData
    ) -> FastFitterData:
        def fit_non_passive(model: FastFitterData) -> FastFitterData:
            best_model = model
            for _ in range(model.num_iters):
                model = model.iterate_fit()

                if (
                    num_poles_range[0] <= len(model.poles)
                    and len(model.poles) <= num_poles_range[1]
                    and model.rms_error < best_model.rms_error
                ):

                    best_model = model
            return best_model

        model = fit_non_passive(model)

        if model.eps_inf < 1:
            model = model.updated_copy(
                eps_inf=max(1, np.mean(np.real(model.eps))), optimize_eps_inf=False
            )
            model = fit_non_passive(model)
        model = model.enforce_passivity()
        return model

    # pylint: disable=arguments-renamed, too-many-locals, too-many-arguments
    def fit(
        self,
        min_num_poles: PositiveInt = 1,
        max_num_poles: PositiveInt = DEFAULT_MAX_POLES,
        eps_inf: float = None,
        tolerance_rms: NonNegativeFloat = DEFAULT_TOLERANCE_RMS,
        advanced_param: AdvancedFastFitterParam = None,
    ) -> Tuple[PoleResidue, float]:
        """Fit data using a fast fitting algorithm.

        Note
        ----
        The algorithm is described in::

            B. Gustavsen and A. Semlyen, "Rational approximation
            of frequency domain responses by vector fitting,"
            IEEE Trans. Power. Deliv. 14, 3 (1999).

            B. Gustavsen, "Improving the pole relocation properties
            of vector fitting," IEEE Trans. Power Deliv. 21, 3 (2006).

            B. Gustavsen, "Enforcing Passivity for Admittance Matrices
            Approximated by Rational Functions," IEEE Trans. Power
            Syst. 16, 1 (2001).

        Note
        ----
        The fit is performed after weighting the real and imaginary parts,
        so the RMS error is also weighted accordingly. By default, the weights
        are chosen based on typical values of the data. To change this behavior,
        use 'AdvancedFastFitterParam.weights'.


        Parameters
        ----------
        min_num_poles: PositiveInt, optional
            Minimum number of poles in the model.
        max_num_poles: PositiveInt, optional
            Maximum number of poles in the model.
        eps_inf : float, optional
            Value of eps_inf to use in fit. If None, then eps_inf is also fit.
            Note: fitting eps_inf is not guaranteed to yield a global optimum, so
            the result may occasionally be better with a fixed value of eps_inf.
        tolerance_rms : float, optional
            Weighted RMS error below which the fit is successful and the result is returned.
        advanced_param : :class:`AdvancedFastFitterParam`, optional
            Advanced parameters for fitting.

        Returns
        -------
        Tuple[:class:`.PoleResidue`, float]
            Best fitting result: (dispersive medium, weighted RMS error).
        """

        omega = 2 * np.pi * HBAR * self.freqs[::-1]
        eps = self.eps_data[::-1]

        init_model = FastFitterData.initialize(
            omega, eps, eps_inf, advanced_param or AdvancedFastFitterParam()
        )
        log.info(f"Fitting weights=({init_model.weights[0]:.3g}, " f"{init_model.weights[1]:.3g}).")

        def make_configs():
            configs = [[p] for p in range(max(min_num_poles // 2, 1), max_num_poles + 1)]
            for setting in [
                init_model.relaxed,
                init_model.smooth,
                init_model.logspacing,
                init_model.optimize_eps_inf,
            ]:
                if setting is None:
                    configs = [c + [r] for c in configs for r in [True, False]]
                else:
                    configs = [c + [r] for c in configs for r in [setting]]
            return configs

        best_model = init_model
        warned_about_passivity_num_iters = False
        warned_about_slsqp_constraint_scale = False

        configs = make_configs()

        with Progress() as progress:

            task = progress.add_task(
                f"Fitting to weighted RMS of {tolerance_rms}...",
                total=len(configs),
                visible=init_model.show_progress,
            )

            while not progress.finished:

                # try different initial pole configurations
                for num_poles, relaxed, smooth, logspacing, optimize_eps_inf in configs:
                    model = init_model.updated_copy(
                        num_poles=num_poles,
                        relaxed=relaxed,
                        smooth=smooth,
                        logspacing=logspacing,
                        optimize_eps_inf=optimize_eps_inf,
                    )
                    model = self._fit_fixed_parameters((min_num_poles, max_num_poles), model)

                    if model.rms_error < best_model.rms_error:
                        log.debug(
                            f"Fitter: possible improved fit with "
                            f"rms_error={model.rms_error:.3g} found using "
                            f"relaxed={model.relaxed}, "
                            f"smooth={model.smooth}, "
                            f"logspacing={model.logspacing}, "
                            f"optimize_eps_inf={model.optimize_eps_inf}, "
                            f"loss_in_bounds={model.loss_in_bounds}, "
                            f"passivity_optimized={model.passivity_optimized}, "
                            f"sellmeier_passivity={model.sellmeier_passivity}."
                        )
                        if model.loss_in_bounds and model.sellmeier_passivity:
                            best_model = model
                        else:
                            if (
                                not warned_about_passivity_num_iters
                                and model.passivity_num_iters_too_small
                            ):
                                warned_about_passivity_num_iters = True
                                log.warning(
                                    "Did not finish enforcing passivity in dispersion fitter. "
                                    "If the fit is not good enough, consider increasing "
                                    "'AdvancedFastFitterParam.passivity_num_iters'."
                                )
                            if (
                                not warned_about_slsqp_constraint_scale
                                and model.slsqp_constraint_scale_too_small
                            ):
                                warned_about_slsqp_constraint_scale = True
                                log.warning(
                                    "SLSQP constraint scale may be too small. "
                                    "If the fit is not good enough, consider increasing "
                                    "'AdvancedFastFitterParam.slsqp_constraint_scale'."
                                )
                    progress.update(
                        task,
                        advance=1,
                        description=f"Best weighted RMS error so far: {best_model.rms_error:.3g}",
                        refresh=True,
                    )

                    # if below tolerance, return
                    if best_model.rms_error < tolerance_rms:
                        progress.update(
                            task,
                            completed=len(configs),
                            description=f"Best weighted RMS error: {best_model.rms_error:.3g}",
                            refresh=True,
                        )
                        log.info(
                            "Found optimal fit with weighted RMS error %.3g",
                            best_model.rms_error,
                        )
                        if best_model.show_unweighted_rms:
                            log.info(
                                "Unweighted RMS error %.3g",
                                best_model.unweighted_rms_error,
                            )

                        return best_model.pole_residue, best_model.rms_error

        # if exited loop, did not reach tolerance (warn)
        progress.update(
            task,
            completed=len(configs),
            description=f"Best weighted RMS error: {best_model.rms_error:.3g}",
            refresh=True,
        )

        log.warning(
            "Unable to fit with weighted RMS error under 'tolerance_rms' of %.3g", tolerance_rms
        )
        log.info("Returning best fit with weighted RMS error %.3g", best_model.rms_error)
        if best_model.show_unweighted_rms:
            log.info(
                "Unweighted RMS error %.3g",
                best_model.unweighted_rms_error,
            )

        return best_model.pole_residue, best_model.rms_error
