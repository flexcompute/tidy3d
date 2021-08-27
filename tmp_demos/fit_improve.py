import sys
import numpy as np
import matplotlib.pyplot as plt
import nlopt
import tidy3d as td

"""
NK data fitting tool

All frequency units are in eV currently to keep typical values on the order of unity.
This simplifies operator validation of the fit parameters, and also random initialization
of model parameters.
"""

""" constants """

EPSILON_0 = np.float32(8.85418782e-18)  # vacuum permittivity [F/um]
MU_0 = np.float32(1.25663706e-12)  # vacuum permeability [H/um]
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum [um/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # vacuum impedance
HBAR = 6.582119569e-16  # reduced Planck constant [eV*s]

""" Helper functions """


def unpack_complex(z):
    return z.real, z.imag


def pack_complex(r, i):
    return r + 1j * i


def unpack_coeffs(coeffs):

    assert len(coeffs) % 4 == 0
    num_poles = len(coeffs) // 4
    indices = 4 * np.arange(num_poles)

    ar = coeffs[indices + 0]
    ai = coeffs[indices + 1]
    cr = coeffs[indices + 2]
    ci = coeffs[indices + 3]

    a = pack_complex(ar, ai)
    c = pack_complex(cr, ci)
    return a, c


def pack_coeffs(a, c):

    ar, ai = unpack_complex(a)
    cr, ci = unpack_complex(c)

    stacked_coeffs = np.stack((ar, ai, cr, ci), axis=1)
    return stacked_coeffs.flatten()


def test_coeffs():
    """ pytest to make sure pack and unpack of coefficients is identity """

    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = unpack_coeffs(coeffs)
    coeffs_packed = pack_coeffs(a, c)
    assert np.allclose(coeffs, coeffs_packed)


def model(coeffs, w):
    # coeffs: A length 4*n list or numpy array, corresponding to data for n poles
    #         Each set of 4 elements in coeffs is (ar, ai, cr, ci); the real and
    #         imaginary parts of a (pole) and c (residue).
    # w: omega, in the same units as a and c.
    # returns: result of model

    # expand dimensions of omega to allow vectorization
    a, c = unpack_coeffs(coeffs)
    w_ = w[:, None]
    pole_contrib = c / (1j * w_ - a) + np.conj(c) / (1j * w_ - np.conj(a))
    return 1.0 + np.sum(pole_contrib, axis=-1)


def fit_eps_poles(ws, eps, npoles, lossy, globalopt=True, guess=None):
    # Performs the NLopt fitting for a pole/residue model
    # with a specified number of poles.

    # ws:  vector of frequencies
    # eps: vector of (complex) epsilon values, at each of the frequencies in eV.
    # npoles: The number of pole/residue pairs to use in the model.
    # lossy: Boolean value, if False, the returned model is guaranteed to be lossless.
    #        This simply adds a constraint that the real parts of a and c must be zero.
    # globalopt: If True, performs a randomly-initialized optimization using an
    #            augmented Lagrangian method, and enforces all nonlinear constraints.
    #            Otherwise, a local optimization is performed without enforcing nonlinear
    #            constraints (only the bounds constraints are enforced, which includes
    #            the dynamic stability criterion).
    # guess: If not None, then specifies the starting parameter values with which to
    #        initialize the optimization.

    # Currently the vector of frequencies is assumed to be in eV.
    # This makes it so that all frequency values are on the order of unity for typical
    # frequency ranges, so using standard normal random values for initialization is
    # pretty reasonable. Note however the units do not have to be in eV.

    def constraint(coeffs, _grad):
        # Evaluates the nonlinear stability criterion of
        #   Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
        #   "Comprehensive Study on Numerical Aspects of Modified
        #    Lorentz Model Based Dispersive FDTD Formulations,"
        #   IEEE TAP 2019.
        #
        # coeffs: length 4*n vector of (ar, ai, cr, ci).
        # grad: The gradient to return (not used).

        a, c = unpack_coeffs(coeffs)
        ar, ai = unpack_complex(a)
        cr, ci = unpack_complex(c)
        prstar = ar * cr + ai * ci
        res = 2 * prstar * ar - cr * (ar * ar + ai * ai)
        res[res >= 0] == 0
        return np.sum(res)

    def obj(coeffs, _grad):
        """ objective function for fit """

        residual = eps - model(coeffs, ws)
        rms_error = np.sqrt(np.sum(np.square(np.abs(residual))) / len(eps))
        cons = constraint(coeffs, _grad)
        return rms_error  # + 1e2 * cons    # note: not sure about 1e2 factor

    # set initial guess
    num_DoF = npoles * 4
    if guess is None or True:  # not, change to accept guess
        coeffs0 = 2 * (np.random.random(num_DoF) - 0.5)
    else:
        coeffs0 = guess

    # set method and objective
    method = nlopt.LN_NELDERMEAD if globalopt else nlopt.LN_SBPLX
    opt = nlopt.opt(method, num_DoF)
    opt.set_min_objective(obj)

    # set bounds
    ub = np.zeros(num_DoF, dtype=float)
    lb = np.zeros(num_DoF, dtype=float)
    indices = 4 * np.arange(npoles)

    if lossy:
        # if lossy, the real parts can be anything
        lb[indices] = -float("inf")
        ub[indices + 2] = float("inf")
    else:
        # otherwise, they need to be 0
        coeffs0[indices] = 0
        coeffs0[indices + 2] = 0

    lb[indices + 1] = -float("inf")
    ub[indices + 1] = float("inf")
    lb[indices + 3] = -float("inf")
    ub[indices + 3] = float("inf")

    opt.set_lower_bounds(lb.tolist())
    opt.set_upper_bounds(ub.tolist())

    # make the real parts the correct sign
    coeffs0[indices] = -np.abs(coeffs0[indices])
    coeffs0[indices + 2] = +np.abs(coeffs0[indices + 2])

    # opt.add_inequality_constraint(constraint)
    opt.set_xtol_rel(1e-4)
    opt.set_ftol_rel(1e-4)

    if globalopt:
        optglob = nlopt.opt(nlopt.LN_AUGLAG, num_DoF)
        optglob.set_min_objective(obj)
        optglob.set_lower_bounds(lb.tolist())
        optglob.set_upper_bounds(ub.tolist())
        optglob.add_inequality_constraint(constraint)
        optglob.set_xtol_rel(1e-4)
        optglob.set_maxeval(10000)
        optglob.set_ftol_rel(1e-5)
        optglob.set_local_optimizer(opt)
        coeffs = optglob.optimize(coeffs0)
        rms_error = optglob.last_optimum_value()
    else:
        coeffs = opt.optimize(coeffs0)
        rms_error = opt.last_optimum_value()

    return coeffs, rms_error


def fit_eps_poles_multiple(ws, eps, npoles, lossy, num_tries=100, tolerance_rms=1e-3, globalopt=True):
    # Perform fitting for a specified number of poles.
    # This is just a wrapper routine to run the optimization several times.

    # Initial guess:
    guess = None

    # Run it a number of times.
    best_coeffs = None
    best_rms = np.inf
    for iteration in range(num_tries):
        coeffs, rms_error = fit_eps_poles(ws, eps, npoles, lossy, globalopt, guess)
        if rms_error < best_rms:
            best_coeffs = coeffs.copy()
            best_rms = rms_error
        print(
            f"\t\ton try ({iteration+1}/{num_tries}), got RMS error of {rms_error:.2e}"
        )
        if best_rms < tolerance_rms:
            print(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
            return best_coeffs, best_rms
    print(
        f"\twarning: did not find fit with RMS error under tolerance_rms of {tolerance_rms}"
    )
    print(f"\treturning best fit with RMS error {best_rms:.2f}")
    return best_coeffs, best_rms


def make_medium(poles, name=None):
    """ turns result of optimization into medium """
    model = td.DispersionModel(poles=poles)
    # print(f'medium = tidy3d.Medium(epsilon=tidy3d.DispersionModel(poles={poles}), name="{name}")\n')
    return td.Medium(epsilon=model, name=name)


def make_poles(coeffs):
    """ turns array of coeffs into list of (a, c) poles """
    coeffs_scaled = coeffs / td.constants.HBAR
    avec, cvec = unpack_coeffs(coeffs_scaled)
    poles = [(complex(a), complex(c)) for (a, c) in zip(avec, cvec)]
    return poles


def fit(wavelengths, n_data, k_data=None, num_poles=3, num_tries=100, tolerance_rms=1e-2, plot=True, globalopt=True):

    lossy = True

    if k_data is None:
        k_data = np.zeros_like(n_data)
    else:
        k_data = k_data

    # check if lossy
    if np.allclose(k_data, 0.0):
        lossy = False

    assert len(wavelengths.shape) == 1
    num_data = len(wavelengths)
    assert n_data.shape == wavelengths.shape
    assert k_data.shape == wavelengths.shape

    # Convert from wavelength to eV
    eV = C_0 / wavelengths * 2 * np.pi * HBAR
    eps = (n_data - 1j * k_data) ** 2
    coeffs, min_val = fit_eps_poles_multiple(eV, eps, num_poles, lossy, tolerance_rms=tolerance_rms, num_tries=num_tries)

    model_eps = model(coeffs, eV)
    model_nk = np.sqrt(np.conj(model_eps))
    model_n, model_k = unpack_complex(model_nk)

    if plot:
        plt.plot(wavelengths, n_data, "o", color="black", label="n (data)")
        plt.plot(wavelengths, model_n, "-", color="firebrick", label="n (model)")
        if lossy:
            plt.plot(wavelengths, k_data, "o", color="grey", label="k (data)")
            plt.plot(
                wavelengths, model_k, "-", color="dodgerblue", label="k (model)"
            )
        plt.ylabel("value")
        plt.xlabel("Wavelength (um)")
        plt.title(f"{num_poles} pole fit")
        plt.legend()
        plt.show()

    poles = make_poles(coeffs)
    return poles


def load_nk_file(fname, **kwargs):
    """ Wrapper for np.loadtxt
        https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
    """
    data = np.loadtxt(fname, **kwargs)
    assert (
        len(data.shape) == 2
    ), "data must contain [wavelength, ndata, kdata] in columns"
    assert data.shape[-1] in (2, 3), "data must have either 2 or 3 rows (if k data)"
    num_data = len(data)
    if data.shape[-1] == 2:
        data = np.concatenate((data, np.zeros((num_data, 1))), axis=-1)
    wavelengths, n_data, k_data = data.T
    return wavelengths, n_data, k_data


if __name__ == "__main__":
    fname = "VO2-Povinelli_metal.csv"
    wavelengths, n_data, k_data = load_nk_file(fname, skiprows=1, delimiter=",")
    fits = []

    num_poles = [4]
    for pole_number in num_poles:
        print(f"fitting # poles = {pole_number}")
        poles = fit(wavelengths, n_data, k_data, num_poles=pole_number, num_tries=400, tolerance_rms=.1, plot=True)
        fits.append(poles)

    my_medium = make_medium(fits[0], name="my_medium")
