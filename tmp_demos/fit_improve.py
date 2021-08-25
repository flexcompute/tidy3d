import sys
import numpy as np
import matplotlib.pyplot as plt
import nlopt
import tidy3d as td

# import glob

"""
A quick and dirty material model fitting script.

About half the code (bottom half) currently is just reading in data
and managing user specifications, while the other (top half) is related
to optimizing the fit using NLopt.

All frequency units are in eV currently to keep typical values on the order of unity.
This simplifies operator validation of the fit parameters, and also random initialization
of model parameters.
"""

EPSILON_0 = np.float32(8.85418782e-18)  # vacuum permittivity [F/um]
MU_0 = np.float32(1.25663706e-12)  # vacuum permeability [H/um]
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum [um/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # vacuum impedance
HBAR = 6.582119569e-16  # reduced Planck constant [eV*s]


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

    def pole_contribution(w, a, c):
        w_ = w[:, None]
        part1 = c / (1j * w_ - a)
        part2 = np.conj(c) / (1j * w_ - np.conj(a))
        return part1 + part2

    a, c = unpack_coeffs(coeffs)
    pole_contributions = pole_contribution(w, a, c)
    return 1.0 + np.sum(pole_contributions, axis=-1)


def fit_eps_poles(ws, eps, npoles, lossy, globalopt=False, guess=None):
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

        def pole_contribution(a, c):
            ar, ai = unpack_complex(a)
            cr, ci = unpack_complex(c)
            prstar = ar * cr + ai * ci
            res = 2 * prstar * ar - cr * (ar * ar + ai * ai)
            res[res >= 0] == 0
            return -res

        pole_contributions = pole_contribution(a, c)
        return np.sum(pole_contributions)

    def obj(coeffs, grad):
        # Evaluates the objective function for the fit.
        #
        # x: length 4*n vector of (ar, ai, cr, ci).
        # grad: The gradient to return (not used).

        # Compute error metric. The eV**2 weighting factor is empirical; change as needed.
        err = (model(coeffs, ws) - eps) * (ws ** -1)
        nrm = np.linalg.norm(err)
        cons = constraint(coeffs, grad)
        return nrm  + 1e2*cons

    num_DoF = npoles * 4
    if guess is None or True:  # not, change to accept guess
        coeffs0 = np.random.random(num_DoF)
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

    # make the real parts positive
    coeffs0[indices] = np.abs(coeffs0[indices])
    coeffs0[indices + 2] = np.abs(coeffs0[indices + 2])

    # opt.add_inequality_constraint(constraint)
    opt.set_xtol_rel(1e-4)
    opt.set_ftol_rel(1e-4)

    if globalopt:
        optglob = nlopt.opt(nlopt.LN_AUGLAG, num_DoF)
        optglob.set_min_objective(obj)
        optglob.set_lower_bounds(lb)
        optglob.set_upper_bounds(ub)
        optglob.add_inequality_constraint(constraint)
        optglob.set_xtol_rel(1e-4)
        optglob.set_maxeval(100000)
        optglob.set_ftol_rel(1e-5)
        optglob.set_local_optimizer(opt)
        coeffs = optglob.optimize(coeffs0)
        min_val = optglob.last_optimum_value()
    else:
        coeffs = opt.optimize(coeffs0)
        min_val = opt.last_optimum_value()

    return coeffs, min_val


def fit_eps_poles_multiple(ws, eps, npoles, lossy, num_tries=100, tolerance=1e-2):
    # Perform fitting for a specified number of poles.
    # This is just a wrapper routine to run the optimization several times.

    # Initial guess:
    guess = None

    # Run it a number of times.
    best_coeffs = None
    best_val = np.inf
    for iter in range(num_tries):
        for globalopt in (True, False):
            coeffs, min_val = fit_eps_poles(ws, eps, npoles, lossy, globalopt, guess)
            if min_val < best_val:
                best_coeffs = coeffs.copy()
                best_val = min_val

            if best_val < tolerance:
                return best_coeffs, best_val
    return best_coeffs, best_val

def make_material(coeffs):
    avec, cvec = unpack_coeffs(coeffs)
    h = td.constants.HBAR
    avec /= h
    cvec /= h

    poles = [((a, c)) for (a, c) in zip(avec, cvec)]
    model = td.DispersionModel(poles)
    return td.Medium(epsilon=model)


def perform_fit(matname, variant, ndata, kdata, nktol=1e-3, npoles=None):
    # matname: (string) Name of material.
    # variant: (string) Variant of the material.
    # ndata: Size n x 2 array of n vs. wavelength.
    # kdata: Size n x 2 array of k vs. wavelength. May be None for a lossless fit.
    # nktol: fit tolerance.
    # npoles: Number of poles to use, or None to automatically choose a number.

    wavelengths = ndata[:, 0]
    n = ndata[:, 1]
    lossy = True
    if kdata is not None:
        k = kdata[:, 1]
    else:
        k = np.zeros(n.shape)
        lossy = False

    # Convert from wavelength to eV
    eV = C_0 / wavelengths * 2 * np.pi * HBAR
    eps = (n - 1j * k) ** 2
    if npoles is None:
        for npoles in range(1, 10):
            coeffs, min_val = fit_eps_poles_multiple(eV, eps, npoles, lossy)
            if err < nktol:
                break
    else:
        coeffs, min_val = fit_eps_poles_multiple(eV, eps, npoles, lossy)

    model_eps = model(coeffs, eV)
    model_nk = np.sqrt(np.conj(model_eps))
    model_n, model_k = unpack_complex(model_nk)

    plt.plot(wavelengths, n, "o", color='black', label='n (data)')
    plt.plot(wavelengths, model_n, "-", color='firebrick', label='n (model)')
    if lossy:
        plt.plot(wavelengths, k, "o", color='grey', label='k (data)')
        plt.plot(wavelengths, model_k, "-", color='dodgerblue', label='k (model)')        
    plt.ylabel('value')
    plt.xlabel("Wavelength (um)")
    plt.title(f'{npoles} pole fit of "{matname}"')
    plt.legend()
    plt.show()
    # return make_material(coeffs)


def csv_to_nkdata(filename):
    # Read a CSV file (in a format similar to those from refractiveindex.info)
    # and return two numpy arrays of size n x 2 for ndata and kdata (see above).

    nkdatalines = []
    ndatalines = []
    kdatalines = []
    state = 0
    wl_scale = 1
    with open(filename, "rb") as fp:
        content = fp.readlines()
        for line in content:
            line = line.decode("utf-8").strip()
            if 0 == state:
                if "wl,n" == line:
                    state = 1
                elif "wl,n,k" == line:
                    state = 10
                elif "wavelength_nm,real_index,imaginary_index" == line:
                    state = 10
                    wl_scale = 1e-3
                elif "wavelength_m,real_index,imaginary_index" == line:
                    state = 10
                    wl_scale = 1e6
            elif 1 == state:
                if "wl,k" == line:
                    state = 2
                else:
                    ndatalines.append(line)
            elif 2 == state:
                kdatalines.append(line)
            elif 10 == state:
                nkdatalines.append(line)
    if 10 == state:
        nkdata = np.genfromtxt(nkdatalines, delimiter=",")
        ndata = nkdata[:, 0:2]
        kdata = nkdata[:, (0, 2)]
        ndata[:, 0] *= wl_scale
        kdata[:, 0] *= wl_scale
    else:
        ndata = np.genfromtxt(ndatalines, delimiter=",")
        ndata[:, 0] *= wl_scale
        kdata = None
        if len(kdatalines) > 0:
            kdata = np.genfromtxt(kdatalines, delimiter=",")
            kdata[:, 0] *= wl_scale
    if kdata is not None and 0 == np.count_nonzero(kdata[:, 1]):
        kdata = None
    return (ndata, kdata)


def fit_file(filename, npoles=None, wlrange=None):
    # Perform fit on a given CSV file.

    nktol = 1e-3
    ind = filename[0:-4].rfind("-")
    matname = filename[0:ind]
    variant = filename[ind + 1 : -4]
    (ndata, kdata) = csv_to_nkdata(filename)

    if wlrange is not None:
        istart = np.searchsorted(ndata[:, 0], wlrange[0], side="right")
        iend = np.searchsorted(ndata[:, 0], wlrange[1], side="left")
        ndata = ndata[istart : iend + 1, :]
        if kdata is not None:
            kdata = kdata[istart : iend + 1, :]

    perform_fit(matname, variant, ndata, kdata, nktol, npoles)

def fit_all():
    # Run fits on all CSV files in this directory
    for filename in glob.glob("*.csv"):
        fit_file(filename)

fit_file("cladding-material.csv", 2)
fit_file("waveguide-material.csv", 1)

# fit_file('mat-cladding.csv', 1, wlrange = (0.1, 1))
# fit_file('Si-Li1993_293K.csv', 2)
# fit_file('metal-neg.csv', 1)
# fit_file('Si-Green2008.csv', 4)
