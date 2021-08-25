import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import nlopt
import glob

'''
A quick and dirty material model fitting script.

About half the code (bottom half) currently is just reading in data
and managing user specifications, while the other (top half) is related
to optimizing the fit using NLopt.

All frequency units are in eV currently to keep typical values on the order of unity.
This simplifies operator validation of the fit parameters, and also random initialization
of model parameters.
'''

EPSILON_0 = np.float32(8.85418782e-18)         # vacuum permittivity [F/um]
MU_0 = np.float32(1.25663706e-12)              # vacuum permeability [H/um]
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)            # speed of light in vacuum [um/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)              # vacuum impedance
HBAR = 6.582119569e-16                         # reduced Planck constant [eV*s]

def model(coeffs, w):
    # coeffs: A length 4*n list or numpy array, corresponding to data for n poles
    #         Each set of 4 elements in coeffs is (ar, ai, cr, ci); the real and
    #         imaginary parts of a (pole) and c (residue).
    # w: omega, in the same units as a and c.
    n = len(coeffs)//4
    e = 1
    for i in range(n):
        ar = coeffs[4*i+0]
        ai = coeffs[4*i+1]
        cr = coeffs[4*i+2]
        ci = coeffs[4*i+3]
        a = ar+1j*ai
        c = cr+1j*ci
        e += c/(1j*w-a) + c.conjugate()/(1j*w-a.conjugate())
    return e

def vec_to_ac(v):
    # Convert from a length 4*n vector of pole/residue data
    # to a list of tuple pairs of complex numbers (a,c).
    # This is just a convenience function to re-shuffle the data
    # to be easier to use.
    npoles = len(v)//4
    acpoles = []
    for i in range(npoles):
        a = v[4*i+0] + 1j * v[4*i+1]
        c = v[4*i+2] + 1j * v[4*i+3]
        acpoles.append((a,c))
    return acpoles

def ac_to_vec(ac):
    # Convert from a length n vector of tuples of complex numbers (a,c)
    # to a vector of list of length 4*n of (ar, ai, cr, ci).
    # This is just a convenience function to re-shuffle the data
    # to be easier to use.
    v = []
    for (a,c) in ac:
        v.append(a.real)
        v.append(a.imag)
        v.append(c.real)
        v.append(c.imag)
    return v

def fit_eps_poles_1pass(eV, eps, npoles, lossy, globalopt=False, guess=None):
    # Performs the NLopt fitting for a pole/residue model
    # with a specified number of poles.
    
    # eV:  vector of frequencies
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

    def constraint(coeffs, grad):
        # Evaluates the nonlinear stability criterion of
        #   Hongjin Choi, Jae-Woo Baek, and Kyung-Young Jung,
        #   "Comprehensive Study on Numerical Aspects of Modified
        #    Lorentz Model Based Dispersive FDTD Formulations,"
        #   IEEE TAP 2019.
        #
        # coeffs: length 4*n vector of (ar, ai, cr, ci).
        # grad: The gradient to return (not used).
        n = len(coeffs)//4
        e = 0.
        for i in range(n):
            ar = coeffs[4*i+0]
            ai = coeffs[4*i+1]
            cr = coeffs[4*i+2]
            ci = coeffs[4*i+3]
            prstar = ar*cr + ai*ci
            val = 2*prstar*ar - cr*(ar*ar + ai*ai)
            if val < 0:
                e -= val
        return e

    def obj(x, grad):
        # Evaluates the objective function for the fit.
        #
        # x: length 4*n vector of (ar, ai, cr, ci).
        # grad: The gradient to return (not used).
        
        # Compute error metric. The eV**2 weighting factor is empirical; change as needed.
        err = (model(x, eV) - eps) * eV**2
        nrm = np.linalg.norm(err)#, ord=np.inf)
        cons = constraint(x, grad)
        print(nrm, cons)
        return nrm #+ 1e2*cons

    nn = npoles*4
    if guess is None:
        x0 = np.random.rand(nn)
    else:
        x0 = ac_to_vec(guess)
    
    if globalopt:
        method = nlopt.LN_NELDERMEAD
    else:
        method = nlopt.LN_SBPLX
    
    opt = nlopt.opt(method, nn)
    opt.set_min_objective(obj)
    ub = []
    lb = []
    for i in range(npoles):
        # Re(a)
        if not lossy:
            lb.append(0)
            x0[4*i+0] = 0
        else:
            lb.append(-float('inf'))
        ub.append(0)
        # Im(a)
        lb.append(-float('inf'))
        ub.append(float('inf'))
        # Re(c)
        lb.append(0)
        if not lossy:
            ub.append(0)
            x0[4*i+2] = 0
        else:
            ub.append(float('inf'))
        # Im(c)
        lb.append(-float('inf'))
        ub.append(float('inf'))
        
        if x0[4*i+0] > 0:
            x0[4*i+0] *= -1
        if x0[4*i+2] < 0:
            x0[4*i+2] *= -1
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    #opt.add_inequality_constraint(constraint)
    if globalopt:
        opt.set_xtol_rel(1e-4)
    else:
        opt.set_xtol_rel(1e-4)
    opt.set_ftol_rel(1e-4)
    
    if globalopt:
        optglob = nlopt.opt(nlopt.LN_AUGLAG, nn)
        optglob.set_min_objective(obj)
        optglob.set_lower_bounds(lb)
        optglob.set_upper_bounds(ub)
        optglob.add_inequality_constraint(constraint)
        optglob.set_xtol_rel(1e-4)
        optglob.set_maxeval(100000)
        optglob.set_ftol_rel(1e-5)
        optglob.set_local_optimizer(opt)
        x = optglob.optimize(x0)
        minf = optglob.last_optimum_value()
    else:
        x = opt.optimize(x0)
        minf = opt.last_optimum_value()
    
    print("optimum at ", vec_to_ac(x))
    print("minimum value = ", minf)
    #print("result code = ", opt.last_optimize_result())
    
    return vec_to_ac(x), minf

def fit_eps_poles(eV, eps, npoles, lossy):
    # Perform fitting for a specified number of poles.
    # This is just a wrapper routine to run the optimization several times.
    
    # Initial guess:
    ac = None
    #ac = [(13.587564477728654j, -7.6129270108833165j)]
    
    # Run it a number of times.
    for iter in range(4):
        ac, minf = fit_eps_poles_1pass(eV, eps, npoles, lossy, False, ac)
        ac, minf = fit_eps_poles_1pass(eV, eps, npoles, lossy, True, ac)
    return ac, minf

def perform_fit(matname, variant, ndata, kdata, nktol = 1e-3, npoles = None):
    # matname: (string) Name of material.
    # variant: (string) Variant of the material.
    # ndata: Size n x 2 array of n vs. wavelength.
    # kdata: Size n x 2 array of k vs. wavelength. May be None for a lossless fit.
    # nktol: fit tolerance.
    # npoles: Number of poles to use, or None to automatically choose a number.
    
    wl = ndata[:,0]
    n  = ndata[:,1]
    lossy = True
    if kdata is not None:
        k = kdata[:,1]
    else:
        k = np.zeros(n.shape)
        lossy = False
        
    # Convert from wavelength to eV
    eV = C_0/wl * 2*math.pi*HBAR
    eps = (n-1j*k)**2
    if npoles is None:
        for npoles in range(1,10):
            ac, err = fit_eps_poles(eV, eps, npoles, lossy)
            if err < nktol:
                break
    else:
        ac, err = fit_eps_poles(eV, eps, npoles, lossy)
    coeffs = np.array(ac_to_vec(ac))
    meps = model(coeffs, eV)
    mn = np.sqrt(meps).real
    mk = -np.sqrt(meps).imag
    
    x = wl
    plt.xlabel('Wavelength (um)')

    if lossy:
        plt.plot(x, n, 'o', x, k, 'o', x, mn, '-', x, mk, '-')
        plt.legend(['n','k','n model','k model'])
    else:
        plt.plot(x, n, 'o', x, mn, '-')
    plt.show()

def csv_to_nkdata(filename):
    # Read a CSV file (in a format similar to those from refractiveindex.info)
    # and return two numpy arrays of size n x 2 for ndata and kdata (see above).
    
    nkdatalines = []
    ndatalines = []
    kdatalines = []
    state = 0
    wl_scale = 1
    with open(filename, 'rb') as fp:
        content = fp.readlines()
        for line in content:
            line = line.decode("utf-8").strip()
            if 0 == state:
                if 'wl,n' == line:
                    state = 1
                elif 'wl,n,k' == line:
                    state = 10
                elif 'wavelength_nm,real_index,imaginary_index' == line:
                    state = 10
                    wl_scale = 1e-3
                elif 'wavelength_m,real_index,imaginary_index' == line:
                    state = 10
                    wl_scale = 1e6
            elif 1 == state:
                if 'wl,k' == line:
                    state = 2
                else:
                    ndatalines.append(line)
            elif 2 == state:
                kdatalines.append(line)
            elif 10 == state:
                nkdatalines.append(line)
    if 10 == state:
        nkdata = np.genfromtxt(nkdatalines, delimiter=',')
        ndata = nkdata[:,0:2]
        kdata = nkdata[:,(0,2)]
        ndata[:,0] *= wl_scale
        kdata[:,0] *= wl_scale
    else:
        ndata = np.genfromtxt(ndatalines, delimiter=',')
        ndata[:,0] *= wl_scale
        kdata = None
        if len(kdatalines) > 0:
            kdata = np.genfromtxt(kdatalines, delimiter=',')
            kdata[:,0] *= wl_scale
    if kdata is not None and 0 == np.count_nonzero(kdata[:,1]):
        kdata = None
    return (ndata, kdata)

def fit_file(filename, npoles = None, wlrange = None):
    # Perform fit on a given CSV file.
    
    nktol = 1e-3
    ind = filename[0:-4].rfind('-')
    matname = filename[0:ind]
    variant = filename[ind+1:-4]
    (ndata, kdata) = csv_to_nkdata(filename)

    if wlrange is not None:
        istart = np.searchsorted(ndata[:, 0], wlrange[0], side = 'right')
        iend = np.searchsorted(ndata[:, 0], wlrange[1], side = 'left')
        ndata = ndata[istart:iend+1,:]
        if kdata is not None:
            kdata = kdata[istart:iend+1,:]

    perform_fit(matname, variant, ndata, kdata, nktol, npoles)

def fit_all():
    # Run fits on all CSV files in this directory
    for filename in glob.glob('*.csv'):
        fit_file(filename)

#fit_file('mat-cladding.csv', 1, wlrange = (0.1, 1))
# fit_file('Si-Li1993_293K.csv', 2)
#fit_file('metal-neg.csv', 1)
#fit_file('Si-Green2008.csv', 4)

fit_file("waveguide-material.csv", 2)
