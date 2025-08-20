# tests/test_dispersive_vjps_fd.py
from __future__ import annotations

import numpy as np
import pytest

from tidy3d import C_0, Debye, Drude, Lorentz, PoleResidue, Sellmeier


class _FreqCoord:
    def __init__(self, values):
        self.values = np.array(values, float)


class _DummyDA:
    def __init__(self, values, freqs):
        self.values = np.array(values, np.complex128)
        self.coords = {"f": _FreqCoord(freqs)}

    def sel(self, f):
        freqs = self.coords["f"].values
        idxs = np.where(np.isclose(freqs, f))[0]
        if idxs.size == 0:
            raise ValueError("frequency not found in coords")
        return self.values[int(idxs[0])]


class _DI:
    def __init__(self, freqs, paths):
        self.frequencies = np.array(freqs, float)
        self.paths = paths
        self.E_der_map = None
        self.bounds = None


def _monkey_derivative_eps_complex_volume(inst, dJ, freqs):
    """Monkey-patch instance method without changing production code."""

    def _fake(self, E_der_map=None, bounds=None):
        return _DummyDA(dJ, freqs)

    # make a validated deep copy to avoid adding extra attributes to the original instance
    inst_copy = inst.copy(deep=True, validate=True)
    object.__setattr__(
        inst_copy,
        "_derivative_eps_complex_volume",
        _fake.__get__(inst_copy, inst_copy.__class__),
    )
    return inst_copy


def _phi(model, freqs, dJ):
    eps = model.eps_model(np.array(freqs, float))
    # real inner product: Re⟨dJ, eps⟩
    return float(np.sum(np.real(dJ) * np.real(eps) + np.imag(dJ) * np.imag(eps)))


def _central_fd(model, theta0, set_theta, freqs, dJ, h_scale):
    g = np.zeros_like(theta0, dtype=float)
    for i in range(theta0.size):
        h = max(h_scale, 1e-8 * (abs(theta0[i]) + 1.0))
        t_plus = theta0.copy()
        t_plus[i] += h
        t_minus = theta0.copy()
        t_minus[i] -= h
        m_plus = set_theta(model, t_plus)
        p_plus = _phi(m_plus, freqs, dJ)
        m_minus = set_theta(model, t_minus)
        p_minus = _phi(m_minus, freqs, dJ)
        g[i] = (p_plus - p_minus) / (2 * h)
    return g


def _impl_grad(model, di, paths):
    # vectorize dict in 'paths' order
    gd = model._compute_derivatives(di)
    out = []
    for p in paths:
        v = gd[p]
        v = np.array(v, dtype=float)  # uniform mediums -> scalar
        out.append(np.sum(v))  # sum if array, scalar otherwise
    return np.array(out, dtype=float)


def _pack_sellmeier(m):
    B = [float(b) for (b, _c) in m.coeffs]
    C = [float(c) for (_b, c) in m.coeffs]
    return np.array(B + C, float)


def _set_sellmeier(m, theta):
    N = len(m.coeffs)
    coeffs = [(float(theta[i]), float(theta[N + i])) for i in range(N)]
    return m.updated_copy(coeffs=tuple(coeffs))


def _paths_sellmeier(m):
    N = len(m.coeffs)
    return [("coeffs", i, 0) for i in range(N)] + [("coeffs", i, 1) for i in range(N)]


def _pack_lorentz(m):
    de, f0, dl = zip(*m.coeffs) if m.coeffs else ([], [], [])
    return np.array([float(m.eps_inf), *map(float, de), *map(float, f0), *map(float, dl)])


def _set_lorentz(m, theta):
    N = len(m.coeffs)
    eps_inf = float(theta[0])
    de = theta[1 : 1 + N]
    f0 = theta[1 + N : 1 + 2 * N]
    dl = theta[1 + 2 * N : 1 + 3 * N]
    coeffs = [(float(de[i]), float(f0[i]), float(dl[i])) for i in range(N)]
    return m.updated_copy(eps_inf=eps_inf, coeffs=tuple(coeffs))


def _paths_lorentz(m):
    N = len(m.coeffs)
    return (
        [("eps_inf",)]
        + [("coeffs", i, 0) for i in range(N)]
        + [("coeffs", i, 1) for i in range(N)]
        + [("coeffs", i, 2) for i in range(N)]
    )


def _pack_drude(m):
    fp, dl = zip(*m.coeffs) if m.coeffs else ([], [])
    return np.array([float(m.eps_inf), *map(float, fp), *map(float, dl)])


def _set_drude(m, theta):
    N = len(m.coeffs)
    eps_inf = float(theta[0])
    fp = theta[1 : 1 + N]
    dl = theta[1 + N : 1 + 2 * N]
    coeffs = [(float(fp[i]), float(dl[i])) for i in range(N)]
    return m.updated_copy(eps_inf=eps_inf, coeffs=tuple(coeffs))


def _paths_drude(m):
    N = len(m.coeffs)
    return (
        [("eps_inf",)] + [("coeffs", i, 0) for i in range(N)] + [("coeffs", i, 1) for i in range(N)]
    )


def _rand_complex(F, rng):
    return rng.normal(size=F) + 1j * rng.normal(size=F)


@pytest.mark.parametrize("F,N,seed", [(6, 3, 0), (5, 2, 1)])
def test_sellmeier_vjp_fd(F, N, seed):
    rng = np.random.RandomState(seed)
    freqs = np.linspace(2.0e14, 4.0e14, F)
    lam2_min = (C_0 / float(np.max(freqs))) ** 2
    coeffs = [
        (0.2 + 0.5 * abs(rng.randn()), 0.2 * lam2_min * (0.6 + 0.3 * rng.rand())) for _ in range(N)
    ]
    m_base = Sellmeier(coeffs=coeffs)

    dJ = _rand_complex(F, rng)
    m = _monkey_derivative_eps_complex_volume(m_base, dJ, freqs)

    theta0 = _pack_sellmeier(m_base)
    paths = _paths_sellmeier(m_base)
    di = _DI(freqs, paths)

    # FD on immutable base, autograd on patched instance
    g_fd = _central_fd(
        m_base,
        theta0,
        lambda _mm, th: _set_sellmeier(m_base, th),
        freqs,
        dJ,
        h_scale=1e-8,
    )
    g_impl = _impl_grad(m, di, paths)

    np.testing.assert_allclose(g_impl, g_fd, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("F,N,seed", [(6, 2, 4)])
def test_pole_residue_vjp_fd(F, N, seed):
    rng = np.random.RandomState(seed)
    freqs = np.linspace(1.0e14, 3.0e14, F)

    # generate stable real poles (a<0) and real residues
    a_list = []
    c_list = []
    for _ in range(N):
        a_val = -(0.5 + rng.rand()) * 2 * np.pi * 1.0e14
        c_val = (0.5 + rng.rand()) * 2 * np.pi * 1.0e14
        a_list.append(float(a_val))
        c_list.append(float(c_val))
    poles = tuple((a_list[i], c_list[i]) for i in range(N))
    eps_inf = 1.25
    m_base = PoleResidue(eps_inf=eps_inf, poles=poles)

    dJ = _rand_complex(F, rng)
    m = _monkey_derivative_eps_complex_volume(m_base, dJ, freqs)

    def _pack_pr(mm):
        def _to_float(x):
            return float(np.real(np.array(x, dtype=np.complex128)))

        a = [_to_float(a) for a, _ in mm.poles]
        c = [_to_float(c) for _, c in mm.poles]
        return np.array([float(mm.eps_inf), *a, *c], float)

    def _set_pr(mm, theta):
        eps_inf = float(theta[0])
        a = theta[1 : 1 + N]
        c = theta[1 + N : 1 + 2 * N]
        new_poles = tuple((float(a[i]), float(c[i])) for i in range(N))
        return mm.updated_copy(eps_inf=eps_inf, poles=new_poles)

    def _paths_pr(mm):
        return (
            [("eps_inf",)]
            + [("poles", i, 0) for i in range(N)]
            + [("poles", i, 1) for i in range(N)]
        )

    # one-sided PR epsilon to match analytical chain rule used in _compute_derivatives
    def _eps_pr_one_sided(mm, freqs):
        w = 2 * np.pi * np.array(freqs)
        eps = np.array(mm.eps_inf, np.complex128)
        for a, c in mm.poles:
            eps = eps - c / (1j * w + a)
        return eps

    def _phi_pr(mm, freqs, dJ):
        eps = _eps_pr_one_sided(mm, freqs)
        return float(np.sum(np.real(dJ) * np.real(eps) + np.imag(dJ) * np.imag(eps)))

    # FD on base model against one-sided PR epsilon
    theta0 = _pack_pr(m_base)
    paths = _paths_pr(m_base)
    di = _DI(freqs, paths)

    g_fd = np.zeros_like(theta0, dtype=float)
    for i in range(theta0.size):
        h = max(1e-8, 1e-8 * (abs(theta0[i]) + 1.0))
        t_plus = theta0.copy()
        t_plus[i] += h
        t_minus = theta0.copy()
        t_minus[i] -= h
        m_plus = _set_pr(m_base, t_plus)
        m_minus = _set_pr(m_base, t_minus)
        p_plus = _phi_pr(m_plus, freqs, dJ)
        p_minus = _phi_pr(m_minus, freqs, dJ)
        g_fd[i] = (p_plus - p_minus) / (2 * h)

    # autograd mapping on patched instance should match
    g_impl = _impl_grad(m, di, paths)
    np.testing.assert_allclose(g_impl, g_fd, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("F,N,seed", [(7, 2, 2)])
def test_lorentz_vjp_fd(F, N, seed):
    rng = np.random.RandomState(seed)
    freqs = np.linspace(1.0e14, 3.0e14, F)
    eps_inf = 2.2
    coeffs = []
    for _ in range(N):
        de = 0.1 + 0.4 * abs(rng.randn())
        f0 = 4.0e14 + 0.2e14 * rng.rand()
        dl = 0.1e14 + 0.2e14 * abs(rng.randn())
        coeffs.append((de, f0, dl))
    m_base = Lorentz(eps_inf=eps_inf, coeffs=coeffs)

    dJ = _rand_complex(F, rng)
    m = _monkey_derivative_eps_complex_volume(m_base, dJ, freqs)

    theta0 = _pack_lorentz(m_base)
    paths = _paths_lorentz(m_base)
    di = _DI(freqs, paths)

    g_fd = _central_fd(
        m_base,
        theta0,
        lambda _mm, th: _set_lorentz(m_base, th),
        freqs,
        dJ,
        h_scale=1e-8,
    )
    g_impl = _impl_grad(m, di, paths)

    np.testing.assert_allclose(g_impl, g_fd, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("F,N,seed", [(6, 2, 3)])
def test_drude_vjp_fd(F, N, seed):
    rng = np.random.RandomState(seed)
    freqs = np.linspace(1.0e14, 3.0e14, F)
    eps_inf = 1.5
    coeffs = []
    for _ in range(N):
        fp = 2.5e14 + 0.3e14 * rng.rand()
        dl = 0.05e14 + 0.1e14 * abs(rng.randn())
        coeffs.append((fp, dl))
    m_base = Drude(eps_inf=eps_inf, coeffs=coeffs)

    dJ = _rand_complex(F, rng)
    m = _monkey_derivative_eps_complex_volume(m_base, dJ, freqs)

    theta0 = _pack_drude(m_base)
    paths = _paths_drude(m_base)
    di = _DI(freqs, paths)

    g_fd = _central_fd(
        m_base,
        theta0,
        lambda _mm, th: _set_drude(m_base, th),
        freqs,
        dJ,
        h_scale=1e-8,
    )
    g_impl = _impl_grad(m, di, paths)

    np.testing.assert_allclose(g_impl, g_fd, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("F,N,seed", [(6, 2, 6)])
def test_debye_vjp_fd(F, N, seed):
    rng = np.random.RandomState(seed)
    freqs = np.linspace(1.0e14, 3.0e14, F)
    eps_inf = 1.2 + 0.5 * rng.rand()
    coeffs = []
    for _ in range(N):
        de = 0.1 + 0.4 * abs(rng.randn())
        tau = 0.5e-14 + 0.5e-14 * rng.rand()
        coeffs.append((de, tau))
    m_base = Debye(eps_inf=eps_inf, coeffs=coeffs)

    dJ = _rand_complex(F, rng)
    m = _monkey_derivative_eps_complex_volume(m_base, dJ, freqs)

    def _pack_debye(mm):
        de, tau = zip(*mm.coeffs) if mm.coeffs else ([], [])
        return np.array([float(mm.eps_inf), *map(float, de), *map(float, tau)])

    def _set_debye(mm, theta):
        eps_inf = float(theta[0])
        de = theta[1 : 1 + N]
        tau = theta[1 + N : 1 + 2 * N]
        coeffs = [(float(de[i]), float(tau[i])) for i in range(N)]
        return mm.updated_copy(eps_inf=eps_inf, coeffs=tuple(coeffs))

    def _paths_debye(mm):
        N = len(mm.coeffs)
        return (
            [("eps_inf",)]
            + [("coeffs", i, 0) for i in range(N)]
            + [("coeffs", i, 1) for i in range(N)]
        )

    theta0 = _pack_debye(m_base)
    paths = _paths_debye(m_base)
    di = _DI(freqs, paths)

    # custom FD to respect tau>0 constraint
    g_fd = np.zeros_like(theta0, dtype=float)
    Nloc = N
    tau_start = 1 + Nloc
    for i in range(theta0.size):
        # base step as in other tests
        h = max(1e-8, 1e-8 * (abs(theta0[i]) + 1.0))
        t_plus = theta0.copy()
        t_minus = theta0.copy()
        # for tau entries, keep strictly positive and use tiny relative step
        if i >= tau_start:
            tau0 = theta0[i]
            h_tau = max(1e-6 * tau0, 1e-20)
            h = min(h, h_tau)
        t_plus[i] += h
        t_minus[i] -= h
        m_plus = _set_debye(m_base, t_plus)
        m_minus = _set_debye(m_base, t_minus)
        p_plus = _phi(m_plus, freqs, dJ)
        p_minus = _phi(m_minus, freqs, dJ)
        g_fd[i] = (p_plus - p_minus) / (2 * h)
    g_impl = _impl_grad(m, di, paths)

    np.testing.assert_allclose(g_impl, g_fd, rtol=5e-6, atol=5e-7)
