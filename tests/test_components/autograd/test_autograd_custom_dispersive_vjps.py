from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np

from tidy3d import (
    C_0,
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomSellmeier,
    Debye,
    Drude,
    Lorentz,
    Sellmeier,
    SpatialDataArray,
)
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.data.data_array import ScalarFieldDataArray


def _coords(shape):
    nx, ny, nz = shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    return {"x": x, "y": y, "z": z}


def _deriv_info(freq):
    # minimal DerivativeInfo matching pattern used in custom pole-residue test
    eps_no = ScalarFieldDataArray([[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [1.0]})
    eps_inf = ScalarFieldDataArray([[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [1.0]})
    return {
        "E_der_map": {},
        "D_der_map": {},
        "E_fwd": {},
        "D_fwd": {},
        "E_adj": {},
        "D_adj": {},
        "eps_data": {},
        "frequencies": [freq],
        "bounds": ((-1, -1, -1), (1, 1, 1)),
        "eps_out": eps_no,
        "eps_in": eps_inf,
        "bounds_intersect": ((-1, -1, -1), (1, 1, 1)),
        "simulation_bounds": ((-2, -2, -2), (2, 2, 2)),
        "updated_epsilon": None,
    }


def _patch_derivative_field_cmp_custom(cls, dJ):
    # Return dJ/3 for each of x,y,z so their sum equals dJ
    def _fake(self, E_der_map, spatial_data, dim, sum_over_freqs=True, **kwargs):
        if sum_over_freqs:
            if dJ.ndim > 3:
                return dJ.sum(axis=-1) / 3.0
            return dJ / 3.0
        dJ_with_freq = dJ[..., None] if dJ.ndim == 3 else dJ
        return dJ_with_freq / 3.0

    return _fake


def _J(eps):
    # Real objective yielding real dJ via holomorphic_grad
    return anp.sum(anp.abs(eps))


def test_custom_sellmeier_vjp():
    rng = np.random.RandomState(0)
    shape = (2, 3, 2)
    coords = _coords(shape)
    # Positive B and C
    B1 = SpatialDataArray(0.2 + 0.5 * rng.rand(*shape), coords=coords)
    C1 = SpatialDataArray(0.3 + 0.5 * rng.rand(*shape), coords=coords)
    B2 = SpatialDataArray(0.2 + 0.5 * rng.rand(*shape), coords=coords)
    C2 = SpatialDataArray(0.3 + 0.5 * rng.rand(*shape), coords=coords)
    med = CustomSellmeier(coeffs=[(B1, C1), (B2, C2)])

    freq = 2.5e14
    lam2 = (C_0 / freq) ** 2
    eps_arr = (
        1.0 + B1.values * lam2 / (lam2 - C1.values) + B2.values * lam2 / (lam2 - C2.values) + 0j
    )
    dJ = np.conj(ag.holomorphic_grad(_J)(eps_arr))

    # Monkeypatch derivative provider
    from tidy3d.components import medium as medium_mod

    _orig = CustomSellmeier._derivative_field_cmp_custom
    try:
        medium_mod.CustomSellmeier._derivative_field_cmp_custom = (
            _patch_derivative_field_cmp_custom(CustomSellmeier, dJ)
        )

        paths = [("coeffs", 0, 0), ("coeffs", 0, 1), ("coeffs", 1, 0), ("coeffs", 1, 1)]
        di = DerivativeInfo(paths=paths, **_deriv_info(freq))
        grads = med._compute_derivatives(di)

        # Expected gradients via weight statics
        gB1 = np.real(dJ * np.conj(Sellmeier._w_B(freq, C1.values)))
        gC1 = np.real(dJ * np.conj(Sellmeier._w_C(freq, B1.values, C1.values)))
        gB2 = np.real(dJ * np.conj(Sellmeier._w_B(freq, C2.values)))
        gC2 = np.real(dJ * np.conj(Sellmeier._w_C(freq, B2.values, C2.values)))

        np.testing.assert_allclose(grads[("coeffs", 0, 0)], gB1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 1)], gC1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 0)], gB2, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 1)], gC2, rtol=5e-6, atol=5e-7)
    finally:
        medium_mod.CustomSellmeier._derivative_field_cmp_custom = _orig


def test_custom_lorentz_vjp():
    rng = np.random.RandomState(1)
    shape = (2, 2, 2)
    coords = _coords(shape)
    eps_inf = SpatialDataArray(1.2 + 0.3 * rng.rand(*shape), coords=coords)
    de1 = SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    f01 = SpatialDataArray(3.0e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    dl1 = SpatialDataArray(0.1e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    de2 = SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    f02 = SpatialDataArray(3.5e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    dl2 = SpatialDataArray(0.1e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    med = CustomLorentz(eps_inf=eps_inf, coeffs=[(de1, f01, dl1), (de2, f02, dl2)])

    freq = 2.0e14

    den1 = Lorentz._den(freq, f01.values, dl1.values)
    den2 = Lorentz._den(freq, f02.values, dl2.values)
    eps_arr = (
        eps_inf.values
        + (de1.values * (f01.values**2)) / den1
        + (de2.values * (f02.values**2)) / den2
    ) + 0j
    dJ = np.conj(ag.holomorphic_grad(_J)(eps_arr))

    from tidy3d.components import medium as medium_mod

    _orig = CustomLorentz._derivative_field_cmp_custom
    try:
        medium_mod.CustomLorentz._derivative_field_cmp_custom = _patch_derivative_field_cmp_custom(
            CustomLorentz, dJ
        )

        paths = [
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 0, 2),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
            ("coeffs", 1, 2),
        ]
        di = DerivativeInfo(paths=paths, **_deriv_info(freq))
        grads = med._compute_derivatives(di)

        g_ei = np.real(dJ)
        g_de1 = np.real(dJ * np.conj(Lorentz._w_de(freq, f01.values, dl1.values)))
        g_f01 = np.real(dJ * np.conj(Lorentz._w_f0(freq, de1.values, f01.values, dl1.values)))
        g_dl1 = np.real(dJ * np.conj(Lorentz._w_delta(freq, de1.values, f01.values, dl1.values)))
        g_de2 = np.real(dJ * np.conj(Lorentz._w_de(freq, f02.values, dl2.values)))
        g_f02 = np.real(dJ * np.conj(Lorentz._w_f0(freq, de2.values, f02.values, dl2.values)))
        g_dl2 = np.real(dJ * np.conj(Lorentz._w_delta(freq, de2.values, f02.values, dl2.values)))

        np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_de1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_f01, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 2)], g_dl1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_de2, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_f02, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 2)], g_dl2, rtol=5e-6, atol=5e-7)
    finally:
        medium_mod.CustomLorentz._derivative_field_cmp_custom = _orig


def test_custom_drude_vjp():
    rng = np.random.RandomState(2)
    shape = (2, 2, 2)
    coords = _coords(shape)
    eps_inf = SpatialDataArray(1.1 + 0.2 * rng.rand(*shape), coords=coords)
    fp1 = SpatialDataArray(2.0e14 + 0.2e14 * rng.rand(*shape), coords=coords)
    dl1 = SpatialDataArray(0.05e14 + 0.05e14 * rng.rand(*shape), coords=coords)
    fp2 = SpatialDataArray(2.5e14 + 0.2e14 * rng.rand(*shape), coords=coords)
    dl2 = SpatialDataArray(0.05e14 + 0.05e14 * rng.rand(*shape), coords=coords)
    med = CustomDrude(eps_inf=eps_inf, coeffs=[(fp1, dl1), (fp2, dl2)])

    freq = 2.2e14

    den1 = Drude._den(freq, dl1.values)
    den2 = Drude._den(freq, dl2.values)
    eps_arr = eps_inf.values - (fp1.values**2) / den1 - (fp2.values**2) / den2 + 0j
    dJ = np.conj(ag.holomorphic_grad(_J)(eps_arr))

    from tidy3d.components import medium as medium_mod

    _orig = CustomDrude._derivative_field_cmp_custom
    try:
        medium_mod.CustomDrude._derivative_field_cmp_custom = _patch_derivative_field_cmp_custom(
            CustomDrude, dJ
        )

        paths = [
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
        ]
        di = DerivativeInfo(paths=paths, **_deriv_info(freq))
        grads = med._compute_derivatives(di)

        g_ei = np.real(dJ)
        g_fp1 = np.real(dJ * np.conj(Drude._w_fp(freq, fp1.values, dl1.values)))
        g_dl1 = np.real(dJ * np.conj(Drude._w_delta(freq, fp1.values, dl1.values)))
        g_fp2 = np.real(dJ * np.conj(Drude._w_fp(freq, fp2.values, dl2.values)))
        g_dl2 = np.real(dJ * np.conj(Drude._w_delta(freq, fp2.values, dl2.values)))

        np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_fp1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_dl1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_fp2, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_dl2, rtol=5e-6, atol=5e-7)
    finally:
        medium_mod.CustomDrude._derivative_field_cmp_custom = _orig


def test_custom_debye_vjp():
    rng = np.random.RandomState(3)
    shape = (2, 2, 2)
    coords = _coords(shape)
    eps_inf = SpatialDataArray(1.05 + 0.2 * rng.rand(*shape), coords=coords)
    de1 = SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    tau1 = SpatialDataArray(0.5e-14 + 0.5e-14 * rng.rand(*shape), coords=coords)
    de2 = SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    tau2 = SpatialDataArray(0.5e-14 + 0.5e-14 * rng.rand(*shape), coords=coords)
    med = CustomDebye(eps_inf=eps_inf, coeffs=[(de1, tau1), (de2, tau2)])

    freq = 1.8e14

    den1 = Debye._den(freq, tau1.values)
    den2 = Debye._den(freq, tau2.values)
    eps_arr = eps_inf.values + de1.values / den1 + de2.values / den2 + 0j
    dJ = np.conj(ag.holomorphic_grad(_J)(eps_arr))

    from tidy3d.components import medium as medium_mod

    _orig = CustomDebye._derivative_field_cmp_custom
    try:
        medium_mod.CustomDebye._derivative_field_cmp_custom = _patch_derivative_field_cmp_custom(
            CustomDebye, dJ
        )

        paths = [
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
        ]
        di = DerivativeInfo(paths=paths, **_deriv_info(freq))
        grads = med._compute_derivatives(di)

        g_ei = np.real(dJ)
        g_de1 = np.real(dJ * np.conj(Debye._w_de(freq, tau1.values)))
        g_tau1 = np.real(dJ * np.conj(Debye._w_tau(freq, de1.values, tau1.values)))
        g_de2 = np.real(dJ * np.conj(Debye._w_de(freq, tau2.values)))
        g_tau2 = np.real(dJ * np.conj(Debye._w_tau(freq, de2.values, tau2.values)))

        np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_de1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_tau1, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_de2, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_tau2, rtol=5e-6, atol=5e-7)
    finally:
        medium_mod.CustomDebye._derivative_field_cmp_custom = _orig
