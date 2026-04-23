"""Multi-frequency adjoint gradient checks for custom dispersive media."""

from __future__ import annotations

import numpy as np

import tidy3d as td
from tidy3d.components.autograd.derivative_utils import DerivativeInfo


def _make_derivative_info(freqs, paths):
    freqs = np.asarray(freqs, float)
    return DerivativeInfo(
        paths=paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        frequencies=list(freqs),
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0 for f in freqs]]]], coords={"x": [0], "y": [0], "z": [0], "f": list(freqs)}
        ),
    )


def _patch_derivative_field_cmp_custom(dJ):
    def _fake(self, E_der_map, spatial_data, dim, sum_over_freqs=True, **kwargs):
        if sum_over_freqs:
            return dJ.sum(axis=-1) / 3.0
        return dJ / 3.0

    return _fake


def test_custom_sellmeier_multifrequency_weights(monkeypatch):
    freqs = np.array([1.5e14, 2.1e14])
    coords = {"x": [0.0], "y": [0.0], "z": [0.0]}

    B = td.SpatialDataArray(np.array([[[0.2]]]), coords=coords)
    C = td.SpatialDataArray(np.array([[[0.3]]]), coords=coords)
    med = td.CustomSellmeier(coeffs=[(B, C)])

    dJ = np.array([[[[1.1 + 0.4j, -0.6 + 0.2j]]]])
    monkeypatch.setattr(
        td.CustomSellmeier, "_derivative_field_cmp_custom", _patch_derivative_field_cmp_custom(dJ)
    )

    paths = [("coeffs", 0, 0), ("coeffs", 0, 1)]
    grads = med._compute_derivatives(_make_derivative_info(freqs, paths))

    expected_gB = 0.0
    expected_gC = 0.0
    for idx, f in enumerate(freqs):
        dJ_f = dJ[..., idx]
        expected_gB += np.real(dJ_f * np.conj(td.Sellmeier._w_B(f, C.values)))
        expected_gC += np.real(dJ_f * np.conj(td.Sellmeier._w_C(f, B.values, C.values)))

    np.testing.assert_allclose(grads[("coeffs", 0, 0)], expected_gB)
    np.testing.assert_allclose(grads[("coeffs", 0, 1)], expected_gC)


def test_custom_pole_residue_multifrequency_weights(monkeypatch):
    freqs = np.array([1.2e14, 1.8e14])
    coords = {"x": [0.0], "y": [0.0], "z": [0.0]}

    eps_inf = td.SpatialDataArray(np.array([[[2.0]]]), coords=coords)
    a1 = td.SpatialDataArray(np.array([[[-1.0 + 0.1j]]]), coords=coords)
    c1 = td.SpatialDataArray(np.array([[[0.5 + 0.2j]]]), coords=coords)
    med = td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1)])

    dJ = np.array([[[[0.4 + 0.2j, -0.1 + 0.3j]]]])
    monkeypatch.setattr(
        td.CustomPoleResidue, "_derivative_field_cmp_custom", _patch_derivative_field_cmp_custom(dJ)
    )

    paths = [("eps_inf",), ("poles", 0, 0), ("poles", 0, 1)]
    grads = med._compute_derivatives(_make_derivative_info(freqs, paths))

    poles_vals = [(np.array(a1.values, dtype=complex), np.array(c1.values, dtype=complex))]
    expected = {}
    for idx, f in enumerate(freqs):
        dJ_f = dJ[..., idx]
        vjps_f = td.PoleResidue._get_vjps_from_params(
            dJ_deps_complex=dJ_f,
            poles_vals=poles_vals,
            omega=2 * np.pi * f,
            requested_paths=paths,
            project_real=False,
        )
        for path, vjp in vjps_f.items():
            expected[path] = expected.get(path, 0.0) + vjp

    for path in paths:
        np.testing.assert_allclose(grads[path], expected[path])


def test_custom_pole_residue_zero_gradient_chunk_preserves_frequency_axis():
    freqs = np.array([1.2e14])
    coords = {"x": np.linspace(-1.0, 1.0, 4), "y": np.linspace(-0.5, 0.5, 3), "z": [0.0]}
    field_coords = {**coords, "f": freqs}

    eps_inf = td.SpatialDataArray(np.full((4, 3, 1), 2.0), coords=coords)
    a1 = td.SpatialDataArray(np.full((4, 3, 1), -1.0 + 0.1j), coords=coords)
    c1 = td.SpatialDataArray(np.full((4, 3, 1), 0.5 + 0.2j), coords=coords)
    med = td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1)])

    zero_field = td.ScalarFieldDataArray(
        np.zeros((4, 3, 1, 1), dtype=complex),
        coords=field_coords,
    )
    derivative_info = DerivativeInfo(
        paths=[("eps_inf",), ("poles", 0, 0), ("poles", 0, 1)],
        E_der_map={"Ex": zero_field, "Ey": zero_field, "Ez": zero_field},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        frequencies=list(freqs),
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: zero_field,
    )

    grads = med._compute_derivatives(derivative_info)

    for path in derivative_info.paths:
        np.testing.assert_allclose(grads[path], 0.0)
        assert grads[path].shape == (4, 3, 1)
