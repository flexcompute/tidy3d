from __future__ import annotations

import numpy as np

import tidy3d as td
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.medium import AnisotropicMedium
from tidy3d.constants import EPSILON_0


def _scalar_field(value: complex, coords: dict[str, list[float]]):
    """Helper to build a ``ScalarFieldDataArray`` with a single sample."""

    data = np.array([[[[value]]]], dtype=np.complex128)
    return td.ScalarFieldDataArray(data, coords=coords)


def _derivative_info(paths: list[tuple[str, ...]], freq: float):
    """Construct a ``DerivativeInfo`` with synthetic field products."""

    coords = {"x": [0.0], "y": [0.0], "z": [0.0], "f": [freq]}
    ex_val = 1.2 + 0.4j
    ey_val = -0.8 + 0.1j
    ez_val = 0.5 - 0.3j

    field_map = {
        "Ex": _scalar_field(ex_val, coords),
        "Ey": _scalar_field(ey_val, coords),
        "Ez": _scalar_field(ez_val, coords),
    }

    eps_no = _scalar_field(1.0, coords)
    eps_inf = _scalar_field(2.0, coords)

    return DerivativeInfo(
        paths=paths,
        E_der_map=field_map,
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        frequencies=np.array([freq]),
        bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        bounds_intersect=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        simulation_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        eps_out=eps_no,
        eps_in=eps_inf,
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )


def test_anisotropic_medium_axis_gradients_uniform():
    """Gradients respect axis-specific contributions for diagonal anisotropy."""

    freq = 2.0e14
    paths = [("xx", "permittivity"), ("yy", "conductivity"), ("zz", "permittivity")]
    info = _derivative_info(paths=paths, freq=freq)

    medium = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2.0),
        yy=td.Medium(permittivity=2.5),
        zz=td.Medium(permittivity=3.0),
    )

    grads = medium._compute_derivatives(info)

    ex_real = np.real(info.E_der_map["Ex"].values).item()
    ey_imag = np.imag(info.E_der_map["Ey"].values).item()
    ez_real = np.real(info.E_der_map["Ez"].values).item()

    assert np.isclose(grads[("xx", "permittivity")], ex_real)
    assert np.isclose(grads[("zz", "permittivity")], ez_real)

    expected_sigma = -ey_imag / (2.0 * np.pi * freq * EPSILON_0)
    assert np.isclose(grads[("yy", "conductivity")], expected_sigma)


def _linear_variation_coords(shape: tuple[int, int, int]):
    coords = {
        "x": np.linspace(-1.0, 1.0, shape[0]),
        "y": np.linspace(-1.0, 1.0, shape[1]),
        "z": np.linspace(-1.0, 1.0, shape[2]),
    }
    X, Y, Z = np.meshgrid(coords["x"], coords["y"], coords["z"], indexing="ij")
    factors = 1 + 0.2 * (X + Y + Z) / 3.0
    return coords, factors


def test_custom_anisotropic_medium_axis_gradients():
    """Custom anisotropic medium delegates gradients per axis with spatial data."""

    freq = 1.5e14
    paths = [("xx", "permittivity"), ("yy", "permittivity"), ("zz", "permittivity")]
    info = _derivative_info(paths=paths, freq=freq)

    coords, factors = _linear_variation_coords((3, 4, 5))

    def _make_custom_medium(base_val):
        values = base_val * factors
        data = td.SpatialDataArray(values, coords=coords)
        return td.CustomMedium(permittivity=data)

    medium = td.CustomAnisotropicMedium(
        xx=_make_custom_medium(1.8),
        yy=_make_custom_medium(2.0),
        zz=_make_custom_medium(2.2),
    )

    grads = medium._compute_derivatives(info)

    for comp_name in ("xx", "yy", "zz"):
        comp_info = AnisotropicMedium._component_derivative_info(info, comp_name)
        comp_grad = medium.components[comp_name]._compute_derivatives(comp_info)
        np.testing.assert_allclose(grads[(comp_name, "permittivity")], comp_grad[("permittivity",)])


def test_anisotropic_medium_conductivity_uses_projected_d_map():
    """Conductivity gradients keep only the axis D component."""

    freq = 2.0e14
    paths = [("yy", "conductivity")]

    coords = {"x": [0.0], "y": [0.0], "z": [0.0], "f": [freq]}
    e_map = {
        "Ex": _scalar_field(1.0 + 0.0j, coords),
        "Ey": _scalar_field(2.0 - 0.5j, coords),
        "Ez": _scalar_field(3.0 + 0.0j, coords),
    }
    d_map = {
        "Ex": _scalar_field(1.1 + 0.0j, coords),
        "Ey": _scalar_field(2.2 + 0.0j, coords),
        "Ez": _scalar_field(3.3 + 0.0j, coords),
    }

    eps_no = _scalar_field(1.0, coords)
    eps_inf = _scalar_field(2.0, coords)

    info = DerivativeInfo(
        paths=paths,
        E_der_map=e_map,
        D_der_map=d_map,
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        frequencies=np.array([freq]),
        bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        bounds_intersect=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        simulation_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        eps_out=eps_no,
        eps_in=eps_inf,
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    medium = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2.0),
        yy=td.Medium(permittivity=2.5, conductivity=0.1),
        zz=td.Medium(permittivity=3.0),
    )

    comp_info = AnisotropicMedium._component_derivative_info(info, "yy")

    assert np.allclose(comp_info.D_der_map["Ex"].values, 0.0)
    assert np.allclose(comp_info.D_der_map["Ez"].values, 0.0)
    assert np.allclose(comp_info.D_der_map["Ey"].values, d_map["Ey"].values)

    grads = medium.components["yy"]._compute_derivatives(comp_info)
    expected_sigma = -np.imag(e_map["Ey"].values).item() / (2.0 * np.pi * freq * EPSILON_0)

    assert np.isclose(grads[("conductivity",)], expected_sigma)
