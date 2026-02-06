"""Regression tests comparing mesh-based derivatives to legacy implementations."""

from __future__ import annotations

import copy

import numpy as np
import numpy.testing as npt

import tidy3d as td


class DummyDerivativeInfo:
    """Minimal derivative info stub used for geometry unit tests."""

    def __init__(self, grad_func, paths):
        self.paths = paths
        self._grad_func = grad_func
        self.frequencies = [200e12]
        self.eps_in = 12.0
        self._spacing = 0.05
        self.simulation_bounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
        self.bounds = self.simulation_bounds
        self.bounds_intersect = self.simulation_bounds
        self.interpolators = None

    def adaptive_vjp_spacing(self) -> float:
        return self._spacing

    def create_interpolators(self, dtype=None):
        return {}

    def updated_copy(self, **kwargs):
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        new_info = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new_info, key, value)
        return new_info

    @property
    def wavelength_min(self) -> float:
        return td.C_0 / max(self.frequencies)

    def evaluate_gradient_at_points(
        self,
        spatial_coords=None,
        normals=None,
        perps1=None,
        perps2=None,
        interpolators=None,
    ):
        coords = spatial_coords if spatial_coords is not None else np.zeros((0, 3))
        return self._grad_func(coords)


def linear_grad(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=float)
    return points[:, 0] + 0.5 * points[:, 1] - 0.25 * points[:, 2]


def _assert_mesh_legacy_match(geometry: td.Geometry, derivative_info: DummyDerivativeInfo) -> None:
    derivative_info.bounds = geometry.bounds
    derivative_info.bounds_intersect = geometry.bounds
    mesh_vjps = geometry._compute_derivatives_via_mesh(derivative_info)
    legacy_vjps = geometry._compute_derivatives(derivative_info)

    assert set(mesh_vjps) == set(legacy_vjps)
    for key in mesh_vjps:
        mesh_val = mesh_vjps[key]
        legacy_val = legacy_vjps[key]
        npt.assert_allclose(mesh_val, legacy_val, rtol=1e-4, atol=1e-6)


def test_box_mesh_derivatives_match_legacy_gradients():
    box = td.Box(center=(0.2, -0.1, 0.05), size=(1.2, 0.9, 0.8))

    derivative_info = DummyDerivativeInfo(
        linear_grad,
        paths=[("center",), ("size",)],
    )

    _assert_mesh_legacy_match(box, derivative_info)


def test_cylinder_mesh_derivatives_match_legacy_gradients():
    cylinder = td.Cylinder(
        center=(-0.3, 0.15, 0.0),
        radius=0.45,
        length=1.1,
        axis=2,
        sidewall_angle=0.08,
    )

    derivative_info = DummyDerivativeInfo(
        linear_grad,
        paths=[("center", 0), ("center", 1), ("radius",), ("length",), ("sidewall_angle",)],
    )

    _assert_mesh_legacy_match(cylinder, derivative_info)
