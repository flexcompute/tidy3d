"""Snapshot of the ``config.adjoint`` values that geometry's adjoint code reads.

``GeometryAdjointSettings`` lives in :mod:`.types`, which geometry imports and which
therefore stays free of the config; reading the config happens here instead.
"""

from __future__ import annotations

from tidy3d.config import config

from .types import GeometryAdjointSettings


def adjoint_settings_from_config() -> GeometryAdjointSettings:
    """Snapshot the geometry-facing ``config.adjoint`` values for one adjoint pass."""
    adjoint_cfg = config.adjoint
    return GeometryAdjointSettings(
        gradient_dtype_float=adjoint_cfg.gradient_dtype_float,
        edge_clip_tolerance=adjoint_cfg.edge_clip_tolerance,
        quadrature_sample_fraction=adjoint_cfg.quadrature_sample_fraction,
        gauss_quadrature_order=adjoint_cfg.gauss_quadrature_order,
        points_per_wavelength=adjoint_cfg.points_per_wavelength,
    )
