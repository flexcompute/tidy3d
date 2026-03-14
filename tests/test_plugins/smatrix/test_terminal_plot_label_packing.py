"""Tests for automated label packing used in terminal port plotting."""

from __future__ import annotations

import numpy as np

from tidy3d.plugins.smatrix.component_modelers.terminal import _pack_label_centers_1d


def _assert_non_overlapping_centers(
    centers_px: np.ndarray, widths_px: np.ndarray, pad_px: float
) -> None:
    order = np.argsort(centers_px)
    for left_idx, right_idx in zip(order[:-1], order[1:]):
        left_right_edge = centers_px[left_idx] + widths_px[left_idx] / 2
        right_left_edge = centers_px[right_idx] - widths_px[right_idx] / 2
        assert right_left_edge >= left_right_edge + pad_px - 1e-9


def test_pack_label_centers_1d_non_overlapping_and_in_bounds():
    """Packed labels stay non-overlapping and within the given bounds."""
    anchors = np.array([10.0, 11.0, 12.0])
    widths = np.array([10.0, 10.0, 10.0])
    x_min, x_max = 0.0, 40.0
    pad = 2.0

    centers = _pack_label_centers_1d(
        anchor_centers_px=anchors, widths_px=widths, x_min_px=x_min, x_max_px=x_max, pad_px=pad
    )

    _assert_non_overlapping_centers(centers_px=centers, widths_px=widths, pad_px=pad)
    assert np.all(centers - widths / 2 >= x_min - 1e-9)
    assert np.all(centers + widths / 2 <= x_max + 1e-9)


def test_pack_label_centers_1d_handles_right_overflow():
    """Labels anchored near the right edge are shifted left to stay in bounds."""
    anchors = np.array([30.0, 31.0, 32.0])
    widths = np.array([12.0, 12.0, 12.0])
    x_min, x_max = 0.0, 40.0
    pad = 2.0

    centers = _pack_label_centers_1d(
        anchor_centers_px=anchors, widths_px=widths, x_min_px=x_min, x_max_px=x_max, pad_px=pad
    )

    _assert_non_overlapping_centers(centers_px=centers, widths_px=widths, pad_px=pad)
    assert np.all(centers + widths / 2 <= x_max + 1e-9)
