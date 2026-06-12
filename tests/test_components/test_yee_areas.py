"""Unit tests for ``tidy3d.components.grid.yee_areas``.

The helper underpins :meth:`ElectromagneticFieldData._diff_area_at_yee_positions`
(``ModeSolverData.dot``, ``FieldData.flux`` with ``use_colocated_integration=False``,
and ``ModeMonitor`` flux integration), so these tests pin the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidy3d.components.grid.yee_areas import (
    colocated_edges_1d,
    colocated_widths_1d,
    yee_primal_dual_widths_1d,
)


def test_yee_primal_dual_widths_1d_degenerate_single_cell():
    """Single-cell axis returns unit widths per the legacy 2D convention."""
    bnds = np.array([0.0, 1.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, mnt_min=0.0, mnt_max=1.0)
    assert cell.tolist() == [1.0]
    assert dual.tolist() == [1.0]


def test_yee_primal_dual_widths_1d_uniform_full_coverage():
    """Uniform 4-cell axis, monitor covers full extent."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, mnt_min=0.0, mnt_max=4.0)
    assert cell.shape == (4,)
    assert dual.shape == (4,)
    # cell sizes: clip([0,1,2,3, 4], 0, 4) → [0,1,2,3,4], diff → [1,1,1,1]
    np.testing.assert_allclose(cell, [1.0, 1.0, 1.0, 1.0])
    # dual sizes: centers=[0, 0.5, 1.5, 2.5, 3.5], clip → unchanged, diff → [0.5, 1, 1, 1]
    np.testing.assert_allclose(dual, [0.5, 1.0, 1.0, 1.0])


def test_yee_primal_dual_widths_1d_monitor_edge_clipping():
    """Monitor clipped inside the first/last cells reduces edge cell widths."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, mnt_min=0.3, mnt_max=2.7)
    # cell sizes: clip([0,1,2,3], 0.3, 2.7) → [0.3,1,2,2.7], diff → [0.7, 1, 0.7]
    np.testing.assert_allclose(cell, [0.7, 1.0, 0.7])
    # dual sizes: clip([0.3, 0.5, 1.5, 2.5], 0.3, 2.7) → [0.3, 0.5, 1.5, 2.5], diff → [0.2, 1, 1]
    np.testing.assert_allclose(dual, [0.2, 1.0, 1.0])


def test_yee_primal_dual_widths_1d_infinite_bounds_integrate_grid_extent():
    """Infinite monitor bounds integrate exactly the (default) grid extent."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0])
    cell_inf, dual_inf = yee_primal_dual_widths_1d(bnds, mnt_min=-np.inf, mnt_max=np.inf)
    cell_full, dual_full = yee_primal_dual_widths_1d(bnds, mnt_min=0.0, mnt_max=3.0)
    np.testing.assert_allclose(cell_inf, cell_full)
    np.testing.assert_allclose(dual_inf, dual_full)


def test_yee_primal_dual_widths_1d_widths_match_full_grid_spans():
    """Sum of primal widths == monitor extent (when clipped within grid)."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cell, _ = yee_primal_dual_widths_1d(bnds, mnt_min=0.3, mnt_max=3.7)
    assert cell.sum() == pytest.approx(3.4)


def test_yee_primal_dual_widths_1d_nonuniform_grid():
    """Non-uniform grid produces non-uniform primal and dual widths."""
    bnds = np.array([0.0, 0.5, 1.5, 3.5, 4.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, mnt_min=0.0, mnt_max=4.0)
    # primal cell widths
    np.testing.assert_allclose(cell, [0.5, 1.0, 2.0, 0.5])
    # dual: centers = [0.25, 1.0, 2.5, 3.75], with edges clipped
    # clip([0, 0.25, 1.0, 2.5, 3.75], 0, 4) = [0, 0.25, 1.0, 2.5, 3.75]
    # diff = [0.25, 0.75, 1.5, 1.25]
    np.testing.assert_allclose(dual, [0.25, 0.75, 1.5, 1.25])


def test_colocated_widths_1d_uniform_interior_monitor():
    """Per-sample widths: half cells at the monitor edges, full cells inside."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0])
    widths = colocated_widths_1d(bnds, mnt_min=0.0, mnt_max=3.0)
    np.testing.assert_allclose(widths, [0.5, 1.0, 1.0, 0.5])


def test_colocated_widths_1d_clipped_to_monitor():
    """Samples outside the monitor bounds collapse to zero width; straddling cells truncate."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0])
    widths = colocated_widths_1d(bnds, mnt_min=0.75, mnt_max=2.25)
    np.testing.assert_allclose(widths, [0.0, 0.75, 0.75, 0.0])


def test_colocated_widths_1d_inf_bounds_cannot_overcount():
    """Infinite monitor bounds integrate exactly the sample extent -- never beyond it."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0])
    widths = colocated_widths_1d(bnds)
    assert widths.sum() == pytest.approx(bnds[-1] - bnds[0])


def test_colocated_widths_1d_single_sample_unit_convention():
    """A single-sample (zero-sized) axis collapses to width 1.0 (W/um units)."""
    np.testing.assert_allclose(colocated_widths_1d(np.array([0.7])), [1.0])


def test_colocated_edges_1d_match_widths():
    """colocated_widths_1d is exactly the diff of colocated_edges_1d."""
    bnds = np.array([0.0, 0.5, 1.5, 3.5, 4.0])
    edges = colocated_edges_1d(bnds, mnt_min=0.2, mnt_max=3.8)
    np.testing.assert_allclose(np.diff(edges), colocated_widths_1d(bnds, 0.2, 3.8))


def test_yee_primal_dual_widths_1d_valid_bounds_clamps_inf():
    """With valid_bounds, an infinite monitor bound resolves to the valid extent, not the
    padded grid edge -- the integration covers exactly the valid (halo-free) region."""
    # grid padded by one cell on each side; valid extent is [1.0, 3.0]
    bnds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, valid_bounds=(1.0, 3.0))
    assert cell.sum() == pytest.approx(2.0)
    assert dual.sum() == pytest.approx(2.0)
    # without valid_bounds the padding cells are integrated too
    cell_pad, _ = yee_primal_dual_widths_1d(bnds)
    assert cell_pad.sum() == pytest.approx(4.0)


def test_yee_primal_dual_widths_1d_valid_bounds_clamps_oversized():
    """Finite monitor bounds wider than the valid extent are clamped to it."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cell, dual = yee_primal_dual_widths_1d(bnds, -100.0, 100.0, valid_bounds=(1.0, 3.0))
    assert cell.sum() == pytest.approx(2.0)
    assert dual.sum() == pytest.approx(2.0)


def test_yee_primal_dual_widths_1d_valid_bounds_interior_noop():
    """Monitor bounds inside the valid extent are unaffected by the clamp."""
    bnds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with_vb = yee_primal_dual_widths_1d(bnds, 1.3, 2.7, valid_bounds=(1.0, 3.0))
    without_vb = yee_primal_dual_widths_1d(bnds, 1.3, 2.7)
    np.testing.assert_array_equal(with_vb[0], without_vb[0])
    np.testing.assert_array_equal(with_vb[1], without_vb[1])
