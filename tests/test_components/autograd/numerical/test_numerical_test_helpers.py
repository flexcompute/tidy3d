from __future__ import annotations

import numpy as np
import pytest

from . import numerical_test_helpers
from .numerical_test_helpers import (
    evaluate_fd_adjoint_alignment_group,
    fd_adjoint_alignment_diagnostics,
    select_fd_gradient_from_step_sweep,
)


def test_fd_adjoint_alignment_diagnostics_sparse_match_has_zero_error():
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([1.0, 0.0]),
        adj_data=np.array([1.0, 0.0]),
    )

    assert diagnostics["overlap_deg"] == pytest.approx(0.0)
    assert diagnostics["error_mean"] == pytest.approx(0.0)
    assert diagnostics["error_std"] == pytest.approx(0.0)
    assert diagnostics["error_norm_mean"] == pytest.approx(0.0)
    assert diagnostics["error_norm_std"] == pytest.approx(0.0)


def test_fd_adjoint_alignment_diagnostics_zero_reference_mismatch_is_finite():
    failed_value = 123.0
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([1.0, 1.0]),
        adj_data=np.array([1.0, 0.0]),
        failed_value=failed_value,
    )

    assert np.isfinite(diagnostics["error_mean"])
    assert diagnostics["error_mean"] == pytest.approx(0.5 * failed_value)
    assert np.isfinite(diagnostics["error_norm_mean"])
    assert diagnostics["error_norm_mean"] > 0.0


def test_fd_adjoint_alignment_diagnostics_clips_overlap_dot(monkeypatch):
    original_sum = numerical_test_helpers.np.sum

    def sum_with_roundoff(value):
        summed = original_sum(value)
        if np.shape(value) == (2,):
            return summed + np.finfo(float).eps
        return summed

    monkeypatch.setattr(numerical_test_helpers.np, "sum", sum_with_roundoff)

    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([1.0, 0.0]),
        adj_data=np.array([1.0, 0.0]),
    )

    assert diagnostics["overlap_deg"] == pytest.approx(0.0)


def test_fd_adjoint_alignment_diagnostics_keeps_small_nonzero_vectors():
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([1.0e-9, 0.0]),
        adj_data=np.array([1.0e-9, 0.0]),
    )

    assert diagnostics["overlap_deg"] == pytest.approx(0.0)
    assert diagnostics["error_mean"] == pytest.approx(0.0)


def test_fd_adjoint_alignment_diagnostics_zero_vectors_agree():
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([0.0, 0.0]),
        adj_data=np.array([0.0, 0.0]),
    )

    assert diagnostics["overlap_deg"] == pytest.approx(0.0)
    assert diagnostics["error_mean"] == pytest.approx(0.0)
    assert diagnostics["error_std"] == pytest.approx(0.0)
    assert diagnostics["error_norm_mean"] == pytest.approx(0.0)
    assert diagnostics["error_norm_std"] == pytest.approx(0.0)


def test_fd_adjoint_alignment_diagnostics_zero_nonzero_mismatch_fails():
    failed_value = 123.0
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data=np.array([0.0, 0.0]),
        adj_data=np.array([1.0, 0.0]),
        failed_value=failed_value,
    )

    assert diagnostics["overlap_deg"] == pytest.approx(failed_value)
    assert diagnostics["error_mean"] == pytest.approx(failed_value)


def test_evaluate_fd_adjoint_alignment_group_gates_overlap():
    regression_metrics, observation_metrics, diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="grad",
        fd_data=np.array([1.0, 1.0]),
        adj_data=np.array([1.0, -1.0]),
        min_count=2,
        overlap_threshold_deg=45.0,
    )

    overlap_metric = next(
        metric for metric in regression_metrics if metric.name == "grad_overlap_deg"
    )
    assert diagnostics["overlap_deg"] == pytest.approx(90.0)
    assert overlap_metric.observed == pytest.approx(90.0)
    assert overlap_metric.expected == pytest.approx(45.0)
    assert overlap_metric.comparator == "lt"
    assert observation_metrics == []


def test_evaluate_fd_adjoint_alignment_group_accepts_zero_vector_agreement():
    regression_metrics, observation_metrics, diagnostics = evaluate_fd_adjoint_alignment_group(
        prefix="grad",
        fd_data=np.array([0.0, 0.0]),
        adj_data=np.array([0.0, 0.0]),
        min_count=2,
        overlap_threshold_deg=45.0,
    )

    overlap_metric = next(
        metric for metric in regression_metrics if metric.name == "grad_overlap_deg"
    )
    assert diagnostics["overlap_deg"] == pytest.approx(0.0)
    assert overlap_metric.passes()
    assert observation_metrics == []


def test_select_fd_gradient_from_step_sweep_keeps_stable_zero_gradients():
    fd_grad, valid_mask = select_fd_gradient_from_step_sweep(
        fd_gradients_by_step=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.01, 2.0],
            ]
        ),
        fd_steps=np.array([1.0, 2.0]),
        run_with_fd_convergence=True,
        fd_convergence_threshold=0.05,
    )

    assert np.allclose(fd_grad, [0.0, 1.0, 2.0])
    assert np.array_equal(valid_mask, [True, True, True])


def test_select_fd_gradient_from_step_sweep_rejects_zero_nonzero_mismatch():
    fd_grad, valid_mask = select_fd_gradient_from_step_sweep(
        fd_gradients_by_step=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.01],
            ]
        ),
        fd_steps=np.array([1.0, 2.0]),
        run_with_fd_convergence=True,
        fd_convergence_threshold=0.05,
    )

    assert np.allclose(fd_grad, [0.0, 1.0])
    assert np.array_equal(valid_mask, [False, True])
