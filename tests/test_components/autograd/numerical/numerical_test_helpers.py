from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, TypeVar

import numpy as np
import pytest
from autograd.numpy.numpy_boxes import ArrayBox
from pydantic import BaseModel

from tidy3d.components.base import make_json_compatible
from tidy3d.components.data.sim_data import SimulationData

from .result_models import Metric, NumericalResult

EVALUATION_DATA_FILENAME = "evaluation_data.npz"
RESULT_FILENAME = "result.json"
EvalFnResult: TypeAlias = float | ArrayBox
EvalFn: TypeAlias = Callable[[SimulationData], EvalFnResult]
EvaluationDataValue: TypeAlias = np.ndarray | np.generic | float | int | bool | complex
EvaluationData: TypeAlias = dict[str, EvaluationDataValue]
GradientComparisonDiagnostics: TypeAlias = dict[str, float]
MetricGroups: TypeAlias = tuple[list[Metric], list[Metric], GradientComparisonDiagnostics]
CaseIdentityT = TypeVar("CaseIdentityT", bound=BaseModel)
CASE_IDENTITY_JSON_KEY = "__case_identity_json__"


def _canonicalize_case_identity(case_identity: BaseModel) -> str:
    """Serialize a pydantic case-identity payload into canonical JSON."""
    payload = case_identity.model_dump(mode="json")
    case_identity_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    case_identity_json = make_json_compatible(case_identity_json)
    json.loads(case_identity_json, parse_constant=_raise_non_strict_json_constant)
    return case_identity_json


def _raise_non_strict_json_constant(constant: str) -> None:
    """Reject non-standard JSON constants not handled by ``make_json_compatible``."""
    raise ValueError(f"Case identity contains non-standard JSON constant {constant!r}.")


def case_identity_id(case_identity: BaseModel, prefix: str = "case", digest_len: int = 12) -> str:
    """Build a stable short id string from canonicalized case identity."""
    case_identity_json = _canonicalize_case_identity(case_identity)
    digest = hashlib.sha256(case_identity_json.encode("utf-8")).hexdigest()[:digest_len]
    return f"{prefix}-{digest}" if prefix else digest


def case_identity_from_parameters(
    case_identity_type: type[CaseIdentityT],
    parameters: BaseModel,
) -> CaseIdentityT:
    """Project a full parameter model down to its semantic case identity fields."""
    identity_fields = set(case_identity_type.model_fields)
    return case_identity_type.model_validate(parameters.model_dump(include=identity_fields))


def scale_length_by_wavelength(value: float, wavelength: float) -> float:
    """Scale a wavelength-normalized length while preserving infinite monitor dimensions."""
    if np.isinf(value):
        return np.inf
    return value * wavelength


def coords_for_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Build x/y/z coordinate arrays spanning bounds for a spatial dataset."""
    return {
        "x": np.linspace(bounds[0][0], bounds[1][0], shape[0]),
        "y": np.linspace(bounds[0][1], bounds[1][1], shape[1]),
        "z": np.linspace(bounds[0][2], bounds[1][2], shape[2]),
    }


def write_evaluation_data(
    case_dir: Path,
    evaluation_data: EvaluationData,
    case_identity: BaseModel,
) -> Path:
    """Write the canonical evaluation dataset for one numerical test case."""
    evaluation_data_path = case_dir / EVALUATION_DATA_FILENAME
    serialized_data = {name: np.asarray(value) for name, value in evaluation_data.items()}
    case_identity_json = _canonicalize_case_identity(case_identity)
    serialized_data[CASE_IDENTITY_JSON_KEY] = np.asarray(case_identity_json)
    np.savez(evaluation_data_path, **serialized_data)
    return evaluation_data_path


def load_evaluation_data(
    case_dir: Path,
    expected_case_identity: BaseModel,
    npz_file: str = EVALUATION_DATA_FILENAME,
) -> EvaluationData:
    """Load the canonical evaluation dataset for one numerical test case."""
    evaluation_data_path = case_dir / npz_file
    with np.load(evaluation_data_path, allow_pickle=False) as evaluation_data:
        if CASE_IDENTITY_JSON_KEY not in evaluation_data:
            raise ValueError(
                "Saved evaluation data is missing case identity information required for "
                "eval-only replay validation."
            )

        expected_case_identity_json = _canonicalize_case_identity(expected_case_identity)
        saved_case_identity_json = str(np.asarray(evaluation_data[CASE_IDENTITY_JSON_KEY]).item())

        if saved_case_identity_json != expected_case_identity_json:
            raise ValueError(
                "Saved evaluation data does not match the current case identity.\n"
                f"Expected: {expected_case_identity_json}\n"
                f"Found: {saved_case_identity_json}"
            )

        return {
            name: evaluation_data[name]
            for name in evaluation_data.files
            if name != CASE_IDENTITY_JSON_KEY
        }


def load_or_collect_evaluation_data(
    *,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    case_identity: BaseModel,
    collect_evaluation_data: Callable[[], EvaluationData],
) -> EvaluationData:
    """Load saved evaluation data in eval-only mode, otherwise collect and save it."""
    if numerical_eval_only:
        try:
            return load_evaluation_data(numerical_case_dir, case_identity)
        except FileNotFoundError as exc:
            pytest.fail(
                "Eval-only mode requires a saved evaluation dataset. "
                f"Run this case once without `--numerical-eval-only` first. Missing: {exc.filename}"
            )

    evaluation_data = collect_evaluation_data()
    write_evaluation_data(numerical_case_dir, evaluation_data, case_identity)
    return evaluation_data


def evaluate_fd_adjoint_gradient_agreement(
    fd_grad: np.ndarray,
    adj_grad_projected: np.ndarray,
    *,
    relative_rms_threshold: float,
    metric_name: str = "rms_error",
) -> MetricGroups:
    """Evaluate finite-difference and projected adjoint gradient agreement metrics."""
    fd_grad = np.asarray(fd_grad)
    adj_grad_projected = np.asarray(adj_grad_projected)
    rms_error = float(np.linalg.norm(fd_grad - adj_grad_projected))
    fd_mag = float(np.linalg.norm(fd_grad))
    adj_mag = float(np.linalg.norm(adj_grad_projected))
    percentage_error = float(
        100.0
        * np.mean(
            np.abs(fd_grad - adj_grad_projected) / (np.abs(fd_grad) + np.finfo(np.float64).eps)
        )
    )
    expected = relative_rms_threshold * fd_mag

    regression_metrics = [
        Metric(
            name=metric_name,
            observed=rms_error,
            expected=float(expected),
            comparator="lt",
        )
    ]
    observation_metrics: list[Metric] = []
    diagnostics = {
        "rms_error": rms_error,
        "fd_mag": fd_mag,
        "adj_mag": adj_mag,
        "percentage_error": percentage_error,
    }
    return regression_metrics, observation_metrics, diagnostics


def gradient_angle_deg(reference_grad: np.ndarray, adjoint_grad: np.ndarray) -> float:
    """Return the angle in degrees between two gradient vectors."""
    reference_grad = np.asarray(reference_grad, dtype=float)
    adjoint_grad = np.asarray(adjoint_grad, dtype=float)
    reference_norm = np.linalg.norm(reference_grad)
    adjoint_norm = np.linalg.norm(adjoint_grad)
    if np.isclose(reference_norm, 0.0) or np.isclose(adjoint_norm, 0.0):
        if np.isclose(reference_norm, 0.0) and np.isclose(adjoint_norm, 0.0):
            return 0.0
        return np.inf
    dot = np.sum((reference_grad / reference_norm) * (adjoint_grad / adjoint_norm))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(dot) * 180.0 / np.pi)


def evaluate_gradient_angle_agreement(
    reference_grad: np.ndarray,
    adjoint_grad: np.ndarray,
    *,
    angle_threshold_deg: float,
    metric_name: str = "gradient_overlap_deg",
) -> MetricGroups:
    """Evaluate gradient agreement using vector-angle overlap."""
    reference_grad = np.asarray(reference_grad, dtype=float)
    adjoint_grad = np.asarray(adjoint_grad, dtype=float)
    angle_deg = gradient_angle_deg(reference_grad, adjoint_grad)
    rms_error = float(np.linalg.norm(reference_grad - adjoint_grad))
    reference_mag = float(np.linalg.norm(reference_grad))
    adjoint_mag = float(np.linalg.norm(adjoint_grad))
    regression_metrics = [
        Metric(
            name=metric_name,
            observed=angle_deg,
            expected=float(angle_threshold_deg),
            comparator="lt",
        )
    ]
    diagnostics = {
        "gradient_overlap_deg": angle_deg,
        "rms_error": rms_error,
        "reference_mag": reference_mag,
        "adjoint_mag": adjoint_mag,
    }
    return regression_metrics, [], diagnostics


def condition_metric(name: str, condition: bool) -> Metric:
    """Represent a boolean assertion as a regression metric."""
    return Metric(
        name=name,
        observed=float(bool(condition)),
        expected=1.0,
        comparator="eq",
    )


def _safe_relative_error(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    zero_reference_error: float,
) -> np.ndarray:
    """Compute relative error without producing NaN/inf for zero reference entries."""
    actual = np.asarray(actual, dtype=float)
    desired = np.asarray(desired, dtype=float)
    abs_error = np.abs(actual - desired)
    reference_abs = np.abs(desired)
    relative_error = np.divide(
        abs_error,
        reference_abs,
        out=np.full_like(abs_error, zero_reference_error, dtype=float),
        where=reference_abs > 0,
    )
    return np.where((reference_abs <= 0) & (abs_error == 0), 0.0, relative_error)


def fd_adjoint_alignment_diagnostics(
    fd_data: np.ndarray,
    adj_data: np.ndarray,
    *,
    failed_value: float = np.finfo(np.float64).max,
) -> GradientComparisonDiagnostics:
    """Compute historical FD/adjoint vector-alignment diagnostics."""
    fd_data = np.asarray(fd_data, dtype=float)
    adj_data = np.asarray(adj_data, dtype=float)
    norm_fd = np.linalg.norm(fd_data)
    norm_adj = np.linalg.norm(adj_data)
    zero_norm_threshold = np.finfo(np.float64).eps
    if (
        fd_data.size == 0
        or adj_data.size == 0
        or not np.all(np.isfinite(fd_data))
        or not np.all(np.isfinite(adj_data))
    ):
        return {
            "overlap_deg": float(failed_value),
            "error_mean": float(failed_value),
            "error_std": float(failed_value),
            "error_norm_mean": float(failed_value),
            "error_norm_std": float(failed_value),
        }
    if norm_fd <= zero_norm_threshold and norm_adj <= zero_norm_threshold:
        return {
            "overlap_deg": 0.0,
            "error_mean": 0.0,
            "error_std": 0.0,
            "error_norm_mean": 0.0,
            "error_norm_std": 0.0,
        }
    if norm_fd <= zero_norm_threshold or norm_adj <= zero_norm_threshold:
        return {
            "overlap_deg": float(failed_value),
            "error_mean": float(failed_value),
            "error_std": float(failed_value),
            "error_norm_mean": float(failed_value),
            "error_norm_std": float(failed_value),
        }

    normalized_fd = fd_data / norm_fd
    normalized_adj = adj_data / norm_adj
    dot_fd_adj = np.sum(normalized_fd * normalized_adj)
    dot_fd_adj = np.clip(dot_fd_adj, -1.0, 1.0)
    directional_overlap_deg = np.arccos(dot_fd_adj) * 180.0 / np.pi

    relative_error = _safe_relative_error(
        fd_data,
        adj_data,
        zero_reference_error=failed_value,
    )
    relative_norm_error = _safe_relative_error(
        normalized_fd,
        normalized_adj,
        zero_reference_error=failed_value,
    )
    return {
        "overlap_deg": float(directional_overlap_deg),
        "error_mean": float(np.mean(relative_error)),
        "error_std": float(np.std(relative_error)),
        "error_norm_mean": float(np.mean(relative_norm_error)),
        "error_norm_std": float(np.std(relative_norm_error)),
    }


def evaluate_fd_adjoint_alignment_group(
    *,
    prefix: str,
    fd_data: np.ndarray,
    adj_data: np.ndarray,
    min_count: int,
    overlap_threshold_deg: float,
    failed_value: float = np.finfo(np.float64).max,
) -> MetricGroups:
    """Evaluate one FD/adjoint vector-alignment group with historical diagnostics."""
    fd_data = np.asarray(fd_data, dtype=float)
    adj_data = np.asarray(adj_data, dtype=float)
    diagnostics = fd_adjoint_alignment_diagnostics(
        fd_data,
        adj_data,
        failed_value=failed_value,
    )
    regression_metrics = [
        Metric(
            name=f"{prefix}_comparison_count",
            observed=float(fd_data.size),
            expected=float(min_count),
            comparator="gte",
        ),
        condition_metric(f"{prefix}_fd_values_finite", bool(np.all(np.isfinite(fd_data)))),
        condition_metric(f"{prefix}_adjoint_values_finite", bool(np.all(np.isfinite(adj_data)))),
        Metric(
            name=f"{prefix}_overlap_deg",
            observed=diagnostics["overlap_deg"],
            expected=float(overlap_threshold_deg),
            comparator="lt",
        ),
    ]
    return regression_metrics, [], diagnostics


def centered_fd_gradients_from_objective_values(
    objective_values_by_step: np.ndarray,
    fd_steps: np.ndarray,
) -> np.ndarray:
    """Compute centered FD gradients from plus/minus objective values for each step."""
    objective_values_by_step = np.asarray(objective_values_by_step, dtype=float)
    fd_steps = np.asarray(fd_steps, dtype=float)
    if objective_values_by_step.ndim == 1:
        objective_values_by_step = objective_values_by_step.reshape(1, -1)
    if objective_values_by_step.shape[0] != fd_steps.size:
        raise ValueError(
            "The number of finite-difference objective rows must match the number of steps."
        )
    if objective_values_by_step.shape[1] % 2:
        raise ValueError("Centered finite-difference objective rows must contain plus/minus pairs.")

    objective_plus = objective_values_by_step[:, 0::2]
    objective_minus = objective_values_by_step[:, 1::2]
    return (objective_plus - objective_minus) / (2 * fd_steps[:, np.newaxis])


def select_fd_gradient_from_step_sweep(
    fd_gradients_by_step: np.ndarray,
    fd_steps: np.ndarray,
    *,
    run_with_fd_convergence: bool,
    fd_convergence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the finest FD gradient and recompute the convergence-valid mask."""
    fd_gradients_by_step = np.asarray(fd_gradients_by_step, dtype=float)
    fd_steps = np.asarray(fd_steps, dtype=float)
    if fd_gradients_by_step.ndim == 1:
        fd_gradients_by_step = fd_gradients_by_step.reshape(1, -1)

    if run_with_fd_convergence:
        if fd_steps.size != 2:
            raise ValueError("FD convergence selection currently expects exactly two step sizes.")

        argmin_convergence_test = np.argmin(fd_steps)
        argmax_convergence_test = np.argmax(fd_steps)
        fd_grad_fine = fd_gradients_by_step[argmin_convergence_test]
        fd_grad_coarse = fd_gradients_by_step[argmax_convergence_test]
        abs_diff = np.abs(fd_grad_coarse - fd_grad_fine)
        scale = np.maximum.reduce(
            [
                np.abs(fd_grad_fine),
                np.abs(fd_grad_coarse),
                np.full_like(fd_grad_fine, np.finfo(np.float64).eps, dtype=float),
            ]
        )
        relative_diff = abs_diff / scale
        valid_mask = np.abs(relative_diff) < fd_convergence_threshold
        fd_grad = np.squeeze(fd_grad_fine)
    else:
        fd_grad = np.squeeze(fd_gradients_by_step)
        valid_mask = np.ones(np.shape(fd_grad), dtype=bool)

    return np.asarray(fd_grad, dtype=float), np.asarray(valid_mask, dtype=bool)


def select_fd_gradient_from_objective_values(
    objective_values_by_step: np.ndarray,
    fd_steps: np.ndarray,
    *,
    run_with_fd_convergence: bool,
    fd_convergence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute and select centered FD gradients from objective values by step."""
    fd_gradients_by_step = centered_fd_gradients_from_objective_values(
        objective_values_by_step,
        fd_steps,
    )
    return select_fd_gradient_from_step_sweep(
        fd_gradients_by_step,
        fd_steps,
        run_with_fd_convergence=run_with_fd_convergence,
        fd_convergence_threshold=fd_convergence_threshold,
    )


def evaluate_allclose_agreement(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    rtol: float,
    atol: float,
    metric_name: str = "max_allclose_scaled_error",
) -> MetricGroups:
    """Evaluate NumPy ``assert_allclose``-style agreement as one scalar metric."""
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    abs_error = np.abs(actual - desired)
    tolerance = atol + rtol * np.abs(desired)
    scaled_error = np.divide(
        abs_error,
        tolerance,
        out=np.full_like(abs_error, np.finfo(np.float64).max, dtype=float),
        where=tolerance > 0,
    )
    scaled_error = np.where((tolerance <= 0) & (abs_error == 0), 0.0, scaled_error)

    max_scaled_error = float(np.max(scaled_error)) if scaled_error.size else 0.0
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    max_allowed_abs_error = float(np.max(tolerance)) if tolerance.size else 0.0
    rms_error = float(np.sqrt(np.mean(abs_error**2))) if abs_error.size else 0.0
    reference_norm = float(np.linalg.norm(desired))
    rms_error_normalized = rms_error / max(reference_norm, np.finfo(np.float64).eps)

    regression_metrics = [
        Metric(
            name=metric_name,
            observed=max_scaled_error,
            expected=1.0,
            comparator="lte",
        )
    ]
    observation_metrics: list[Metric] = []
    diagnostics = {
        "max_allclose_scaled_error": max_scaled_error,
        "max_abs_error": max_abs_error,
        "max_allowed_abs_error": max_allowed_abs_error,
        "rms_error": rms_error,
        "reference_norm": reference_norm,
        "rms_error_normalized": float(rms_error_normalized),
    }
    return regression_metrics, observation_metrics, diagnostics


def finalize_result(
    *,
    pytest_nodeid: str,
    numerical_case_dir: Path,
    regression_metrics: list[Metric],
    observation_metrics: list[Metric],
    failure_message: str,
) -> NumericalResult:
    """Write the canonical result and hand the final outcome back to pytest."""
    result_record = NumericalResult.from_metrics(
        pytest_nodeid=pytest_nodeid,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
    )
    result_record.to_json_file(numerical_case_dir / RESULT_FILENAME)

    if not result_record.passes():
        pytest.fail(failure_message)

    return result_record
