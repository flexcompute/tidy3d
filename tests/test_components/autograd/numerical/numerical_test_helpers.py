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
