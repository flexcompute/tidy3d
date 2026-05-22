from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest
from pydantic import BaseModel

from .result_models import Metric, NumericalResult

EVALUATION_DATA_FILENAME = "evaluation_data.npz"
RESULT_FILENAME = "result.json"
EvaluationDataValue: TypeAlias = np.ndarray | np.generic | float | int | bool | complex
EvaluationData: TypeAlias = dict[str, EvaluationDataValue]
CASE_IDENTITY_JSON_KEY = "__case_identity_json__"


def _canonicalize_case_identity(case_identity: BaseModel) -> str:
    """Serialize a pydantic case-identity payload into canonical JSON."""
    payload = case_identity.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def case_identity_id(case_identity: BaseModel, prefix: str = "case", digest_len: int = 12) -> str:
    """Build a stable short id string from canonicalized case identity."""
    case_identity_json = _canonicalize_case_identity(case_identity)
    digest = hashlib.sha256(case_identity_json.encode("utf-8")).hexdigest()[:digest_len]
    return f"{prefix}-{digest}" if prefix else digest


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
