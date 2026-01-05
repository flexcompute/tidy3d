from __future__ import annotations

import pytest

from tidy3d.plugins.expressions.base import Expression
from tidy3d.plugins.expressions.variables import Constant


def test_expression_parse_obj_round_trip():
    expr = Constant(3.14)
    parsed = Expression.model_validate(expr.model_dump())
    assert isinstance(parsed, Constant)
    assert parsed.value == pytest.approx(3.14)


def test_expression_parse_obj_rejects_unrelated_types():
    # Simulation registers a distinct type in the global map; parsing via Expression should fail.
    with pytest.raises(ValueError, match="Cannot parse type"):
        Expression.model_validate({"type": "Simulation"})
