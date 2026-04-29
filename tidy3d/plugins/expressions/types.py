from __future__ import annotations

from typing import TYPE_CHECKING, Union

from tidy3d.components.types import Complex
from tidy3d.components.types.base import ArrayLikeStrict, discriminated_union

if TYPE_CHECKING:
    from .functions import Cos, Exp, Log, Log10, Sin, Sqrt, Tan
    from .metrics import ModeAmp, ModePower
    from .operators import (
        Abs,
        Add,
        Divide,
        FloorDivide,
        MatMul,
        Modulus,
        Multiply,
        Negate,
        Power,
        Subtract,
    )
    from .variables import Constant, Variable

NumberType = int | float | Complex | ArrayLikeStrict

OperatorType = discriminated_union(
    Union[
        "Add",
        "Subtract",
        "Multiply",
        "Divide",
        "Power",
        "Modulus",
        "FloorDivide",
        "MatMul",
        "Negate",
        "Abs",
    ]
)

FunctionType = discriminated_union(
    Union[
        "Sin",
        "Cos",
        "Tan",
        "Exp",
        "Log",
        "Log10",
        "Sqrt",
    ]
)

MetricType = discriminated_union(
    Union[
        "Constant",
        "Variable",
        "ModeAmp",
        "ModePower",
    ]
)

ExpressionType = OperatorType | FunctionType | MetricType

NumberOrExpression = ExpressionType | NumberType
