from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union

from pydantic.v1 import Field

from tidy3d.components.types.base import TYPE_TAG_STR, ArrayLike, Complex

if TYPE_CHECKING:
    pass

NumberType = Union[int, float, Complex, ArrayLike]

OperatorType = Annotated[
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
    ],
    Field(discriminator=TYPE_TAG_STR),
]

FunctionType = Annotated[
    Union[
        "Sin",
        "Cos",
        "Tan",
        "Exp",
        "Log",
        "Log10",
        "Sqrt",
    ],
    Field(discriminator=TYPE_TAG_STR),
]

MetricType = Annotated[
    Union[
        "Constant",
        "Variable",
        "ModeAmp",
        "ModePower",
    ],
    Field(discriminator=TYPE_TAG_STR),
]

ExpressionType = Union[
    OperatorType,
    FunctionType,
    MetricType,
]

NumberOrExpression = Union[NumberType, ExpressionType]
