from typing import TYPE_CHECKING, Annotated, Union

from pydantic.v1 import Field

from tidy3d.components.types import TYPE_TAG_STR, ArrayLike, Complex

if TYPE_CHECKING:
    from .constants import Constant
    from .functions import Cos, Exp, Log, Log10, Sin, Tan
    from .metrics import ModeCoefficient, ModePower
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
    ],
    Field(discriminator=TYPE_TAG_STR),
]

MetricType = Annotated[
    Union[
        "Constant",
        "ModeCoefficient",
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
