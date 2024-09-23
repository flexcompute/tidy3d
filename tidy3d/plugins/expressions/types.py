from typing import TYPE_CHECKING, Annotated, Union

from pydantic.v1 import Field

from tidy3d.components.types import TYPE_TAG_STR, ArrayLike, Complex

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
