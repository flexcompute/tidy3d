from typing import Any

import autograd.numpy as anp
import pydantic.v1 as pd

from .base import Expression
from .types import NumberOrExpression, NumberType


class Function(Expression):
    """
    Base class for mathematical functions in expressions.
    """

    operand: NumberOrExpression = pd.Field(
        ...,
        title="Operand",
        description="The operand for the function.",
    )

    _format: str = "{func}({operand})"

    @pd.validator("operand", pre=True, always=True)
    def validate_operand(cls, v):
        """
        Validate and convert operand to an expression.
        """
        return cls._to_expression(v)

    def __init__(self, operand: NumberOrExpression, **kwargs: dict[str, Any]) -> None:
        """
        Initialize the function with an operand.

        Parameters
        ----------
        operand : NumberOrExpression
            The operand for the function.
        kwargs : dict[str, Any]
            Additional keyword arguments.
        """
        super().__init__(operand=operand, **kwargs)

    def __repr__(self):
        """
        Return a string representation of the function.
        """
        return self._format.format(func=self.type, operand=self.operand)


class Sin(Function):
    """
    Sine function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the sine function.

        Returns
        -------
        NumberType
            Sine of the input value.
        """
        return anp.sin(self.operand(*args, **kwargs))


class Cos(Function):
    """
    Cosine function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the cosine function.

        Returns
        -------
        NumberType
            Cosine of the input value.
        """
        return anp.cos(self.operand(*args, **kwargs))


class Tan(Function):
    """
    Tangent function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the tangent function.

        Returns
        -------
        NumberType
            Tangent of the input value.
        """
        return anp.tan(self.operand(*args, **kwargs))


class Exp(Function):
    """
    Exponential function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the exponential function.

        Returns
        -------
        NumberType
            Exponential of the input value.
        """
        return anp.exp(self.operand(*args, **kwargs))


class Log(Function):
    """
    Natural logarithm function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the natural logarithm function.

        Returns
        -------
        NumberType
            Natural logarithm of the input value.
        """
        return anp.log(self.operand(*args, **kwargs))


class Log10(Function):
    """
    Base-10 logarithm function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the base-10 logarithm function.

        Returns
        -------
        NumberType
            Base-10 logarithm of the input value.
        """
        return anp.log10(self.operand(*args, **kwargs))


class Sqrt(Function):
    """
    Square root function expression.
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        """
        Evaluate the square root function.

        Returns
        -------
        NumberType
            Square root of the input value.
        """
        return anp.sqrt(self.operand(*args, **kwargs))
