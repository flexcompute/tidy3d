from __future__ import annotations

from typing import Any

import pydantic.v1 as pd

from .base import Expression
from .types import NumberOrExpression, NumberType


class UnaryOperator(Expression):
    """
    Base class for unary operators in the metrics module.

    This class represents an operation with a single operand.
    Subclasses should implement the evaluate method to define the specific operation.
    """

    operand: NumberOrExpression

    _symbol: str
    _format: str = "({symbol}{operand})"

    @pd.validator("operand", pre=True, always=True)
    def validate_operand(cls, v):
        return cls._to_expression(v)

    def __repr__(self) -> str:
        return self._format.format(symbol=self._symbol, operand=self.operand)


class BinaryOperator(Expression):
    """
    Base class for binary operators in the metrics module.

    This class represents an operation with two operands.
    Subclasses should implement the evaluate method to define the specific operation.
    """

    left: NumberOrExpression
    right: NumberOrExpression

    _symbol: str
    _format: str = "({left} {symbol} {right})"

    @pd.validator("left", "right", pre=True, always=True)
    def validate_operands(cls, v):
        return cls._to_expression(v)

    def __repr__(self) -> str:
        return self._format.format(left=self.left, symbol=self._symbol, right=self.right)


class Add(BinaryOperator):
    _symbol: str = "+"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) + self.right(*args, **kwargs)


class Subtract(BinaryOperator):
    _symbol: str = "-"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) - self.right(*args, **kwargs)


class Multiply(BinaryOperator):
    _symbol: str = "*"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) * self.right(*args, **kwargs)


class Negate(UnaryOperator):
    _symbol: str = "-"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return -self.operand(*args, **kwargs)


class Abs(UnaryOperator):
    _symbol: str = "abs"
    _format = "{symbol}({operand})"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return abs(self.operand(*args, **kwargs))


class Divide(BinaryOperator):
    _symbol: str = "/"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) / self.right(*args, **kwargs)


class Power(BinaryOperator):
    _symbol: str = "**"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) ** self.right(*args, **kwargs)


class Modulus(BinaryOperator):
    _symbol: str = "%"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) % self.right(*args, **kwargs)


class FloorDivide(BinaryOperator):
    _symbol: str = "//"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) // self.right(*args, **kwargs)


class MatMul(BinaryOperator):
    _symbol: str = "@"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) @ self.right(*args, **kwargs)
