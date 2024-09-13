from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import pydantic.v1 as pd

from .base import Expression
from .types import NumberOrExpression

if TYPE_CHECKING:
    from .types import ExpressionType


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

    @abc.abstractmethod
    def evaluate(self, x: ExpressionType) -> ExpressionType:
        pass

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

    @abc.abstractmethod
    def evaluate(self, x: ExpressionType, y: ExpressionType) -> ExpressionType:
        pass

    def __repr__(self) -> str:
        return self._format.format(left=self.left, symbol=self._symbol, right=self.right)


class Add(BinaryOperator):
    _symbol: str = "+"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) + self.right(x)


class Subtract(BinaryOperator):
    _symbol: str = "-"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) - self.right(x)


class Multiply(BinaryOperator):
    _symbol: str = "*"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) * self.right(x)


class Negate(UnaryOperator):
    _symbol: str = "-"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return -self.operand(x)


class Abs(UnaryOperator):
    _symbol: str = "abs"
    _format = "{symbol}({operand})"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return abs(self.operand(x))


class Divide(BinaryOperator):
    _symbol: str = "/"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) / self.right(x)


class Power(BinaryOperator):
    _symbol: str = "**"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) ** self.right(x)


class Modulus(BinaryOperator):
    _symbol: str = "%"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) % self.right(x)


class FloorDivide(BinaryOperator):
    _symbol: str = "//"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) // self.right(x)


class MatMul(BinaryOperator):
    _symbol: str = "@"

    def evaluate(self, x: ExpressionType) -> ExpressionType:
        return self.left(x) @ self.right(x)
