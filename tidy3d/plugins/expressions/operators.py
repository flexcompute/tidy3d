from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from .base import Expression
from .types import NumberOrExpression

if TYPE_CHECKING:
    from .types import ExpressionType, NumberType


class UnaryOperator(Expression):
    """Base class for unary operators in the metrics module.

    Notes
    -----
        This class represents an operation with a single operand.
        Subclasses should implement the evaluate method to define the specific operation.
    """

    operand: NumberOrExpression = Field(
        title="Operand",
        description="The operand for the unary operator.",
    )

    _symbol: str
    _format: str = "({symbol}{operand})"

    @field_validator("operand")
    @classmethod
    def validate_operand(cls, v: NumberOrExpression) -> ExpressionType:
        return cls._to_expression(v)

    def __repr__(self) -> str:
        return self._format.format(symbol=self._symbol, operand=self.operand)


class BinaryOperator(Expression):
    """Base class for binary operators in the metrics module.

    Notes
    -----
        This class represents an operation with two operands.
        Subclasses should implement the evaluate method to define the specific operation.
    """

    left: NumberOrExpression = Field(
        title="Left",
        description="The left operand for the binary operator.",
    )
    right: NumberOrExpression = Field(
        title="Right",
        description="The right operand for the binary operator.",
    )

    _symbol: str
    _format: str = "({left} {symbol} {right})"

    @field_validator("left", "right")
    @classmethod
    def validate_operands(cls, v: NumberOrExpression) -> ExpressionType:
        return cls._to_expression(v)

    def __repr__(self) -> str:
        return self._format.format(left=self.left, symbol=self._symbol, right=self.right)


class Add(BinaryOperator):
    """
    Represents the addition operation.
    """

    _symbol: str = "+"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) + self.right(*args, **kwargs)


class Subtract(BinaryOperator):
    """
    Represents the subtraction operation.
    """

    _symbol: str = "-"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) - self.right(*args, **kwargs)


class Multiply(BinaryOperator):
    """
    Represents the multiplication operation.
    """

    _symbol: str = "*"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) * self.right(*args, **kwargs)


class Negate(UnaryOperator):
    """
    Represents the negation operation.
    """

    _symbol: str = "-"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return -self.operand(*args, **kwargs)


class Abs(UnaryOperator):
    """
    Represents the absolute value operation.
    """

    _symbol: str = "abs"
    _format = "{symbol}({operand})"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return abs(self.operand(*args, **kwargs))


class Divide(BinaryOperator):
    """
    Represents the division operation.
    """

    _symbol: str = "/"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) / self.right(*args, **kwargs)


class Power(BinaryOperator):
    """
    Represents the exponentiation operation.
    """

    _symbol: str = "**"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) ** self.right(*args, **kwargs)


class Modulus(BinaryOperator):
    """
    Represents the modulus operation.
    """

    _symbol: str = "%"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) % self.right(*args, **kwargs)


class FloorDivide(BinaryOperator):
    """
    Represents the floor division operation.
    """

    _symbol: str = "//"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) // self.right(*args, **kwargs)


class MatMul(BinaryOperator):
    """
    Represents the matrix multiplication operation.
    """

    _symbol: str = "@"

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.left(*args, **kwargs) @ self.right(*args, **kwargs)
