import abc
from typing import Any

import autograd.numpy as anp
import pydantic.v1 as pd

from .base import Expression
from .types import NumberOrExpression, NumberType


class Function(Expression):
    operand: NumberOrExpression

    _format: str = "{func}({operand})"

    @pd.validator("operand", pre=True, always=True)
    def validate_operand(cls, v):
        return cls._to_expression(v)

    def __init__(self, operand: NumberOrExpression, **kwargs: dict[str, Any]) -> None:
        super().__init__(operand=operand, **kwargs)

    @abc.abstractmethod
    def evaluate(self, x: NumberType) -> NumberType:
        pass

    def __repr__(self):
        return self._format.format(func=self.type, operand=self.operand)


class Sin(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.sin(self.operand(x))


class Cos(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.cos(self.operand(x))


class Tan(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.tan(self.operand(x))


class Exp(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.exp(self.operand(x))


class Log(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.log(self.operand(x))


class Log10(Function):
    def evaluate(self, x: NumberType) -> NumberType:
        return anp.log10(self.operand(x))
