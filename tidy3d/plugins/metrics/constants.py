from typing import Any

from .base import Expression
from .types import NumberType


class Constant(Expression):
    value: NumberType

    def __init__(self, value: NumberType, **kwargs: dict[str, Any]) -> None:
        super().__init__(value=value, **kwargs)

    def evaluate(self, _: Any = None) -> NumberType:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value}"
