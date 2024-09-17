from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import TYPE_TAG_STR

from .types import ExpressionType, NumberOrExpression, NumberType

if TYPE_CHECKING:
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

TYPE_TO_CLASS_MAP: dict[str, Any] = {}


class Expression(Tidy3dBaseModel, ABC):
    """
    Base class for all expressions in the metrics module.

    This class serves as the foundation for all other components in the metrics module.
    It provides common functionality and operator overloading for derived classes.
    """

    class Config:
        smart_union = True

    def __init_subclass__(cls, **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        type_value = cls.__fields__.get(TYPE_TAG_STR)
        if type_value and type_value.default:
            TYPE_TO_CLASS_MAP[type_value.default] = cls

    @classmethod
    def parse_obj(cls, obj: dict[str, Any]) -> ExpressionType:
        if not isinstance(obj, dict):
            raise TypeError("Input must be a dict")
        type_value = obj.get(TYPE_TAG_STR)
        if type_value is None:
            raise ValueError('Missing "type" in data')
        subclass = TYPE_TO_CLASS_MAP.get(type_value)
        if subclass is None:
            raise ValueError(f"Unknown type: {type_value}")
        return subclass(**obj)

    @staticmethod
    def _to_expression(other: NumberOrExpression | dict[str, Any]) -> ExpressionType:
        if isinstance(other, Expression):
            return other
        elif isinstance(other, dict):
            return Expression.parse_obj(other)
        else:
            from .constants import Constant

            return Constant(other)

    def __call__(self, data: NumberType = None) -> NumberType:
        return self.evaluate(data)

    def __neg__(self) -> Negate:
        from .operators import Negate

        return Negate(operand=self)

    def __add__(self, other: NumberOrExpression) -> Add:
        from .operators import Add

        return Add(left=self, right=other)

    def __radd__(self, other: NumberOrExpression) -> Add:
        return self.__add__(other)

    def __sub__(self, other: NumberOrExpression) -> Subtract:
        from .operators import Subtract

        return Subtract(left=self, right=other)

    def __rsub__(self, other: NumberOrExpression) -> Subtract:
        from .operators import Subtract

        return Subtract(left=other, right=self)

    def __mul__(self, other: NumberOrExpression) -> Multiply:
        from .operators import Multiply

        return Multiply(left=self, right=other)

    def __rmul__(self, other: NumberOrExpression) -> Multiply:
        return self.__mul__(other)

    def __abs__(self) -> Abs:
        from .operators import Abs

        return Abs(operand=self)

    def __truediv__(self, other: NumberOrExpression) -> Divide:
        from .operators import Divide

        return Divide(left=self, right=other)

    def __rtruediv__(self, other: NumberOrExpression) -> Divide:
        from .operators import Divide

        return Divide(left=other, right=self)

    def __pow__(self, other: NumberOrExpression) -> Power:
        from .operators import Power

        return Power(left=self, right=other)

    def __rpow__(self, other: NumberOrExpression) -> Power:
        from .operators import Power

        return Power(left=other, right=self)

    def __mod__(self, other: NumberOrExpression) -> Modulus:
        from .operators import Modulus

        return Modulus(left=self, right=other)

    def __rmod__(self, other: NumberOrExpression) -> Modulus:
        from .operators import Modulus

        return Modulus(left=other, right=self)

    def __floordiv__(self, other: NumberOrExpression) -> FloorDivide:
        from .operators import FloorDivide

        return FloorDivide(left=self, right=other)

    def __rfloordiv__(self, other: NumberOrExpression) -> FloorDivide:
        from .operators import FloorDivide

        return FloorDivide(left=other, right=self)

    def __matmul__(self, other: NumberOrExpression) -> MatMul:
        from .operators import MatMul

        return MatMul(left=self, right=other)

    def __rmatmul__(self, other: NumberOrExpression) -> MatMul:
        from .operators import MatMul

        return MatMul(left=other, right=self)

    def __iadd__(self, other: NumberOrExpression) -> Add:
        return self + other

    def __isub__(self, other: NumberOrExpression) -> Subtract:
        return self - other

    def __imul__(self, other: NumberOrExpression) -> Multiply:
        return self * other

    def __itruediv__(self, other: NumberOrExpression) -> Divide:
        return self / other

    def __ifloordiv__(self, other: NumberOrExpression) -> FloorDivide:
        return self // other

    def __imod__(self, other: NumberOrExpression) -> Modulus:
        return self % other

    def __ipow__(self, other: NumberOrExpression) -> Power:
        return self**other

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
