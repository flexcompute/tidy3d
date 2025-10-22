from __future__ import annotations

from typing import ForwardRef, Union

from tidy3d.components.types import ArrayLike, Complex

NumberType = int | float | Complex | ArrayLike

ExpressionType = ForwardRef("ExpressionType")

NumberOrExpression = Union[NumberType, ExpressionType]  # noqa: UP007
