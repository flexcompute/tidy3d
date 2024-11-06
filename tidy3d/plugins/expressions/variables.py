from typing import Any, Optional

import pydantic.v1 as pd

from .base import Expression
from .types import NumberType


class Variable(Expression):
    """
    Variable class represents a placeholder for a value provided at evaluation time.

    Attributes
    ----------
    name : Optional[str] = None
        The name of the variable used for lookup during evaluation.

    Methods
    -------
    evaluate(*args, **kwargs)
        Evaluates the variable by retrieving its value from provided arguments.

    Notes
    -----
    - If `name` is `None`, the variable expects a single positional argument during evaluation.
    - If `name` is provided, the variable expects a corresponding keyword argument during evaluation.
    - Mixing positional and keyword arguments is allowed.
    - Multiple positional arguments are disallowed and will raise a `ValueError`.

    Examples
    --------
    >>> x = Variable()  # unnamed
    >>> y = Variable(name='y')
    >>> z = Variable(name='z')
    >>> expr = x + y + z
    >>> expr(5, y=3, z=2)
    10
    """

    name: Optional[str] = pd.Field(
        None,
        title="Name",
        description="The name of the variable used for lookup during evaluation.",
    )

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        if self.name:
            if self.name not in kwargs:
                raise ValueError(f"Variable '{self.name}' not provided.")
            return kwargs[self.name]
        else:
            if not args:
                raise ValueError("No positional argument provided for unnamed variable.")
            if len(args) > 1:
                raise ValueError("Multiple positional arguments provided for unnamed variable.")
            return args[0]

    def __repr__(self) -> str:
        return self.name if self.name else "Variable()"


class Constant(Variable):
    """
    Constant class represents a fixed value in an expression.

    Attributes
    ----------
    value : NumberType
        The fixed value of the constant.

    Methods
    -------
    evaluate(*args, **kwargs)
        Returns the value of the constant.

    Examples
    --------
    >>> c = Constant(5)
    >>> c.evaluate()
    5
    """

    value: NumberType = pd.Field(
        ...,
        title="Value",
        description="The fixed value of the constant.",
    )

    def __init__(self, value: NumberType, **kwargs: dict[str, Any]) -> None:
        super().__init__(value=value, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value}"
