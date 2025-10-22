from __future__ import annotations

import importlib
import inspect
from typing import Annotated

from pydantic.v1 import Field

from tidy3d.components.types import TYPE_TAG_STR

from . import types as _types
from .base import Expression
from .functions import Cos, Exp, Log, Log10, Sin, Sqrt, Tan
from .metrics import ModeAmp, ModePower, generate_validation_data
from .variables import Constant, Variable

__all__ = [
    "Constant",
    "Cos",
    "Exp",
    "Expression",
    "Log",
    "Log10",
    "ModeAmp",
    "ModePower",
    "Sin",
    "Sqrt",
    "Tan",
    "Variable",
    "generate_validation_data",
]

# The following code dynamically collects all classes that are subclasses of Expression
# from the specified modules, builds a discriminated union for serialization, and updates their
# forward references. This is necessary to handle cases where classes reference each other before
# they are fully defined. The local_vars dictionary is used to store these classes and any other
# necessary types for the forward reference updates.

_module_names = ["base", "variables", "functions", "metrics", "operators"]
_model_classes = set()
_local_vars: dict[str, type[Expression]] = {}

for module_name in _module_names:
    module = importlib.import_module(f".{module_name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Expression):
            _model_classes.add(obj)
            _local_vars[name] = obj

_concrete_classes = [
    cls
    for cls in sorted(_model_classes, key=lambda candidate: candidate.__name__)
    if not inspect.isabstract(cls)
]

_expression_union = Expression
if _concrete_classes:
    _expression_union = _concrete_classes[0]
    for cls in _concrete_classes[1:]:
        _expression_union = _expression_union | cls

ExpressionType = Annotated[_expression_union, Field(discriminator=TYPE_TAG_STR)]
_types.ExpressionType = ExpressionType
_types.NumberOrExpression = _types.NumberType | ExpressionType

__all__.append("ExpressionType")
_local_vars["ExpressionType"] = ExpressionType

for cls in _model_classes:
    cls.update_forward_refs(**_local_vars)
