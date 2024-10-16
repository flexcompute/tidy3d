from .base import Expression
from .functions import Cos, Exp, Log, Log10, Sin, Sqrt, Tan
from .metrics import ModeAmp, ModePower, generate_validation_data
from .variables import Constant, Variable

__all__ = [
    "Expression",
    "Constant",
    "Variable",
    "ModeAmp",
    "ModePower",
    "generate_validation_data",
    "Sin",
    "Cos",
    "Tan",
    "Exp",
    "Log",
    "Log10",
    "Sqrt",
]

# The following code dynamically collects all classes that are subclasses of Expression
# from the specified modules and updates their forward references. This is necessary to handle
# cases where classes reference each other before they are fully defined. The local_vars dictionary
# is used to store these classes and any other necessary types for the forward reference updates.

import importlib
import inspect

from .types import ExpressionType

_module_names = ["base", "variables", "functions", "metrics", "operators"]
_model_classes = set()
_local_vars = {"ExpressionType": ExpressionType}

for module_name in _module_names:
    module = importlib.import_module(f".{module_name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Expression):
            _model_classes.add(obj)
            _local_vars[name] = obj

for cls in _model_classes:
    cls.update_forward_refs(**_local_vars)
