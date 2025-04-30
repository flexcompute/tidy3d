"""Tidy3D top-level namespace
-  Reads the stub file (__init__.pyi) to discover the public API
-  Loads each sub-module only when you first touch the symbol
"""

import ast
import importlib
import pathlib
import threading
from importlib.metadata import version as _version
from types import ModuleType
from typing import Any

from .log import log

__version__ = _version("tidy3d")


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging_level = {level}``."
    )


log.info(f"Using client version: {__version__}")


def _build_lazy_map() -> dict[str, tuple[str, str]]:
    """Return a mapping  {exported_name: (relative_module_path, attr_in_module)}."""
    stub_path = pathlib.Path(__file__).with_suffix(".pyi")
    source = stub_path.read_text(encoding="utf-8")

    tree = ast.parse(source, filename=str(stub_path))
    mapping: dict[str, tuple[str, str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        dots = "." * node.level
        mod = node.module or ""
        module_path = f"{dots}{mod}" if mod or dots else ""

        for alias in node.names:
            public_name = alias.asname or alias.name
            mapping[public_name] = (module_path, alias.name)

    return mapping


_lazy_map = _build_lazy_map()
_lazy_map["material_library"] = (".material_lib.material_library", "material_library")

__all__ = list(_lazy_map.keys())

_lazy_lock = threading.RLock()


def __getattr__(name: str) -> Any:
    """Load ``name`` on first use, then cache it in the module globals."""
    try:
        module_path, attr_name = _lazy_map[name]
    except KeyError as exc:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}") from exc

    with _lazy_lock:
        if name in globals():
            return globals()[name]

        module: ModuleType = importlib.import_module(module_path, __package__)
        obj = getattr(module, attr_name)
        globals()[name] = obj
        return obj


def __dir__() -> list[str]:
    """Make `dir(mypackage)` show the exported names."""
    return sorted(list(globals().keys()) + __all__)
