"""Preserve Sphinx viewcode links for the split monitor-data package."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from sphinx.pycode import ModuleAnalyzer

from tidy3d.components.data import monitor_data

_FACADE_MODULE = "tidy3d.components.data.monitor_data"
_FACADE_MODULES = {
    _FACADE_MODULE: monitor_data,
    f"{_FACADE_MODULE}.field": monitor_data.field,
    f"{_FACADE_MODULE}.mode": monitor_data.mode,
    f"{_FACADE_MODULE}.projection": monitor_data.projection,
}
_PUBLIC_CLASSES = (monitor_data.field.ElectromagneticFieldData,)


def _viewcode_aliases() -> dict[str, dict[str, str]]:
    """Map facade class members to their focused module-level source tags."""
    aliases: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for cls in _PUBLIC_CLASSES:
        for member_name, descriptor in vars(cls).items():
            if member_name.startswith("_"):
                continue
            if isinstance(descriptor, property):
                implementation = descriptor.fget
            elif isinstance(descriptor, (classmethod, staticmethod)):
                implementation = descriptor.__func__
            else:
                implementation = descriptor
            if not inspect.isfunction(implementation):
                continue

            module_name = implementation.__module__
            if not module_name.startswith(f"{_FACADE_MODULE}."):
                continue
            if "." in implementation.__qualname__:
                continue
            aliases[module_name][f"{cls.__name__}.{member_name}"] = implementation.__name__
    return dict(aliases)


_VIEWCODE_ALIASES = _viewcode_aliases()


def _implementation_module(_app: Any, module_name: str, fullname: str) -> str | None:
    """Return the focused module implementing a legacy facade member."""
    facade = _FACADE_MODULES.get(module_name)
    if facade is None:
        return None

    member_name = fullname.removeprefix(f"{module_name}.")
    value: Any = facade
    try:
        for part in member_name.split("."):
            value = getattr(value, part)
    except AttributeError:
        return None

    implementation = value.fget if isinstance(value, property) else value
    implementation_module = getattr(implementation, "__module__", None)
    if implementation_module and implementation_module.startswith(f"{_FACADE_MODULE}."):
        return implementation_module
    return None


def _find_source(_app: Any, module_name: str):
    """Add facade-qualified aliases to focused source modules."""
    aliases = _VIEWCODE_ALIASES.get(module_name)
    if aliases is None:
        return None

    analyzer = ModuleAnalyzer.for_module(module_name)
    analyzer.find_tags()
    tags = dict(analyzer.tags)
    tags.update({public_name: tags[source_name] for public_name, source_name in aliases.items()})
    return analyzer.code, tags


def setup(app: Any) -> dict[str, bool]:
    """Register source-module resolution for legacy monitor-data members."""
    app.connect("viewcode-follow-imported", _implementation_module)
    app.connect("viewcode-find-source", _find_source)
    return {"parallel_read_safe": True}
