"""Preserve Sphinx viewcode links for extracted simulation methods."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from sphinx.pycode import ModuleAnalyzer

from tidy3d.components.simulation import AbstractYeeGridSimulation, Simulation

_SIMULATION_MODULE_PREFIX = "tidy3d.components.simulation."
_PUBLIC_SIMULATION_CLASSES = (AbstractYeeGridSimulation, Simulation)


def _viewcode_aliases() -> dict[str, dict[str, str]]:
    """Map public class member names to their extracted module-level source tags."""

    aliases: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for cls in _PUBLIC_SIMULATION_CLASSES:
        for member_name, descriptor in vars(cls).items():
            if member_name.startswith("_"):
                continue
            implementation = (
                descriptor.__func__
                if isinstance(descriptor, (classmethod, staticmethod))
                else descriptor
            )
            if not inspect.isfunction(implementation):
                continue
            module_name = implementation.__module__
            if not module_name.startswith(_SIMULATION_MODULE_PREFIX):
                continue
            if "." in implementation.__qualname__:
                continue
            aliases[module_name][f"{cls.__name__}.{member_name}"] = implementation.__name__
    return dict(aliases)


_VIEWCODE_ALIASES = _viewcode_aliases()


def _find_source(_app: Any, module_name: str):
    """Add class-qualified aliases to Sphinx's tags for an extracted source module."""

    aliases = _VIEWCODE_ALIASES.get(module_name)
    if aliases is None:
        return None

    analyzer = ModuleAnalyzer.for_module(module_name)
    analyzer.find_tags()
    tags = dict(analyzer.tags)
    tags.update({public_name: tags[source_name] for public_name, source_name in aliases.items()})
    return analyzer.code, tags


def setup(app: Any) -> dict[str, bool]:
    """Register extracted simulation source aliases with Sphinx viewcode."""

    app.connect("viewcode-find-source", _find_source)
    return {"parallel_read_safe": True}
