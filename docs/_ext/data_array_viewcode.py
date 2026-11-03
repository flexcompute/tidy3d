"""Document public array APIs backed by shared implementations."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from sphinx.pycode import ModuleAnalyzer

from tidy3d.components.data.data_array import DataArray

_DATA_ARRAY_MODULE_PREFIX = "tidy3d.components.data.data_array."
_PUBLIC_DATA_ARRAY_CLASSES = (DataArray,)


def _viewcode_aliases() -> dict[str, dict[str, str]]:
    """Map public data-array member names to extracted module-level source tags."""

    aliases: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for cls in _PUBLIC_DATA_ARRAY_CLASSES:
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
            if not module_name.startswith(_DATA_ARRAY_MODULE_PREFIX):
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
    """Register extracted data-array source aliases with Sphinx viewcode."""

    app.connect("viewcode-find-source", _find_source)
    app.connect("config-inited", _array_members)
    app.connect("autodoc-process-bases", _public_array_bases)
    return {"parallel_read_safe": True}


def _array_members(_app: Any, config: Any) -> None:
    """Expose public array members inherited from shared arrays, excluding xarray."""
    import tidy3d
    from tidy3d.components.data import data_array

    members = {}
    for module in (tidy3d, data_array):
        for name, cls in vars(module).items():
            if name.startswith("_") or not inspect.isclass(cls) or not issubclass(cls, DataArray):
                continue
            members[f"{module.__name__}.{name}"] = sorted(
                {
                    member
                    for base in cls.__mro__
                    if base.__module__.startswith(
                        (_DATA_ARRAY_MODULE_PREFIX, "flexcompute.data.arrays.")
                    )
                    for member in vars(base)
                    if not member.startswith("_") and member != "attrs"
                }
            )
    config.autosummary_context["data_array_members"] = members


def _public_array_bases(_app: Any, _name: str, obj: Any, _options: Any, bases: list) -> None:
    """Show the public hierarchy without exposing shared implementation bases."""
    import xarray as xr

    if inspect.isclass(obj) and issubclass(obj, DataArray):
        bases[:] = [base for base in bases if not base.__module__.startswith("flexcompute.data.")]
        if obj is DataArray:
            bases[:] = [xr.DataArray]
