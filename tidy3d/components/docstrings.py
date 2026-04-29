"""Helpers for rendering concise docstrings for Tidy3D models."""

from __future__ import annotations

import re
import types as _types
from types import UnionType
from typing import Any, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel, TypeAdapter

from tidy3d.log import log

_TYPING_PREFIXES = ("typing_extensions.", "typing.")
_AUTOGRAD_BOX_TYPE = "autograd.tracer.Box"
_DEFAULT_SENTINEL = object()
_DOCSTRING_RAW_ATTR = "__tidy3d_raw_doc__"


def _strip_typing_prefixes(type_str: str) -> str:
    """Remove verbose typing prefixes from rendered annotations."""
    for prefix in _TYPING_PREFIXES:
        type_str = type_str.replace(prefix, "")
    return type_str


def _split_annotated(ann: Any) -> tuple[Any, list[Any]]:
    """Return (base, metadata) for Annotated types, otherwise (ann, [])."""
    origin = get_origin(ann)
    if origin is not None and getattr(origin, "__name__", None) == "Annotated":
        args = get_args(ann)
        if not args:
            return ann, []
        return args[0], list(args[1:])
    return ann, []


def _strip_annotated(ann: Any) -> Any:
    """Remove typing.Annotated wrappers while keeping the base annotation."""
    while True:
        base, metadata = _split_annotated(ann)
        if not metadata:
            return base
        ann = base


def _format_type_name(ann: Any) -> str:
    """Return a concise, human-readable name for a type-like object."""
    if ann is type(None):
        return "None"
    if ann is object:
        return "Any"
    try:
        from tidy3d.components.base import Tidy3dBaseModel

        if isinstance(ann, type) and issubclass(ann, Tidy3dBaseModel):
            return f":class:`~{ann.__module__}.{ann.__name__}`"
    except Exception:
        pass
    forward_arg = getattr(ann, "__forward_arg__", None)
    if forward_arg:
        if forward_arg.startswith("tidy3d."):
            return f":class:`~{forward_arg}`"
        return forward_arg
    if isinstance(ann, str) and ann.startswith("tidy3d."):
        return f":class:`~{ann}`"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return _strip_typing_prefixes(str(ann))


def _clean_default_repr(value: str) -> str:
    """Remove noisy fields from default BaseModel repr strings."""
    cleaned = value.replace("attrs={}, ", "")
    cleaned = cleaned.replace("attrs={}", "")
    # Remove only the discriminator field named exactly `type=...` without touching
    # other fields like `simulation_type=...`.
    cleaned = re.sub(r"(?<=\()\s*type=(?:'[^']*'|\"[^\"]*\")\s*,\s*", "", cleaned)
    cleaned = re.sub(r"(?<=,)\s*type=(?:'[^']*'|\"[^\"]*\")\s*,\s*", "", cleaned)
    cleaned = re.sub(r",\s*type=(?:'[^']*'|\"[^\"]*\")\s*\)", ")", cleaned)
    cleaned = re.sub(r"\(\s*type=(?:'[^']*'|\"[^\"]*\")\s*\)", "()", cleaned)
    cleaned = re.sub(r"\(\s*,\s*", "(", cleaned)
    cleaned = re.sub(r",\s*\)", ")", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    return " ".join(cleaned.split())


def _format_default_value(value: Any, *, show_default_args: bool) -> str:
    """Format default values for docstrings, handling nested models."""
    if isinstance(value, BaseModel):
        return _format_model_default(value, show_default_args=show_default_args)
    return _clean_default_repr(str(value))


def _format_model_default(model: BaseModel, *, show_default_args: bool) -> str:
    """Render a model default, optionally collapsing to only non-default fields."""

    # IMPORTANT: never call `repr(model)` here. Tidy3dBaseModel overrides __repr__ to
    # call `_format_model_default(..., show_default_args=False)`, so `repr(model)`
    # would recurse for any model class that can't be instantiated with no args.
    def base_repr(m: BaseModel) -> str:
        return BaseModel.__repr__(m)

    if show_default_args:
        return _clean_default_repr(" ".join(base_repr(model).split()))

    model_cls = model.__class__
    try:
        # Suppress log output during speculative default model creation, as some models
        # (e.g. CustomMedium, ParameterPerturbation) cannot be instantiated without
        # required arguments and their validators log error messages before raising.
        with log.suppress_output():
            default_model = model_cls()
        current_dump = model.model_dump()
        default_dump = default_model.model_dump()
    except Exception:
        return _clean_default_repr(" ".join(base_repr(model).split()))

    diff_keys: set[str] = set()
    for key, val in current_dump.items():
        if default_dump.get(key, _DEFAULT_SENTINEL) != val:
            diff_keys.add(key)

    if not diff_keys:
        return f"{model_cls.__name__}()"

    ordered_fields = list(getattr(model_cls, "model_fields", {}).keys())
    parts: list[str] = []
    for key in ordered_fields:
        if key in diff_keys:
            try:
                value = getattr(model, key)
            except Exception:
                value = current_dump.get(key)
            parts.append(
                f"{key}={_format_default_value(value, show_default_args=show_default_args)}"
            )

    for key in diff_keys:
        if key not in ordered_fields:
            parts.append(
                f"{key}={_format_default_value(current_dump[key], show_default_args=show_default_args)}"
            )

    return _clean_default_repr(f"{model_cls.__name__}({', '.join(parts)})")


def _array_schema_from_metadata(metadata: list[Any]) -> dict[str, Any] | None:
    """Extract an ArrayLike schema dict from annotation metadata if present."""
    for meta in metadata:
        schema = getattr(meta, "json_schema", None)
        if isinstance(schema, dict) and schema.get("type") == "ArrayLike":
            return schema
    return None


def _format_array_like(schema: dict[str, Any]) -> str:
    """Format ArrayLike metadata into a concise dtype/ndim/shape string."""
    dtype = schema.get("x-array-dtype")
    ndim = schema.get("x-array-ndim")
    shape = schema.get("x-array-shape")

    dtype_str = None
    if dtype is not None:
        try:
            np_dtype = np.dtype(dtype)
            kind = np_dtype.kind
            dtype_str = {
                "f": "float",
                "c": "complex",
                "i": "int",
                "u": "int",
                "b": "bool",
            }.get(kind, np_dtype.name)
        except Exception:
            dtype_str = str(dtype)

    parts: list[str] = []
    if dtype_str is not None:
        parts.append(f"dtype={dtype_str}")
    if ndim is not None:
        parts.append(f"ndim={ndim}")
    if shape is not None:
        parts.append(f"shape={shape}")

    if not parts:
        return "ArrayLike"
    return f"ArrayLike[{', '.join(parts)}]"


def _is_annotation_candidate(val: Any) -> bool:
    """Return True for values that look like a typing annotation."""
    if isinstance(val, type):
        return True
    try:
        return get_origin(val) is not None or bool(get_args(val))
    except Exception:
        return False


def _extract_traced_alias_base(metadata: list[Any]) -> Any | None:
    """Extract a base annotation from validator closures used by traced aliases."""
    for meta in metadata:
        func = getattr(meta, "func", None)
        if func is None or getattr(func, "__name__", "") != "_validate_box_or_container":
            continue
        if not func.__closure__:
            continue
        for cell in func.__closure__:
            val = cell.cell_contents
            if isinstance(val, TypeAdapter):
                continue
            if isinstance(val, (_types.FunctionType, _types.MethodType)):
                continue
            if _is_annotation_candidate(val):
                return val
    return None


def _constraint_alias(base: Any, metadata: list[Any]) -> str | None:
    """Map constrained Annotated metadata to concise alias names when possible."""
    if len(metadata) != 1:
        return None
    meta = metadata[0]
    meta_name = meta.__class__.__name__
    if base is int:
        if meta_name == "Gt" and getattr(meta, "gt", None) == 0:
            return "PositiveInt"
        if meta_name == "Ge" and getattr(meta, "ge", None) == 0:
            return "NonNegativeInt"
    if base is float:
        if meta_name == "Gt" and getattr(meta, "gt", None) == 0:
            return "PositiveFloat"
        if meta_name == "Ge" and getattr(meta, "ge", None) == 0:
            return "NonNegativeFloat"
    return None


def _format_annotation(ann: Any, field_metadata: list[Any] | None = None) -> str:
    """Format annotations for docstrings, stripping Annotated metadata."""
    base, metadata = _split_annotated(ann)
    combined_metadata = list(metadata)
    if field_metadata:
        combined_metadata.extend(field_metadata)
    if combined_metadata:
        array_schema = _array_schema_from_metadata(combined_metadata)
        if array_schema is not None and base is np.ndarray:
            return _format_array_like(array_schema)

        traced_base = _extract_traced_alias_base(combined_metadata)
        if traced_base is not None:
            traced_fmt = _format_annotation(traced_base)
            return f"Union[{traced_fmt}, {_AUTOGRAD_BOX_TYPE}]"
        alias = _constraint_alias(base, combined_metadata)
        if alias is not None:
            return alias
        ann = base
    else:
        ann = base
    origin = get_origin(ann)

    if origin in (Union, UnionType):
        raw_args = list(get_args(ann))
        non_none = [arg for arg in raw_args if _strip_annotated(arg) is not type(None)]
        has_none = len(non_none) != len(raw_args)
        if has_none:
            if len(non_none) == 1:
                return f"Optional[{_format_annotation(non_none[0])}]"
            inner = ", ".join(_format_annotation(arg) for arg in non_none)
            return f"Optional[Union[{inner}]]"
        return f"Union[{', '.join(_format_annotation(arg) for arg in raw_args)}]"

    if origin is not None:
        if getattr(origin, "__name__", None) == "Literal":
            literal_args = ", ".join(repr(arg) for arg in get_args(ann))
            return f"Literal[{literal_args}]"

        origin_name = _format_type_name(origin)
        args = get_args(ann)
        if args:
            arg_strs = []
            for arg in args:
                if arg is Ellipsis:
                    arg_strs.append("...")
                else:
                    arg_strs.append(_format_annotation(arg))
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    return _format_type_name(ann)


def _fmt_ann_literal(ann: Any, field_metadata: list[Any] | None = None) -> str:
    """Render a concise annotation string for docstrings."""
    if ann is None:
        return "Any"
    return _format_annotation(ann, field_metadata=field_metadata)
