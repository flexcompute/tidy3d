"""
Generate Tidy3D JSON Schemas (docs-free, deterministic).

This utility exports JSON Schemas for key Tidy3D models and writes them into
the repository `schemas/` directory, with two strict guarantees:

- Documentation-free: remove all "title", "description", and "units" fields at every level.
- Default-preserving: materialize zero-argument `default_factory` values into exported defaults.
- Canonicalized: deterministically sort keys and certain lists for stable output
  across Python versions.

Note: The behavior is always docs-free and canonicalized.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

# Define the output directory relative to this script's location.
# Assumes the script is in a subdirectory like 'scripts' and 'schemas' is a sibling.
DEFAULT_SCHEMA_DIR = pathlib.Path(__file__).parent.parent / "schemas"

# Dictionary mapping a clean name to the Pydantic model class.
# This is the single source of truth for which schemas to export.
export_api_schema_dictionary = None  # populated lazily when generating


def _schema_definitions(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the definitions mapping for either pydantic v1 or v2 style schemas."""
    for key in ("$defs", "definitions"):
        definitions = schema_dict.get(key)
        if isinstance(definitions, dict):
            return definitions
    return {}


def _iter_model_types(annotation: Any):
    """Yield BaseModel subclasses nested inside a field annotation."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return

    origin = get_origin(annotation)
    if origin is None:
        return

    for arg in get_args(annotation):
        yield from _iter_model_types(arg)


def _find_model_schema(
    schema_dict: dict[str, Any], model_cls: type[BaseModel]
) -> dict[str, Any] | None:
    """Locate the schema node corresponding to a model class."""
    if schema_dict.get("title") == model_cls.__name__:
        return schema_dict

    definitions = _schema_definitions(schema_dict)
    definition = definitions.get(model_cls.__name__)
    if isinstance(definition, dict):
        return definition

    for candidate in definitions.values():
        if isinstance(candidate, dict) and candidate.get("title") == model_cls.__name__:
            return candidate
    return None


def _serialize_field_default(field: Any) -> tuple[bool, Any]:
    """Serialize a field's default_factory value into JSON-compatible data."""
    if field.default_factory is None:
        return False, None
    if getattr(field, "default_factory_takes_validated_data", False):
        return False, None

    try:
        default_value = field.get_default(call_default_factory=True)
    except Exception:
        return False, None

    try:
        return True, TypeAdapter(field.annotation).dump_python(default_value, mode="json")
    except Exception:
        if isinstance(default_value, BaseModel):
            return True, default_value.model_dump(mode="json")
        return False, None


def _materialize_default_factory_defaults(
    schema_dict: dict[str, Any], model_cls: type[BaseModel]
) -> dict[str, Any]:
    """Inject defaults for zero-argument default_factory fields into a schema tree."""
    visited: set[type[BaseModel]] = set()

    def _visit(current_model: type[BaseModel]) -> None:
        if current_model in visited:
            return
        visited.add(current_model)

        current_schema = _find_model_schema(schema_dict, current_model)
        if not isinstance(current_schema, dict):
            return

        properties = current_schema.get("properties")
        if not isinstance(properties, dict):
            return

        for field_name, field in current_model.model_fields.items():
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and "default" not in field_schema:
                has_default, default_value = _serialize_field_default(field)
                if has_default:
                    field_schema["default"] = default_value

            for nested_model in _iter_model_types(field.annotation):
                _visit(nested_model)

    _visit(model_cls)
    return schema_dict


def _stable_sort_key_for_schema_item(item: Any) -> str:
    """Return a stable string key for sorting schema objects in anyOf/oneOf/allOf arrays.

    Input is assumed to be already canonicalized and docs-free; we defensively drop
    top-level doc fields in case of partially processed inputs.
    """
    try:
        if isinstance(item, dict):
            item = {k: v for k, v in item.items() if k not in {"title", "description", "units"}}
        return json.dumps(item, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(item)


def _canonicalize(obj: Any) -> Any:
    """Recursively canonicalize a schema object for deterministic, docs-free output.

    Rules:
    - Drop all "description", "title", and "units" keys everywhere.
    - Sort all dict keys recursively.
    - Sort arrays that are order-insensitive: "required", "enum", and "type" (when list).
    - For "anyOf"/"oneOf"/"allOf", sort entries by a stable key after canonicalization.
    """
    if isinstance(obj, dict):
        # Canonicalize nested values and drop doc keys
        canon: dict[str, Any] = {}
        for k, v in obj.items():
            if k in {"description", "title", "units"}:
                continue  # drop docs
            canon[k] = _canonicalize(v)

        # Normalize known order-insensitive array fields
        if isinstance(canon.get("required"), list):
            canon["required"] = sorted(set(canon["required"]))
        if isinstance(canon.get("enum"), list):
            try:
                canon["enum"] = sorted(canon["enum"], key=lambda x: json.dumps(x, sort_keys=True))
            except Exception:
                canon["enum"] = sorted(canon["enum"], key=str)
        if isinstance(canon.get("type"), list):
            canon["type"] = sorted(canon["type"], key=str)

        # Combination keywords: ensure stable order using canonicalized entries
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(canon.get(key), list):
                canon[key] = sorted(canon[key], key=_stable_sort_key_for_schema_item)

        # Return dict with sorted keys
        return {k: canon[k] for k in sorted(canon.keys())}
    elif isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    else:
        return obj


def _load_tidy3d_models():
    """Import and return the mapping of schema names to Tidy3D model classes.

    Import is done lazily to avoid importing tidy3d when the module is merely inspected.
    """
    try:
        from tidy3d import (
            EMESimulation,
            HeatChargeSimulation,
            HeatSimulation,
            ModeSimulation,
            Simulation,
        )
        from tidy3d.plugins.smatrix import TerminalComponentModeler
    except Exception as e:
        print(
            f"Error: Failed to import from 'tidy3d'. Ensure it's installed. Details: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "Simulation": Simulation,
        "ModeSimulation": ModeSimulation,
        "EMESimulation": EMESimulation,
        "HeatSimulation": HeatSimulation,
        "HeatChargeSimulation": HeatChargeSimulation,
        "TerminalComponentModeler": TerminalComponentModeler,
    }


def generate_schemas(output_dir: pathlib.Path = DEFAULT_SCHEMA_DIR):
    """
    Generates and saves a JSON schema for each class in the global dictionary.

    This function handles the creation of the output directory and iterates
    through each item in `export_api_schema_dictionary`. For each item, it
    generates the schema and writes it to a corresponding '.json' file.
    It includes error handling for file system operations.

    Raises:
        OSError: If there is an issue creating the directory or writing files,
                 such as a permissions error.
    """
    try:
        # Create the output directory if it doesn't exist.
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving schemas to '{output_dir}/'")

        models = _load_tidy3d_models()
        for name, class_instance in models.items():
            output_path = output_dir / f"{name}.json"
            print(f"  -> Generating schema for '{name}'...")

            # Generate the schema dictionary from the class.
            schema_dict = class_instance.model_json_schema()
            schema_dict = _materialize_default_factory_defaults(schema_dict, class_instance)
            schema_dict = _canonicalize(schema_dict)

            # Write the schema to a file with pretty printing.
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(schema_dict, f, indent=2, sort_keys=True, ensure_ascii=True)
                f.write("\n")

    except OSError as e:
        print(
            "\nError: A file system error occurred. Check permissions and paths.", file=sys.stderr
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nSchema generation complete. ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate Tidy3D JSON Schemas")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_SCHEMA_DIR,
        help="Directory to write schema JSON files (default: repo 'schemas').",
    )
    args = parser.parse_args()

    # Encourage use of a pinned Python for stable output
    if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
        print(
            f"Warning: Running with Python {sys.version_info.major}.{sys.version_info.minor}. "
            "For stable schema output, prefer Python 3.11 (matches CI).",
            file=sys.stderr,
        )

    generate_schemas(args.output_dir)
