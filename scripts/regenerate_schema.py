"""
Generates and saves JSON schemas for key tidy3d data structures.

This script iterates through a predefined dictionary of Tidy3D classes,
generates a Pydantic JSON schema for each, and saves it as a formatted
JSON file in the 'schemas' directory. It's designed to be run as a
standalone utility to update schema definitions.

All are GUI supported classes.
"""

from __future__ import annotations

import json
import pathlib
import sys

# Attempt to import necessary classes from tidy3d.
try:
    from tidy3d import (
        EMESimulation,
        HeatChargeSimulation,
        HeatSimulation,
        ModeSimulation,
        Simulation,
    )
    from tidy3d.plugins.smatrix import TerminalComponentModeler
except ImportError as e:
    print(
        f"Error: Failed to import from 'tidy3d'. Ensure it's installed. Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


# Define the output directory relative to this script's location.
# Assumes the script is in a subdirectory like 'scripts' and 'schemas' is a sibling.
SCHEMA_DIR = pathlib.Path(__file__).parent.parent / "schemas"

# Dictionary mapping a clean name to the Pydantic model class.
# This is the single source of truth for which schemas to export.
export_api_schema_dictionary = {
    "Simulation": Simulation,
    "ModeSimulation": ModeSimulation,
    "EMESimulation": EMESimulation,
    "HeatSimulation": HeatSimulation,
    "HeatChargeSimulation": HeatChargeSimulation,
    "TerminalComponentModeler": TerminalComponentModeler,
}


def generate_schemas():
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
        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving schemas to '{SCHEMA_DIR}/'")

        for name, class_instance in export_api_schema_dictionary.items():
            output_path = SCHEMA_DIR / f"{name}.json"
            print(f"  -> Generating schema for '{name}'...")

            # Generate the schema dictionary from the class.
            schema_dict = class_instance.schema()

            # Write the schema to a file with pretty printing.
            with open(output_path, "w") as f:
                json.dump(schema_dict, f, indent=2)

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
    generate_schemas()
