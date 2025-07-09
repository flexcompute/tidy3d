"""Generates JSON schemas for various simulation and solver classes."""

from __future__ import annotations

import json
import pathlib

from tidy3d import EMESimulation, HeatChargeSimulation, ModeSimulation, Simulation

schemas_directory = pathlib.Path(__file__).parent.parent / pathlib.Path("schemas")

export_api_schema_dictionary = {
    "Simulation": Simulation,
    "ModeSimulation": ModeSimulation,
    "EMESimulation": EMESimulation,
    "HeatChargeSimulation": HeatChargeSimulation,
}


def generate_schemas():
    """
    Generates and saves a JSON schema for each class in export_api_schema_dictionary.
    """
    # Create the output directory if it doesn't exist.
    schemas_directory.mkdir(parents=True, exist_ok=True)
    print(f"Saving schemas to '{schemas_directory}/'")

    for name, class_instance in export_api_schema_dictionary.items():
        output_path = schemas_directory / f"{name}.json"
        print(f"  -> Generating schema for '{name}'...")

        # Generate the schema dictionary from the class.
        # Pydantic's .schema() method inspects the model and creates the JSON schema.
        schema_dict = class_instance.schema()

        # Write the schema to a file with pretty printing.
        with open(output_path, "w") as f:
            json.dump(schema_dict, f, indent=2)

    print("\nSchema generation complete. ✨")


if __name__ == "__main__":
    generate_schemas()
