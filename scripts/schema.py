"""Generates schema for Simulation, saves it to file."""

import json
import tidy3d

FNAME = "tidy3d/schema.json"

# https://pydantic-docs.helpmanual.io/usage/schema/
schema_dict = tidy3d.Simulation.schema()
with open(FNAME, "w") as f:
    json.dump(schema_dict, f, indent=2)
