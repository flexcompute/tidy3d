"""Tests for exported schema generation."""

from __future__ import annotations

import tidy3d as td
from scripts.regenerate_schema import _canonicalize, _materialize_default_factory_defaults


def test_simulation_default_factory_defaults_are_exported():
    schema = td.Simulation.model_json_schema()
    schema = _materialize_default_factory_defaults(schema, td.Simulation)
    schema = _canonicalize(schema)

    medium_default = schema["properties"]["medium"]["default"]
    assert medium_default["type"] == "Medium"
    assert medium_default["permittivity"] == 1.0

    boundary_default = schema["properties"]["boundary_spec"]["default"]
    assert boundary_default["type"] == "BoundarySpec"

    grid_default = schema["properties"]["grid_spec"]["default"]
    assert grid_default["type"] == "GridSpec"

    subpixel_default = schema["properties"]["subpixel"]["default"]
    assert subpixel_default["type"] == "SubpixelSpec"

    subpixel_spec = schema["$defs"]["SubpixelSpec"]["properties"]
    assert subpixel_spec["dielectric"]["default"]["type"] == "PolarizedAveraging"
    assert subpixel_spec["metal"]["default"]["type"] == "Staircasing"

    assert "title" not in schema["properties"]["medium"]
    assert "description" not in schema["properties"]["medium"]


def test_field_monitor_apodization_default_is_exported():
    schema = td.FieldMonitor.model_json_schema()
    schema = _materialize_default_factory_defaults(schema, td.FieldMonitor)
    schema = _canonicalize(schema)

    apodization_default = schema["properties"]["apodization"]["default"]
    assert apodization_default["type"] == "ApodizationSpec"
    assert apodization_default["start"] is None
    assert apodization_default["end"] is None
    assert apodization_default["width"] is None
