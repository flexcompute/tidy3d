# Tidy3D Python Client API Schemas

This directory contains JSON Schemas for Tidy3D API classes with GUI support.

Schemas are intentionally docs-free: all `title` and `description` fields are stripped to improve stability and make diffs meaningful. Output is canonicalized (sorted keys and order-insensitive arrays) so files are deterministic across Python versions. Zero-argument `default_factory` values are materialized into exported `default` entries when they can be instantiated and serialized.

## Regenerating Schemas

- Preferred command (stable output):
  `uv run -p 3.11 python scripts/regenerate_schema.py`
- The generator always writes docs‑free, canonicalized schemas to `schemas/`.

If you don’t have `uv`, use any Python (output is deterministic):

`python scripts/regenerate_schema.py`
