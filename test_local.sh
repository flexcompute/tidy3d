#!/bin/bash
set -e

black .
ruff check tidy3d

pytest -rA tests/

# test no vtk available (must be done separately from other tests to reload tidy3d from scratch)
pytest -rA tests/test_data/_test_datasets_no_vtk.py

pytest --doctest-modules tidy3d/components
