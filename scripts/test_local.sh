#!/bin/bash
set -e

black .
ruff check tidy3d

pytest -rA tests/
# to test without vtk, one has to restart pytest
pytest -rA tests/test_data/_test_datasets_no_vtk.py

pytest --doctest-modules tidy3d/components
