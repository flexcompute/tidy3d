#!/bin/bash
set -e

black tidy3d/
black tests/
black scripts/

ruff check tidy3d

pytest -rA tests/
# to test without vtk, one has to restart pytest
pytest -rA tests/_test_data/_test_datasets_no_vtk.py
pytest --doctest-modules tidy3d/ docs/
