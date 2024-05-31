#!/bin/bash
set -e

ruff format tidy3d/ --check --diff
ruff format tests/ --check --diff
ruff format scripts/ --check --diff

ruff check tidy3d --diff

pytest -rA tests/
# to test without vtk, one has to restart pytest
pytest -rA tests/_test_data/_test_datasets_no_vtk.py
pytest --doctest-modules tidy3d/ docs/
