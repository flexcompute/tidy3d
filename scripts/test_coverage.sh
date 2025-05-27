#!/bin/bash
set -e

pytest -rA --cov tests/
# to test without vtk, one has to restart pytest
pytest -rA tests/_test_data/_test_datasets_no_vtk.py
pytest --doctest-modules tidy3d/ docs/
coverage xml
diff-cover --compare-branch=origin/develop coverage.xml
