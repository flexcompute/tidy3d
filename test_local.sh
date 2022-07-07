#!/bin/bash
set -e

black .
python lint.py
pytest -ra tests/test_components.py
pytest -ra tests/test_grid.py
pytest -ra tests/test_boundaries.py
pytest -ra tests/test_IO.py
pytest -ra tests/test_material_library.py
pytest -ra tests/test_plugins.py
pytest -ra tests/test_sidewall.py
pytest -ra tests/test_meshgenerate.py
pytest -ra tests/test_near2far.py

pytest --doctest-modules tidy3d/components