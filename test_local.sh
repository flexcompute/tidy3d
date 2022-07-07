#!/bin/bash
set -e

black .
python lint.py
pytest -ra tests/test_components.py
pytest -ra tests/test_boundaries.py
pytest -ra tests/test_config.py
pytest -ra tests/test_data.py
pytest -ra tests/test_geo_group.py
pytest -ra tests/test_grid.py
pytest -ra tests/test_IO.py
pytest -ra tests/test_log.py
pytest -ra tests/test_main.py
pytest -ra tests/test_material_library.py
pytest -ra tests/test_meshgenerate.py
pytest -ra tests/test_plugins.py
pytest -ra tests/test_sidewall.py

pytest --doctest-modules tidy3d/components