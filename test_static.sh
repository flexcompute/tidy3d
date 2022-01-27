#!/bin/bash

black .
python lint.py
pytest -rA tests/test_components.py
pytest -rA tests/test_grid.py
pytest -rA tests/test_IO.py
pytest -rA tests/test_material_library.py
# pytest -rA tests/test_core.py
pytest -rA tests/test_plugins.py

pytest --doctest-modules tidy3d/components --ignore=tidy3d/components/base.py