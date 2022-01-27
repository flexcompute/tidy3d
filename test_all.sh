#!/bin/bash

black .
python lint.py
pytest -rA tests/
pytest --doctest-modules tidy3d/components --ignore=tidy3d/components/base.py
