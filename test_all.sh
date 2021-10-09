#!/bin/bash

black .
python lint.py
pytest -rA tests/
pytest --doctest-modules tidy3d --ignore=tidy3d/__main__.py