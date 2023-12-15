#!/bin/bash
set -e

black .
ruff check tidy3d

pytest -rA tests/

pytest --doctest-modules tidy3d/components
