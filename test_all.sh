#!/bin/bash

black .
python lint.py
pytest -rA tests/
pytest --doctest-modules tidy3d \
--ignore=tidy3d/__main__.py \
--ignore=tidy3d/components/base.py \
--ignore=tidy3d/web/webapi.py \
--ignore=tidy3d/web/container.py \