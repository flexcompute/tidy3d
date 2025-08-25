#!/bin/bash
set -e

ruff format tidy3d/ --check --diff
ruff format tests/ --check --diff
ruff format scripts/ --check --diff

ruff check tidy3d --diff
