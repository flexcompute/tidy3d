#!/bin/bash

black .
python lint.py
pytest -rA tests/