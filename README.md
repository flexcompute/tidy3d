# Tidy3D Documentation

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/?badge=latest)
![tests](https://github.com/flexcompute/Tidy3D-client-revamp/actions/workflows//run_tests.yml/badge.svg)

## Website

The website can be found [here](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/).

## Notebooks

The notebooks are in `docs/source/notebooks`.

To run the notebooks from browser, click [this link](https://mybinder.org/v2/gh/flexcompute/Tidy3D-docs/HEAD?filepath=docs/notebooks/) or the "Binder" tag at the top of this README.

## Setup

First time you want to use the docs, install all packages and make the docs building script executable.

```bash
git submodule --init tidy3d
pip install -r docs/requirements.txt
pip install -r tests/requirements.txt
```

## Compiling

To compile the docs:

```bash
cd docs/source
rm -rf _build
rm -rf _autosummary
python -m sphinx -T -b html -d _build/doctrees -D language=en . _build/html
open _build/html/index.html
```

## Tests

To run the tests

```bash
pytest -rA tests
```
