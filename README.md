# Tidy3D Documentation

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/?badge=latest)
![tests](https://github.com/flexcompute/Tidy3D-client-revamp/actions/workflows//run_tests.yml/badge.svg)

## Setup

First time you want to use the docs, install all packages and make the docs building script executable.


```bash
git submodule --init tidy3d
pip install -r docs/requirements.txt
pip install -r tests/requirements.txt
chmod +x docs/build_docs.sh
```

## Docs

To compile the docs:

```bash
cd docs
bash build_docs.sh
open _build/index.html
```

## Notebooks

To run the notebooks from browser, click [this link](https://mybinder.org/v2/gh/flexcompute/Tidy3D-docs/HEAD?filepath=docs/notebooks/) or the "Binder" tag at the top of this README.

## Tests

To run the tests

```bash
pytest -rA tests
```
