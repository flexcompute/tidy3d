# Tidy3D Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-docs/HEAD?filepath=docs/notebooks/)

## Setup

First time you want to use the docs, install all packages and make the docs building script executable.

```bash
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
