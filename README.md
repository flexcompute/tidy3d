# Tidy3D (Beta release)

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/HEAD?filepath=docs/source/notebooks/)
![tests](https://github.com/flexcompute/Tidy3D-client-revamp/actions/workflows//run_tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/?badge=latest)


![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/Tidy3D-logo.svg)

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/snippet.png)

## Installation

### Using pip

```
pip install tidy3d-beta
```

### From source

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

Can verify it worked by running

```
python -c "import tidy3d as td; print(td.__version__)"
```

and it should print out the version number, for example:

```
0.2.0
```

## Documentation

View our documentation [here](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/).
And see the source code (if you wish) [here](https://github.com/flexcompute-readthedocs/tidy3d-docs).
