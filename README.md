# Tidy3D (Beta release)

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks)
![tests](https://github.com/flexcompute/tidy3d/actions/workflows//run_tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/flexcompute/tidy3d.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/flexcompute/tidy3d/context:python)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tidy3d-beta.svg)](https://pypi.python.org/pypi/tidy3d-beta/)

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/Tidy3D-logo.svg)

Tidy3D is a software product from Flexcompute that enables large scale electromagnetic simulation using the finite-difference time-domain (FDTD) method.

This repository stores the python interface for the beta release of Tidy3D that will be officially released to the public in early 2022.

This code allows you to:
* Programmatically define FDTD simulations.
* Submit and magange simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.

You can find a detailed documentation and API reference [here](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/).
The source code for our documentation is [here](https://github.com/flexcompute-readthedocs/tidy3d-docs).

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/snippet.png)

## Installation

### Signing up for tidy3d

Note that while this front end package is open source, to run simulations on Flexcompute servers requires an account with credits.
You can sign up [here](https://client.simulation.cloud/register-waiting).  While it's currently a waitlist for new users, we will be rolling out to many more users in the coming weeks!  See [this page](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/en/latest/quickstart.html) in our documentation for more details.

### Installing the package using pip

The easiest way to install this beta version of tidy3d is through [pip](https://pip.pypa.io/en/stable/).

```
pip install tidy3d-beta
```

Note that while our old version is still currently pip installable as `tidy3d`, both versions are imoprted in python as `tidy3d`, eg. `import tidy3d as td`.

### (Alternativelty) installing from source

For development purposes, you can download and install the package from source as:

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

### Did it work?

You can verify the installation worked by running:

```
python -c "import tidy3d as td; print(td.__version__)"
```

and it should print out the version number, for example:

```
1.0.0
```

## Issues / Feedback / Bug Reporting

This is a beta release and your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/main/LICENSE)
