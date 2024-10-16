# Tidy3D
[![PyPI
Name](https://img.shields.io/badge/pypi-tidy3d-blue?style=for-the-badge)](https://pypi.python.org/pypi/tidy3d)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tidy3d.svg?style=for-the-badge)](https://pypi.python.org/pypi/tidy3d/)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest&style=for-the-badge)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/?badge=latest)
![Tests](https://img.shields.io/github/actions/workflow/status/flexcompute/tidy3d/run_tests.yml?style=for-the-badge)
![License](https://img.shields.io/github/license/flexcompute/tidy3d?style=for-the-badge)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/daquinteroflex/4702549574741e87deaadba436218ebd/raw/tidy3d_extension.json)

[![Notebooks](https://img.shields.io/badge/Demo-Live%20notebooks-8A2BE2?style=for-the-badge)](https://github.com/flexcompute/tidy3d-notebooks)

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/Tidy3D-logo.svg)

Tidy3D is a software package for solving extremely large electrodynamics problems using the finite-difference time-domain (FDTD) method. It can be controlled through either an [open source python package](https://github.com/flexcompute/tidy3d) or a [web-based graphical user interface](https://tidy3d.simulation.cloud).

This repository contains the python API to allow you to:

* Programmatically define FDTD simulations.
* Submit and manage simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.


![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/snippet.png)

## Installation

### Signing up for tidy3d

Note that while this front end package is open source, to run simulations on Flexcompute servers requires an account with credits.
You can sign up for an account [here](https://tidy3d.simulation.cloud/signup).
After that, you can install the front end with the instructions below, or visit [this page](https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html) in our documentation for more details.

### Quickstart Installation

To install the Tidy3D Python API locally, the following instructions should work for most users.

```
pip install --user tidy3d
tidy3d configure --apikey=XXX
```

Where `XXX` is your API key, which can be copied from your [account page](https://tidy3d.simulation.cloud/account) in the web interface.

In a hosted jupyter notebook environment (eg google colab), it may be more convenient to install and configure via the following lines at the top of the notebook.

```
!pip install tidy3d
import tidy3d.web as web
web.configure("XXX")
```

**Advanced installation instructions for all platforms is available in the [documentation installation guides](https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html).**

### Authentication Verification

To test the authentication, you may try importing the web interface via.

```
python -c "import tidy3d; tidy3d.web.test()"
```

It should pass without any errors if the API key is set up correctly.

To get started, our documentation has a lot of [examples](https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html) for inspiration.

## Common Documentation References

| API Resource       | URL                                                                              |
|--------------------|----------------------------------------------------------------------------------|
| Installation Guide | https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html              |
| Documentation      | https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html                |
| Example Library    | https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html |
| FAQ                | https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/index.html       |


## Related Source Repositories

| Name              | Repository                                      |
|-------------------|-------------------------------------------------|
| Source Code       | https://github.com/flexcompute/tidy3d           |
| Notebooks Source  | https://github.com/flexcompute/tidy3d-notebooks |
| FAQ Source Code   | https://github.com/flexcompute/tidy3d-faq       |


## Issues / Feedback / Bug Reporting

Your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/main/LICENSE)
