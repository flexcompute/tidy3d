# Tidy3D

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks)
![tests](https://github.com/flexcompute/tidy3d/actions/workflows//run_tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/flexcompute/tidy3d.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/flexcompute/tidy3d/context:python)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tidy3d.svg)](https://pypi.python.org/pypi/tidy3d/)

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/Tidy3D-logo.svg)

Tidy3D is a software product from Flexcompute that enables large scale electromagnetic simulation using the finite-difference time-domain (FDTD) method.

This repository stores the python interface for Tidy3d.

This code allows you to:
* Programmatically define FDTD simulations.
* Submit and magange simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.

You can find a detailed documentation and API reference [here](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/).
The source code for our documentation is [here](https://github.com/flexcompute-readthedocs/tidy3d-docs).

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/snippet.png)

## Installation

### Signing up for tidy3d

Note that while this front end package is open source, to run simulations on Flexcompute servers requires an account with credits.
You can sign up [here](https://client.simulation.cloud/register-waiting).  While it's currently a waitlist for new users, we will be rolling out to many more users in the coming weeks!  See [this page](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/quickstart.html) in our documentation for more details.

### Installing the front end 

#### Using pip (recommended)

The easiest way to install tidy3d is through [pip](https://pip.pypa.io/en/stable/).

```
pip install tidy3d
```

This will install the latest stable version, to get the a "pre-release" version.

```
pip install --pre tidy3d
```

And to get a specific version `x.y.z`

```
pip install tidy3d==x.y.z
```

### Installing from source

For development purposes, and to get the latest development versions, you can download and install the package from source as:

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

### Configuring and authentication

The Tidy3D front end must be configured with your account information, which is done via an API key.

You can find your API key in the [web interface](ehttp://tidy3d.simulation.cloud). After signing in and navigating to the account page by clicking the "user" icon on the left-hand side, copy the API key from the button on the right-hand side of the page.

To set up the API key to work with Tidy3D, you may use one of three following options:
Note: We refer to your API specific API key value as `XXX` below.

#### Command line (recommended)

``tidy3d configure`` and then enter your API key `XXX` when prompted.

Note that Windows users must run the following instead (ideally in an anaconda prompt):

```
pip install pipx
pipx run tidy3d configure
```

Note that the `api-key` flag  can be provided in this command, eg.

```
tidy3d configure --api-key=XXX
```

#### Manually

Alternatively, you can place the API key directly in the file where Tidy3D looks for it.

``echo 'apikey = "XXX"' > ~/.tidy3d/config``

or manually insert the line `'apikey = "XXX` in the `~/.tidy3d/config` file.

#### Environment Variable

Lastly, you may set the API key as an environment variable called `SIMCLOUD_APIKEY`.

``export SIMCLOUD_APIKEY="XXX"``

### Testing the installation and authentication

#### Front end package

You can verify the front end installation worked by running:

```
python -c "import tidy3d as td; print(td.__version__)"
```

and it should print out the version number, for example:

```
1.0.0
```

#### Authentication

To test the authentication, try importing the web interface via.

```
python -c "import tidy3d.web"
```

It should pass without any errors if the API key is set up correctly.

## Issues / Feedback / Bug Reporting

Your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/main/LICENSE)
