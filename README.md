# Tidy3D

[![Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks)
![tests](https://github.com/flexcompute/tidy3d/actions/workflows//run_tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest)](https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/?badge=latest)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tidy3d.svg)](https://pypi.python.org/pypi/tidy3d/)

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/Tidy3D-logo.svg)

Tidy3D is a software product from Flexcompute that enables large scale electromagnetic simulation using the finite-difference time-domain (FDTD) method.

This repository stores the python interface for Tidy3d.

This code allows you to:
* Programmatically define FDTD simulations.
* Submit and magange simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.

You can find a detailed documentation and API reference [here](https://docs.flexcompute.com/projects/tidy3d/en/stable/).
The source code for our documentation is [here](https://github.com/flexcompute-readthedocs/tidy3d-docs).

![](https://raw.githubusercontent.com/flexcompute/tidy3d/main/img/snippet.png)

## Installation

### Signing up for tidy3d

Note that while this front end package is open source, to run simulations on Flexcompute servers requires an account with credits.
You can sign up for an account [here](https://tidy3d.simulation.cloud/signup).
After that, you can install the front end with the instructions below, or visit [this page](https://docs.flexcompute.com/projects/tidy3d/en/stable/quickstart.html) in our documentation for more details.

### Installing the front end 

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

If those commands did not work, advanced installation instructions are below, which should help solve the issue.

### Advanced Installation Instructions

Some users might require more a specialized installation, which we cover below.

#### Using pip (recommended)

The easiest way to install the tidy3d python interface is through [pip](https://pypi.org/project/tidy3d/).

```
pip install tidy3d
```

This will install the latest stable version.

To get a specific version `x.y.z`, including the "pre-release" versions, you may specify the version in the command as:

```
pip install tidy3d==x.y.z
```

### Installing from source

Alternatively, for development purposes, eg. developing your own features, you may download and install the package from source as:

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

### Configuring and authentication

With the front end installed, it must now be configured with your account information, which is done via an "API key".

You can find your API key in the [web interface](http://tidy3d.simulation.cloud). After signing in and navigating to the account page by clicking the "Account Center" icon on the left-hand side. Then, click on the "API key" tab on the right hand side of the menu and copy your API key.

Note: We refer to your API specific API key value as `XXX` below.

To link your API key with Tidy3D, you may use one of three following options:

#### Command line (recommended)

The easiest way is through the command line via the `tidy3d configure` command. Run:

```
tidy3d configure
```

and then enter your API key `XXX` when prompted.

Note that Windows users must run the following instead (ideally in an anaconda prompt):

```
pip install pipx
pipx run tidy3d configure
```

You can also specify your API key directly as an option to this command using the `api-key` argument, for example:

```
tidy3d configure --apikey=XXX
```

#### Manually

Alternatively, you can manually set up the config file where Tidy3D looks for the API key. The API key must be in a file called `.tidy3d/config` located in your home directory, containing the following

```
apikey = "XXX"
```

You can manually set up your file like this, or do it through the command line line:

``echo 'apikey = "XXX"' > ~/.tidy3d/config``

Note the quotes around `XXX`.

Note that Windows users will most likely need to place the `.tidy3d/config` file in their `C:\Users\username\` directory (where `username` is your username).

#### Environment Variable

Lastly, you may set the API key as an environment variable named `SIMCLOUD_APIKEY`.

This can be set up using

``export SIMCLOUD_APIKEY="XXX"``

Note the quotes around `XXX`.

### Testing the installation and authentication

#### Front end package

You can verify the front end installation worked by running:

```
python -c "import tidy3d as td; print(td.__version__)"
```

and it should print out the version number, for example:

```
2.0.0
```

#### Authentication

To test the authentication, you may try importing the web interface via.

```
python -c "import tidy3d.web"
```

It should pass without any errors if the API key is set up correctly.

To get started, our documentation has a lot of [examples](https://docs.flexcompute.com/projects/tidy3d/en/latest/examples.html) for inspiration.

## Issues / Feedback / Bug Reporting

Your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/main/LICENSE)
