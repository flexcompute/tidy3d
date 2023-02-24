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

### Installing on Windows

Pre-release `1.9.0rc1` introduces the `adjoint` plugin, which uses [jax](https://jax.readthedocs.io/en/latest/) for automatic differentiation of tidy3d simulations. As windows users may have trouble installing `jax`, the recommended approach is to use [jax-windows-builder](https://github.com/cloudhan/jax-windows-builder) to first install jaxlib before tidy3d.

```
pip install "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install tidy3d
```

More details can be found [here](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-jaxlib-from-source-on-windows).


### Installing from source

For development purposes, and to get the latest development versions, you can download and install the package from source as:

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

### Configuring and authentication

Authentication (linking the front end to your account) will be done via an API key moving forward.

You can find your API key in the web interface http://tidy3d.simulation.cloud

After signing in, copy the API key for the next steps.

To set up the API key to work with Tidy3D, we need to store it either in the `~/.tidy3d/config` file or an environment variable.

You can set it up using one of three following options.

#### Command line (recommended)

``tidy3d configure`` and then enter your API key when prompted.

#### Manually

For an API key of `{your_api_key}`, you may run

``echo 'apikey = "{your_api_key}"' > ~/.tidy3d/config``

or manually insert the line `'apikey = "{your_api_key}"` in the `~/.tidy3d/config` file.

#### Environment Variable

Set the `SIMCLOUD_API_KEY` environment variable to your API key (in quotes).

``export SIMCLOUD_API_KEY="{your_api_key}"``

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

To test the web / authentication

```
python -c "import tidy3d.web"
```

## Issues / Feedback / Bug Reporting

Your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/main/LICENSE)
