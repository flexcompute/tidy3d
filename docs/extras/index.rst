Extras Plugin |:sparkles:|
==========================
.. currentmodule:: tidy3d-extras

``tidy3d-extras`` is an optional plugin for Tidy3D providing additional, more advanced local functionality. This includes subpixel-averaged permittivity and mode solving, broadband source injection, transmission line terminal analysis, and vector fitting of frequency-domain data.


Installation
------------

Simply go to `tidy3d.simulation.cloud <https://tidy3d.simulation.cloud/>`__ and sign up for an account.
From there, just load our `web GUI <https://tidy3d.simulation.cloud/folders>`__ (no installation required) or continue with the next steps for the Python interface.

``tidy3d-extras`` is a python module that can be easily installed via ``pip``.
The version must match the version of Tidy3D, so the preferred command
for installation is::

   pip install "tidy3d[extras]"

This command will install ``tidy3d-extras`` and its dependencies in the current environment.

.. important::

   ``tidy3d-extras`` is **not compatible with Conda environments**.
   Please use a standard Python virtual environment (e.g., ``venv`` or ``virtualenv``) for installation.

A Tidy3D API key is required to authenticate ``tidy3d-extras`` users.
If you don't have one already configured, you can `get a free API key <https://tidy3d.simulation.cloud/account?tab=apikey>`__ and configure it with the following command::

   tidy3d configure

On **Windows**, it is easier to use ``pipx`` to find the path to the configuration tool::

   pip install pipx
   pipx run tidy3d configure

More information about the installation and configuration of Tidy3D can be found `here <https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html>`__.

You can verify that the ``tidy3d-extras`` installation worked by running the following command to print the installed version::

   python -c 'import tidy3d_extras as tde; print(tde.__version__)'


Usage
-----

Currently, ``tidy3d-extras`` provides the following features:

- Local subpixel-averaged permittivity. This is what one would obtain
  with a ``PermittivityMonitor`` in an FDTD ``Simulation``.
- More accurate local mode solver with subpixel averaging.
- Mode solver for fully tensorial media, or for a mode specification with nonzero ``angle_theta`` and ``angle_rotation=False``, which creates an equivalent fully tensorial solve.
- ``BroadbandPulse`` source time for injecting significant energy across a wide custom frequency range.
- Computation of voltage and current transformation matrices for transmission line terminals. Requires an **RF license**.
- Vector fitting of array-valued frequency-domain data via ``_vector_fit``.

By default, these features are automatically enabled if the ``tidy3d-extras``
package is installed. The following functions benefit from local subpixel
averaging:

- ``Simulation.epsilon`` — returns subpixel-averaged permittivity.
- ``ModeSimulation.run_local`` — runs the mode simulation locally with
  subpixel averaging for improved accuracy.
- ``ModeSolver.solve`` — runs the local mode solver with subpixel averaging.

Sometimes, you may want to change this behavior, for example to speed up
permittivity computation. In this case, you can temporarily disable these
features through the configuration manager::

   from tidy3d import config

   config.simulation.use_local_subpixel = False

Set the value to ``True`` to ensure that subpixel averaging is used, then call
``config.save()`` if you would like the change to persist across sessions.
You can check whether local subpixel averaging is turned on::

   tidy3d.packaging.tidy3d_extras["use_local_subpixel"]

For a broader overview of configuration options and how they are stored, see
:doc:`../configuration/index`.

Licenses
--------

.. toctree::
   :maxdepth: 2

   license
   third_party_licenses
