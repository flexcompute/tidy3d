*************************************
Electromagnetics Platform
*************************************

.. role:: raw-html(raw)
    :format: html

.. image:: https://img.shields.io/badge/pypi-tidy3d-blue?style=for-the-badge
   :target: https://pypi.python.org/pypi/tidy3d
.. image:: https://img.shields.io/pypi/v/tidy3d.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/tidy3d/
.. image:: https://readthedocs.com/projects/flexcompute-tidy3ddocumentation/badge/?version=latest&style=for-the-badge
   :target: https://flexcompute-tidy3ddocumentation.readthedocs-hosted.com/?badge=latest
.. image:: https://img.shields.io/github/actions/workflow/status/flexcompute/tidy3d/run_tests.yml?style=for-the-badge
.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/daquinteroflex/4702549574741e87deaadba436218ebd/raw/tidy3d_extension.json
.. image:: https://img.shields.io/github/license/flexcompute/tidy3d?style=for-the-badge
.. image:: https://img.shields.io/badge/Demo-Notebooks-9647AE?style=for-the-badge
    :target: https://github.com/flexcompute/tidy3d-notebooks
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
   :target: https://github.com/psf/black


Tidy3D is a software package for solving extremely large electrodynamics problems using the finite-difference time-domain (FDTD) method. It can be controlled through either an `open source python package <https://github.com/flexcompute/tidy3d>`_ or a `web-based graphical user interface <https://tidy3d.simulation.cloud>`_.

This python API allows you to:

* Programmatically define FDTD simulations.
* Submit and manage simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.




Get Started
===========

.. tabs::

    .. group-tab:: On Linux |:penguin:| & MacOS |:apple:|

        Install the latest stable python library `tidy3d <https://github.com/flexcompute/tidy3d>`_ for creating, managing, and postprocessing simulations with

        .. code-block:: bash

            pip install --user tidy3d

        Next, configure your ``tidy3d`` package with the API key from your account.

        `Get your free API key <https://tidy3d.simulation.cloud/account?tab=apikey>`_

        .. code-block:: bash

           tidy3d configure --apikey=XXX

        For more detailed installation instructions, see `this page <./install.html>`_,
        and refer to :doc:`configuration/index` if you would like to fine-tune your
        settings.

    .. group-tab:: On Windows |:window:|

        Install the latest stable python library `tidy3d <https://github.com/flexcompute/tidy3d>`_ for creating, managing, and postprocessing simulations in your virtual environment with:

        .. code-block:: bash

            pip install --user tidy3d

        Next, configure your ``tidy3d`` package with the API key from your account.

        `Get your free API key <https://tidy3d.simulation.cloud/account?tab=apikey>`_

        To automatically configure the API key, you will need to install the following extra packages:

        .. code-block:: bash

            pip install pipx
            pipx run tidy3d configure --apikey=XXX

        If you're running into trouble, you may need to manually set the API key
        directly in the configuration file where Tidy3D looks for it. The ``tidy3d
        configure`` command stores the API key for you. If you prefer to manage the
        file yourself, place ``config.toml`` in the directory described in
        :doc:`configuration/index` (for example ``~/.config/tidy3d`` on macOS/Linux
        or ``%APPDATA%\\tidy3d`` on Windows).

    .. group-tab:: In the Cloud |:cloud:|

        If you’d rather skip installation and run an example in one of our web-hosted notebooks, `click here <https://tidy3d.simulation.cloud/notebook?file=StartHere.ipynb>`_.



.. To do items:
.. * open simple example in colab with API saved as environment variable and `!pip install tidy3d` in the first line.
.. * toggle between command line - notebook / python instructions in section 1


Quick Example
=============

Start running simulations with just a few lines of code. Run this sample code to simulate a 3D dielectric box in Tidy3D and plot the corresponding field pattern.

.. code-block:: python

   # !pip install tidy3d # if needed, install tidy3d in a notebook by uncommenting this line

   # import the tidy3d package and configure it with your API key
   import numpy as np
   import tidy3d as td
   import tidy3d.web as web
   # web.configure("YOUR API KEY") # if authentication needed, uncomment this line and paste your API key here

   # set up global parameters of simulation ( speed of light / wavelength in micron )
   freq0 = td.C_0 / 0.75

   # create structure - a box centered at 0, 0, 0 with a size of 1.5 micron and permittivity of 2
   square = td.Structure(
       geometry=td.Box(center=(0, 0, 0), size=(1.5, 1.5, 1.5)),
       medium=td.Medium(permittivity=2.0)
   )

   # create source - A point dipole source with frequency freq0 on the left side of the domain
   source = td.PointDipole(
       center=(-1.5, 0, 0),
       source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
       polarization="Ey",
   )

   # create monitor - Measures electromagnetic fields within the entire domain at z=0
   monitor = td.FieldMonitor(
       size=(td.inf, td.inf, 0),
       freqs=[freq0],
       name="fields",
       colocate=True,
   )

   # Initialize simulation - Combine all objects together into a single specification to run
   sim = td.Simulation(
       size=(4, 3, 3),
       grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
       structures=[square],
       sources=[source],
       monitors=[monitor],
       run_time=120/freq0,
   )

   print(f"simulation grid is shaped {sim.grid.num_cells} for {int(np.prod(sim.grid.num_cells)/1e6)} million cells.")

   # run simulation through the cloud and plot the field data computed by the solver and stored in the monitor
   data = td.web.run(sim, task_name="quickstart", path="data/data.hdf5", verbose=True)
   ax = data.plot_field("fields", "Ey", z=0)



This will produce the following plot, which visualizes the electromagnetic fields on the central plane.

.. image:: _static/quickstart_fields.png
   :width: 1200


You can now postprocess simulation data using the same python session, or view the results of this simulation on our web-based `graphical user interface (GUI) <https://tidy3d.simulation.cloud>`_.

.. admonition:: Caching for repeated simulations
   :class: tip

   Repeated runs of the same simulation can reuse solver results by enabling the
   local cache: ``td.config.local_cache.enabled = True``. You may configure the cache directory
   with ``local_cache.directory``. If the number of entries (``local_cache.max_entries``) or the storage size
   (``local_cache.max_size_gb``) is exceeded, cache entries are evicted by least-recently-used (LRU) order.
   You can clear all stored artifacts with ``td.web.cache.clear()``.
   Additionally, there is server-side caching controlled via ``td.config.web.enable_caching``
   (enabled by default). While this can avoid recomputation on the server, it still requires
   upload/download of results which is why we recommend enabling the local cache.

.. `TODO: open example in colab <https://github.com/flexcompute/tidy3d>`_


Our Ecosystem
=======================

``tidy3d`` is part of an interoperable ecosystem of electromagnetic design tools. You can interface with the tools through the python APIs and the graphical user interfaces.

.. grid:: 3
   :gutter: 2

   .. grid-item-card::
      :img-top: _static/img/gui_overview.png
      :link: https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/
      :text-align: center
      :padding: 1
 
      Online Learning Center
      +++
      **Tidy3D Learning Center**

   .. grid-item-card::
      :img-top: _static/img/photonforge_overview.png
      :link: https://docs.flexcompute.com/projects/photonforge/en/latest/
      :text-align: center
      :padding: 1
      
      Photonic Integrated Circuits Design
      +++
      **Photonforge**

   .. grid-item-card::
      :img-top: _static/img/simcloud_overview.png
      :link: https://tidy3d.simulation.cloud
      :text-align: center
      :padding: 1
      
      Access the graphical user interface & account management.
      +++
      **Tidy3D GUI**

Tidy3D + AI
===========

*Ushering in a new era of AI-assisted photonic design*

The `Tidy3D + AI <ai/index.html>`_ ecosystem enables seamless building, visualization, and iteration on electromagnetic simulations. The Tidy3D Extension for Cursor and VS Code automatically detects simulations in Python scripts and notebooks, opens an interactive `3D Viewer <ai/3d_viewer.html>`_ alongside your code, and enables the IDE AI assistant to leverage the integrated `FlexAgent MCP <ai/flex_agent.html>`_ server for an intelligent, physics‑aware assistance experience.

.. image:: _static/img/tidy3d_extension.png
   :alt: Tidy3D Extension
   :height: 300px

Further Information
====================


Join Our Team
-------------

Do you enjoy using ``tidy3d`` or ``photonforge``? Do you want to help us create the best electromagnetic design tools? We are looking for talented developers to join our team.

A good way to get an interview with us is to fix an open issue within our `open-source python client. <https://github.com/flexcompute/tidy3d/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`_ Choose a task where you can feel you can demonstrate your skillset, and get us involved whilst you implement it for feedback and to see how it goes working together. If you don't know where to start, we recommend the issues labelled "good first issue".

We look forward to your contributions!


Github Repositories
--------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Repository
   * - Source Code
     - https://github.com/flexcompute/tidy3d
   * - Example Notebooks
     - https://github.com/flexcompute/tidy3d-notebooks
   * - FAQ Source Code
     - https://github.com/flexcompute/tidy3d-faq

These repositories are a very good way to interact with the relevant tool developers. We encourage you to ask questions or request features through the "Discussions" or "Issues" tabs of each repository accordingly.


Contents
---------

.. toctree::
  :maxdepth: 2

  install
  configuration/index
  lectures/index
  notebooks/docs/index
  faq/docs/index
  api/index
  ai/index
  GUI <https://tidy3d.simulation.cloud/>
  Photonforge <https://docs.flexcompute.com/projects/photonforge/en/latest/>
  extras/index
  development/index
  changelog
  About our Solver <https://www.flexcompute.com/tidy3d/solver/>
