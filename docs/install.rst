*************************
Installation |:wave:|
*************************

This page will get you set up with Tidy3D and running a simple example.

Recommended: Tidy3D Agent Plugin |:robot:|
===========================================

If you work with an AI coding assistant (Claude Code, Codex, Cursor, GitHub Copilot, or VS Code),
we recommend installing the Tidy3D plugin (``tidy3d@flexcompute``). It equips your assistant with
Tidy3D skills and a documentation-search MCP server so it can help you build, debug, and review
Tidy3D simulations using up-to-date Flexcompute guidance instead of pasting docs into every chat.

A single command sets it up:

.. tabs::

    .. group-tab:: macOS / Linux |:computer:|

        .. code-block:: bash

            curl -LsSf https://raw.githubusercontent.com/flexcompute/plugin-marketplace/main/install.sh | bash

    .. group-tab:: Windows PowerShell |:cloud:|

        .. code-block:: powershell

            powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/flexcompute/plugin-marketplace/main/install.ps1 | iex"

The installer asks which AI coding tool you use. It auto-configures the marketplace and plugin for
Claude Code, Codex, and GitHub Copilot CLI. For Cursor it prints an ``/add-plugin`` command to run
from Cursor Agent chat, and VS Code is a separate preview flow you enable through ``chat.plugins.*``
settings. For Claude Code you can also add the plugin directly:

.. code-block:: bash

    claude plugin marketplace add flexcompute/plugin-marketplace
    claude plugin install tidy3d@flexcompute

The plugin equips your AI assistant; you still install the Tidy3D Python API with
``pip install tidy3d`` as described below. See the `plugin marketplace
<https://github.com/flexcompute/plugin-marketplace>`_ for setup on other tools and more details.

For full details on the agent plugin and the FlexAgent MCP server it configures, including
setup for Claude Code, Codex, and other MCP clients, see :doc:`ai/flex_agent`.

Getting Started
===============

Before using Tidy3D, you must first `sign up <https://tidy3d.simulation.cloud/signup>`_ for a user account.

By signing up for a free account, you can obtain an API key `here <https://tidy3d.simulation.cloud/account?tab=apikey>`_. You can also `manage your simulation jobs <https://tidy3d.simulation.cloud/folders>`_ and access `graphic user interface (GUI) <https://tidy3d.simulation.cloud/workbench?taskId=pa-94c49911-132d-48bc-8ec0-f0a4e55140a3>`_ if needed.

Managing API Keys
------------------

Quick Configuration
~~~~~~~~~~~~~~~~~~~

.. tabs::

    .. group-tab:: Local Installation |:computer:|

        If you wish to install the Tidy3D Python API locally, the following instructions should work for most users.

        .. code-block:: bash

            pip install --user tidy3d
            tidy3d configure --apikey=XXX

    .. group-tab:: Any Hosted Environment |:cloud:|

        Where ``XXX`` is your API key, which can be copied from your `account page <https://tidy3d.simulation.cloud/account>`_ in the web interface.

        In a hosted jupyter notebook environment (eg google colab), it may be more convenient to install and configure via the following lines at the top of the notebook.

        .. code-block:: bash

            pip install tidy3d

        .. code-block:: python

            import tidy3d.web as web
            web.configure("XXX")

Testing the API Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test that the configuration is working from the python browser, you can perform:

.. code-block:: python

    import tidy3d.web as web # if needed
    web.test()

If those commands did not work, there are advanced installation instructions below, which should help solve the issue.


Advanced Key Management
~~~~~~~~~~~~~~~~~~~~~~~~

Now that tidy3d is installed on your python distribution, we need to link it with your account. First you should copy your "API key" from your account page on the `web interface <https://tidy3d.simulation.cloud/account>`_.  To find it, sign in and navigate to the account page by clicking the "Account Center" icon on the left-hand side. Then, find the "API key" tab on the right hand side of the menu and copy your API key from there.

We'll refer to that key as ``XXX`` in the following instructions.

The simplest way to link your account is by typing

.. code-block:: python

    tidy3d configure

and pasting the API key when prompted. Note that one can also specify the API key directly in the configure command as

.. code-block:: python

    tidy3d configure --apikey=XXX

Alternatively, the API key can be set up using the environment variable ``SIMCLOUD_APIKEY`` as:

.. code-block:: python

    export SIMCLOUD_APIKEY="XXX"

Finally, one may manually set the API key directly in the configuration file
where Tidy3D looks for it. The path and file format differ slightly between
platforms; see :doc:`configuration/index` for the up-to-date layout.



Package Version Management
----------------------------

Tidy3D and its dependencies can be installed from the command line via ``pip``, which is installed with Python when the new environment is created. Simply run

.. code-block:: bash

    pip install tidy3d

and the latest version of Tidy3D will be installed in this environment. To test whether the installation was successful you can run

.. code-block:: bash

    python -c "import tidy3d as td; print(td.__version__)"

If the installation is successful, you should see the client version of Tidy3D being displayed. Now you can open your favorite Python IDE and start creating Tidy3D simulations!

If anything goes wrong during setup or when running a simulation, run
``tidy3d troubleshoot report`` to produce a paste-ready diagnostic bundle for
support. See :doc:`troubleshoot` for details.

To get a specific version eg. ``x.y.z`` of tidy3d, including the "pre-release" versions, one may specify the version as follows:

.. code-block:: bash

    pip install tidy3d==x.y.z

The documentation for the most recent release is marked as "latest" and is available `here <https://docs.flexcompute.com/projects/tidy3d/en/latest/>`__. The documentation page also includes a version switcher so you can jump between ``latest``, ``stable``, and published release builds.


Advanced Installation Instructions
==================================

Some users or systems may require a more specialized installation, which we will cover below.

.. tabs::

    .. group-tab:: Conda/Mamba |:snake:|

        If you already have Python installed on your computer, it is possible that some packages in your current environment could have version conflicts with Tidy3D. To avoid this, we strongly recommend that you create a clean Python virtual environment to install Tidy3D.

        We recommend using the Mamba package management system to manage your Python virtual environment as well as installing Tidy3D. You can install Mamba conveniently following `these instructions <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`__.

        After you install Anaconda, open the Anaconda Prompt and enter

        .. code-block:: bash

            mamba create –n tidy3d_env python==3.10 -c conda-forge

        to create a new environment. ``tidy3d_env`` is the name of the new environment, which can be changed to your personal preference. Python version 3.10 and its associated packages will also be installed in this new environment by adding ``python==3.10``. After the environment is created, we need to activate it by

        .. code-block:: bash

            mamba activate tidy3d_env

        You are now ready to install Tidy3D in your new environment, which will be discussed in the next section. More information about Conda environment management tools can be found `here <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`__.

    .. group-tab:: PyCharm/IDEs |:four_leaf_clover:|

        If your Python IDE of choice is not natively included in Anaconda, you need to configure the environment in your IDE manually. We will use the popular PyCharm IDE as an example. In PyCharm, go to File – Settings – Project – Python Interpreter. Click “Add Interpreter” and choose “Conda Environment”. Then click the “…” icon to choose the path for the Conda environment with Tidy3D installed. The path usually looks like

        ``C:\Users\xxx\Anaconda3\envs\tidy3d_env\tidy3d_env\python.exe``.

        After clicking “OK”, your PyCharm project should be using the correct Conda environment. You can import Tidy3D using the usual

        .. code-block:: python

            import tidy3d as td

        in your code.

        .. note:: Please pay attention to any warning or error messages during the installation process as your system configuration might be different. If you are experiencing difficulty in the installation, please reach out to us for help. We would gladly assist you for Tidy3D installation.

    .. group-tab:: uv |:musical_note:|

        Install ``tidy3d`` within a reproducible environment using the ``uv.lock`` installation and the ``uv`` toolchain.

Optional Dependencies
=====================

Tidy3D provides several optional dependency groups that you can install based on your specific needs.

Installing Optional Core Dependencies
-------------------------------------

Tidy3D has several optional dependencies that provide additional functionality. You can install these using the following syntax:

.. code-block:: bash

    pip install "tidy3d[dependency_group]"

Where ``dependency_group`` is one of the following:

- ``gdstk``: Adds support for GDS export using `gdstk <https://github.com/heitzmann/gdstk>`_.
- ``trimesh``: Support for more complex mesh handling and manipulation.
- ``vtk``: Support for working with unstructured data.
- ``heatcharge``: Additional dependencies for heat & charge solvers.

For example, to install Tidy3D with trimesh support:

.. code-block:: bash

    pip install "tidy3d[trimesh]"

Installing Plugin Dependencies
------------------------------

Tidy3D also offers plugins that require additional dependencies:

- ``design``: Design space exploration and optimization.
- ``pytorch``: A PyTorch wrapper for objective functions defined using autograd.

For example, to install the design plugin dependencies:

.. code-block:: bash

    pip install "tidy3d[design]"

Multiple dependency groups can be installed simultaneously:

.. code-block:: bash

    pip install "tidy3d[design,trimesh]"

Extras Plugin
----------------------

An optional plugin providing additional local functionality,
including a more accurate local mode solver:

.. code-block:: bash

    pip install "tidy3d[extras]"

An API key is needed to use the extras plugin.
For more information on the extras plugin, see `Extras Plugin <./extras/index.html>`_.

.. important::

   ``tidy3d-extras`` is **not compatible with Conda environments**.
   Please use a standard Python virtual environment (e.g., ``venv`` or ``virtualenv``) for installation.

Developer Installation
----------------------

For developers and those who want to install all optional dependencies:

.. code-block:: bash

    pip install "tidy3d[dev]"

Next Steps
==========

That should get you started!  

To see some other examples of Tidy3D being used in large scale photonics simulations, see `Examples <./notebooks/docs/index.html>`_.

To learn more about the many features of Tidy3D, check out our `Feature Walkthrough <./notebooks/Simulation.html>`_.

Or, if you're interested in the API documentation, see `API Reference <./api/index.html>`_.
