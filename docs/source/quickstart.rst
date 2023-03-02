**********
Start Here
**********

Welcome to Tidy3d
=================

This page will get you set up with Tidy3D and running a simple example.

.. note:: This is the documentation for the beta release of Tidy3D.

Code Repositories
-----------------

We host several examples and tutorials at our `documentation repository <https://github.com/flexcompute-readthedocs/tidy3d-docs>`_ and its `notebook section <https://github.com/flexcompute-readthedocs/tidy3d-docs/tree/readthedocs/docs/source/notebooks>`_.

You can find our front end python code in its entirety at `this github repository <https://github.com/flexcompute/tidy3d>`_.  This is also a good place to ask questions or request features through the "Discussions" tab.

Getting Started
===============

Before using Tidy3D, you must first `sign up <https://client.simulation.cloud/register-waiting>`_ for a user account.

Signing up also grants you access to our browser-based `interface <https://tidy3d.simulation.cloud/account>`_ for managing simulations.

.. Quick Start (Binder Notebook)
.. -----------------------------

.. `Click this text to get started running a Tidy3D simulation right away without any installation or software setup. <https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks%2FStartHere.ipynb>`_

.. Once there, to run the full example, select "Run -> Run All Cells".  Or you can click through the code blocks by pressing the "play" icon.

.. You will first be prompted to log in using the email and password you used for your user account.

.. Then the notebook will create a simulation and upload it to our server, where it will run for a few minutes before downloading the results and plotting the field patterns.

.. image:: _static/quickstart_fields.png
..    :width: 600

.. To play around with the simulation parameters, you can edit the notebook directly and re-run.

Installation of Tidy3D Python API
---------------------------------

If you wish to run the Tidy3D Python API locally, follow the instructions below.

Create a new Python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have Python installed on your computer, it is possible that some packages in your current environment could have version conflicts with Tidy3D. To avoid this, we strongly recommend that you create a clean Python virtual environment to install Tidy3D.

We recommend using the Conda package management system to manage your Python virtual environment as well as installing Tidy3D. You can install Conda conveniently via `Anaconda <https://www.anaconda.com/>`__.

After you install Anaconda, open the Anaconda Prompt and enter

.. code-block:: bash

    conda create –n tidy3d_env python==3.10

to create a new environment. ``tidy3d_env`` is the name for the new environment, which can be changed to your personal preference. Python version 3.10 and its associated packages will also be installed in this new environment by adding ``python==3.10``. After the environment is created, we need to activate it by

.. code-block:: bash

    conda activate tidy3d_env

You are now ready to install Tidy3D in your new environment, which will be discussed in the next section. More information about Conda environment management tools can be found `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

Install Tidy3D
^^^^^^^^^^^^^^

Tidy3D and its dependencies can be installed from the command line via ``pip``, which is installed with Python when the new environment is created. Simply run

.. code-block:: bash

    pip install tidy3d

and the latest version of Tidy3D will be installed in this environment. To test whether the installation was successful you can run

.. code-block:: bash

    python -c "import tidy3d as td; print(td.__version__)"

If the installation is successful, you should see the client version of Tidy3D being displayed. Now you can open your favorite Python IDE and start creating Tidy3D simulations!

To get the "pre-release" version of tidy3d with the newest features, one may specify the version as follows:

.. code-block:: bash

    pip install tidy3d==1.9.0rc2

The corresponding documentation is marked as "latest" and is available `here <https://docs.flexcompute.com/projects/tidy3d/en/latest/>`__.

Additional Configuration for Python IDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your Python IDE of choice is not natively included in Anaconda, you need to configure the environment in your IDE manually. We will use the popular PyCharm IDE as an example. In PyCharm, go to File – Settings – Project – Python Interpreter. Click “Add Interpreter” and choose “Conda Environment”. Then click the “…” icon to choose the path for the Conda environment with Tidy3D installed. The path usually looks like

``C:\Users\xxx\Anaconda3\envs\tidy3d_env\tidy3d_env\python.exe``.

After clicking “OK”, your PyCharm project should be using the correct Conda environment. You can import Tidy3D using the usual

.. code-block:: bash

    import tidy3d as td

in your codes.

.. note:: Please pay attention to any warning or error messages during the installation process as your system configuration might be different. If you are experiencing difficulty in the installation, please reach out to us for help. We would gladly assist you for Tidy3D installation.

Next Steps
==========

That should get you started!  

To see some other examples of Tidy3D being used in large scale photonics simulations, see `Examples <./examples.html>`_.

To learn more about the many features of Tidy3D, check out our `Feature Walkthrough <./notebooks/Simulation.html>`_.

Or, if you're interested in the API documentation, see `API Reference <./api.html>`_.
