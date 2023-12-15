*************************
Installation
*************************

Welcome to Tidy3D
=================

This page will get you set up with Tidy3D and running a simple example.

Getting Started
===============

Before using Tidy3D, you must first `sign up <https://tidy3d.simulation.cloud/signup>`_ for a user account.

By signing up for a free account, you can obtain an API key `here <https://tidy3d.simulation.cloud/account?tab=apikey>`_. You can also `manage your simulation jobs <https://tidy3d.simulation.cloud/folders>`_ and access `graphic user interface <https://tidy3d.simulation.cloud/workbench?taskId=pa-94c49911-132d-48bc-8ec0-f0a4e55140a3>`_ if needed.

Installation of Tidy3D Python API
---------------------------------

If you wish to install the Tidy3D Python API locally, the following instructions should work for most users.

.. code-block:: bash

    pip install --user tidy3d
    tidy3d configure --apikey=XXX

Where ``XXX`` is your API key, which can be copied from your `account page <https://tidy3d.simulation.cloud/account>`_ in the web interface.

In a hosted jupyter notebook environment (eg google colab), it may be more convenient to install and configure via the following lines at the top of the notebook.

.. code-block:: bash

    !pip install tidy3d
    import tidy3d.web as web
    web.configure("XXX")

To test that the configuration is working from the python browser, you can perform:

.. code-block:: bash

    import tidy3d.web as web # if needed
    web.test()

If those commands did not work, there are advanced installation instructions below, which should help solve the issue.

Advanced Installation Instructions
----------------------------------

Some users or systems may require a more specialized installation, which we will cover below.

Create a new Python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have Python installed on your computer, it is possible that some packages in your current environment could have version conflicts with Tidy3D. To avoid this, we strongly recommend that you create a clean Python virtual environment to install Tidy3D.

We recommend using the Conda package management system to manage your Python virtual environment as well as installing Tidy3D. You can install Conda conveniently via `Anaconda <https://www.anaconda.com/>`__.

After you install Anaconda, open the Anaconda Prompt and enter

.. code-block:: bash

    conda create –n tidy3d_env python==3.10

to create a new environment. ``tidy3d_env`` is the name of the new environment, which can be changed to your personal preference. Python version 3.10 and its associated packages will also be installed in this new environment by adding ``python==3.10``. After the environment is created, we need to activate it by

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

To get a specific version eg. ``x.y.z`` of tidy3d, including the "pre-release" versions, one may specify the version as follows:

.. code-block:: bash

    pip install tidy3d==x.y.z

The documentation for the most recent release is marked as "latest" and is available `here <https://docs.flexcompute.com/projects/tidy3d/en/latest/>`__. The documentation page also allows one to select the state of the docs based on version by toggling the dropdown in the bottom left corner.

Linking Regiestration
^^^^^^^^^^^^^^^^^^^^^

Now that tidy3d is installed on your python distribution, we need to link it with your account. First you should copy your "API key" from your account page on the `web interface <https://tidy3d.simulation.cloud/account>`_.  To find it, sign in and navigate to the account page by clicking the "Account Center" icon on the left-hand side. Then, find the "API key" tab on the right hand side of the menu and copy your API key from there.

We'll refer to that key as ``XXX`` in the following instructions.

The simplest way to link your account is by typing 

.. code-block:: bash

    tidy3d configure

and pasting the API key when prompted. Note that one can also specify the API key directly in the configure command as

.. code-block:: bash

    tidy3d configure --apikey=XXX

Note: Windows users will need to peform a slighlty different step to link the registration. From the anaconda prompt where tidy3d was pip installed, the following commands should be run instead

.. code-block:: bash

    pip install pipx
    pipx run tidy3d configure --apikey=XXX

Alternatively, the API key can be set up using the evironment variable ``SIMCLOUD_APIKEY`` as:

.. code-block:: bash

    export SIMCLOUD_APIKEY="XXX"

Finally, one may manually set the API key directly in the configuration file where Tidy3D looks for it.

The API key must be in a file called ``.tidy3d/config`` located in your home directory, with the following contents

.. code-block:: bash

    apikey = "XXX"


You can manually set up your file like this, or do it through the command line line:

.. code-block:: bash

    echo 'apikey = "XXX"' > ~/.tidy3d/config

Note the quotes around `XXX`.

Note that Windows users will most likely need to place the ``.tidy3d/config`` file in their ``C:\Users\username\`` directory (where ``username`` is your username).


Additional Configuration for Python IDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your Python IDE of choice is not natively included in Anaconda, you need to configure the environment in your IDE manually. We will use the popular PyCharm IDE as an example. In PyCharm, go to File – Settings – Project – Python Interpreter. Click “Add Interpreter” and choose “Conda Environment”. Then click the “…” icon to choose the path for the Conda environment with Tidy3D installed. The path usually looks like

``C:\Users\xxx\Anaconda3\envs\tidy3d_env\tidy3d_env\python.exe``.

After clicking “OK”, your PyCharm project should be using the correct Conda environment. You can import Tidy3D using the usual

.. code-block:: bash

    import tidy3d as td

in your code.

.. note:: Please pay attention to any warning or error messages during the installation process as your system configuration might be different. If you are experiencing difficulty in the installation, please reach out to us for help. We would gladly assist you for Tidy3D installation.

Code Repositories
^^^^^^^^^^^^^^^^^

We host all of the several examples and tutorials from this documentation in the `notebook section <https://github.com/flexcompute-readthedocs/tidy3d-docs/tree/readthedocs/docs/source/notebooks>`_ of our `documentation github repository <https://github.com/flexcompute-readthedocs/tidy3d-docs>`_.

You can find our front end python code in its entirety at `its github repository <https://github.com/flexcompute/tidy3d>`_.  This is also a good place to ask questions or request features through the "Discussions" or "Issues" tabs.

Next Steps
==========

That should get you started!  

To see some other examples of Tidy3D being used in large scale photonics simulations, see `Examples <./examples.html>`_.

To learn more about the many features of Tidy3D, check out our `Feature Walkthrough <./notebooks/Simulation.html>`_.

Or, if you're interested in the API documentation, see `API Reference <./api.html>`_.