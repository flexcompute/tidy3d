Installation
==============

The Fast Lane
^^^^^^^^^^^^^

Maybe you already have ``tidy3d`` installed in some form. After installing version TODOVERSION, you can use a few terminal commands to set you up on the correct environment and perform common development tasks. Just run in your terminal, :code:`tidy3d develop` to get the latest list of commands.

It does not matter how you have installed ``tidy3d`` before, this will set up the environment you require to reproducibly develop.

Transitioning
--------------

If you are transitioning from the old development flow, to this new one, there are a list of commands you can run to make your life easier and set you up well without a hitch.

.. code::

    # Check and install requirements like pipx, poetry, pandoc
    tidy3d develop configure-dev-environment

This command should run successfully. It will first check if you already have installed the development requirements, and if not, it will run the installation scripts for ``pipx``, ``poetry``, and ask you to install the required version of ``pandoc``. It will also install the development requirements and ``tidy3d`` package in a specific ``poetry`` environment.

If you rather install ``poetry``, ``pipx`` and ``pandoc`` yourself, you can run the following command to verify that your environment conforms to the reproducible development environment:

.. code::

    tidy3d develop verify-dev-environment

You can also run the following on your terminal if you desire to run the terminal commands yourself, rather than have a Python ``subprocess`` implement the required tools installation.

.. code::

    # /bin/bash
    cd tidy3d/
    source scripts/development.sh configure-dev-environment

The Detailed Lane
^^^^^^^^^^^^^^^^^

If you do not have any of the above tools already installed, let's go through the process of setting things up from scatch.


Environment Requirements
------------------------

TODO implement platform specific stuff and add links.
Make sure you have installed ``pipx``:

.. code::

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath


Then install `poetry`:

.. code::

    python3 -m pipx install poetry

After restarting the bash terminal, you should be able to find ``poetry`` in your ``PATH`` if it has been installed correctly:

.. code::

    poetry --version
    poetry # prints all commands

Congratulations! Now you have all the required tools installed.

Packaging Equivalent Functionality
-----------------------------------

This package installation process should be  approximately equivalent to the previous ``setup.py`` installation flow. Independent of the ``poetry`` development flow, it is possible to run any of the following commands in any particular virtual environment you have configured:

.. code::

    pip install tidy3d[dev]
    pip install tidy3d[docs]
    pip install tidy3d[web]
    ...
    pip install tidy3d[jax]

All these options can be found inside the ``pyproject.toml`` ``tool.poetry.extras`` section. Each has a corresponding list of dependencies whose versions are defined on the ``tool.poetry.dependencies`` section of the file.

Useful Tool Resources
=======================

.. TODO add links here about poetry etc.