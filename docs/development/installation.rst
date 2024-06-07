Development Environment Installation
=====================================

The Fast Lane
--------------

Maybe you already have ``tidy3d`` installed in some form. After installing version ``tidy3d>=2.6``, you can use a few terminal commands to set you up on the correct environment and perform common development tasks. Just run in your terminal, :code:`tidy3d develop` to get the latest list of commands.

It does not matter how you have installed ``tidy3d`` before as long as you have any form of ``tidy3d>=2.6`` in your environment. This can help you transition from a standard user installation to a development environment installation.

Quick Start
^^^^^^^^^^^^^

Instructions for anyone who wants to migrate to the development flow from a version before 2.6:

For ubuntu:

.. code-block:: bash

    git clone https://github.com/flexcompute/tidy3d.git
    cd tidy3d
    # Make sure you're in a branch > pre/2.6 and you can `import tidy3d` in python
    pip install -e . # or whatever local installation works for you
    tidy3d develop # Read all the new development helper commands
    # tidy3d develop uninstall-dev-envrionment # in case you need to reset your environment
    tidy3d develop install-dev-environment # install all requirements that you don't have and verify the exisiting ones
    poetry run tidy3d develop verify-dev-environment # reproducibly verify development envrionment
    # poetry run tidy3d develop build-docs # eg. reproducibly build documentation

Now you can run the following ``tidy3d`` cli commands to test them.


Automatic Environment Installation *Beta*
""""""""""""""""""""""""""""""""""""""""""""""

If you are transitioning from the old development flow, to this new one, there are a list of commands you can run to make your life easier and set you up well:

.. code::

    # Automatically check and install requirements like pipx, poetry, pandoc
    tidy3d develop install-dev-environment

Note that this is just a automatic script implementation of the :ref:`The Detailed Lane` instructions. It is intended to help you and raise warnings with suggestions of how to fix an environment setup issue. You do not have to use this helper function and can just follow the instructions in  :ref:`The Detailed Lane`. All commands are echo-ed in the terminal so you will be able to observe and reproduce what is failing if you desire.

The way this command works is dependent on the operating system you are running. There are some prerequisites for each platform, but the command line tool will help you identify and install the tools it requires. You should rerun the command after you have installed any prerequisite as it will just progress with the rest of the tools installation. If you already have the tool installed, it will verify that it conforms to the supported versions.

This command will first check if you already have installed the development requirements, and if not, it will run the installation scripts for ``pipx``, ``poetry``, and ask you to install the required version of ``pandoc``. It will also install the development requirements and ``tidy3d`` package in a specific ``poetry`` environment.

Environment Verification
""""""""""""""""""""""""""""

If you rather install ``poetry``, ``pipx`` and ``pandoc`` yourself, you can run the following command to verify that your environment conforms to the reproducible development environment which would be equivalent to the one installed automatically above and described in :ref:`The Detailed Lane`.

.. code::

    tidy3d develop verify-dev-environment


The Detailed Lane
------------------

If you do not have any of the above tools already installed and want to install them manually, let's go through the process of setting things up from scratch:


Environment Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have installed ``pipx``. We provide common installation flows below:

.. tabs::

   .. group-tab:: Ubuntu 22.04

        This installation flow requires a ``python3`` installation. Depending how you have installed ``python3``, you may have to edit this command to run on your target installation. Further instructions by ``pipx`` `here <https://github.com/pypa/pipx?tab=readme-ov-file#on-linux>`_

        .. code-block:: bash

            python3 -m pip install --user pipx
            python3 -m pipx ensurepath

   .. group-tab:: macOS

        This installation flow uses `homebrew <https://brew.sh/>`_. Further instructions by ``pipx`` `here <https://github.com/pypa/pipx?tab=readme-ov-file#on-macos>`_

        .. code-block:: bash

            brew install pipx
            pipx ensurepath

   .. group-tab:: Windows

        This installation flow uses `scoop <https://scoop.sh/>`_. Further instructions by ``pipx`` `here <https://github.com/pypa/pipx?tab=readme-ov-file#on-windows>`_

        .. code-block:: bash

            scoop install pipx
            pipx ensurepath


Then install ``poetry``:

.. tabs::

   .. group-tab:: Ubuntu 22.04

        Further instructions in the `poetry installation instructions <https://python-poetry.org/docs/#installation>`_

        .. code-block:: bash

            python3 -m pipx install poetry

   .. group-tab:: macOS

        Further instructions in the `poetry installation instructions <https://python-poetry.org/docs/#installation>`_

        .. code-block:: bash

            pipx install poetry

   .. group-tab:: Windows

        Further instructions in the `poetry installation instructions <https://python-poetry.org/docs/#installation>`_

        .. code-block:: bash

            pipx install poetry


After restarting the bash terminal, you should be able to find ``poetry`` in your ``PATH`` if it has been installed correctly:

.. code::

    poetry --version
    poetry # prints all commands


If you want to locally build documentation, then it is required to install ``pandoc<3``.

.. tabs::

   .. group-tab:: Ubuntu 22.04

        Further instructions in the `pandoc installation instructions <https://pandoc.org/installing.html#linux>`_. Note you will need permissions to do this.

        .. code-block:: bash

            sudo apt-get update
            sudo apt-get install pandoc

   .. group-tab:: macOS

        Further instructions in the `poetry installation instructions <https://pandoc.org/installing.html#macos>`_

        .. code-block:: bash

            brew install pandoc@2.9

   .. group-tab:: Windows

        This installation flow uses `Chocolatey <https://chocolatey.org/>`_. Further instructions in the `poetry installation instructions <https://pandoc.org/installing.html#windows>`_

        .. code-block:: bash

           choco install pandoc --version="2.9"

Now you need to install the package in the reproducible poetry environment in development mode:

.. code::

    poetry install -E dev

Congratulations! Now you have all the required tools installed, you can now use all the ``poetry run tidy3d develop`` commands reproducibly.

If you want to contribute to the project, read the following section:


More Contribution Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to contribute to the development of ``tidy3d``, you can follow the instructions below to set up your development environment. This will allow you to run the tests, build the documentation, and run the examples. Another thing you need to do before committing to the project is to install the pre-commit hooks. This will ensure that your code is formatted correctly and passes the tests before you commit it. To do this, run the following command:

.. code::

    poetry run pre-commit install

This will run a few file checks on your code before you commit it. After this whenever you commit, the pre-commit hooks will run automatically. If any of the checks fail, you will have to fix the issues before you can commit. If for some reason, it's a check you want to waive, you can follow the instructions of the tool to automatically waive them or you can run the following command to skip the checks **only on minimal circumstances**:

.. code::

    git commit --no-verify

You can also run the checks manually on all files by running the following command:

.. code::

    poetry run pre-commit run --all-files


Packaging Equivalent Functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package installation process should be  approximately equivalent to the previous ``setup.py`` installation flow. Independent of the ``poetry`` development flow, it is possible to run any of the following commands in any particular virtual environment you have configured:

.. code::

    pip install tidy3d[dev]
    pip install tidy3d[docs]
    pip install tidy3d[web]
    ...
    pip install tidy3d[jax]

All these options can be found inside the ``pyproject.toml`` ``tool.poetry.extras`` section. Each has a corresponding list of dependencies whose versions are defined on the ``tool.poetry.dependencies`` section of the file.

