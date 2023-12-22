Installation
==============

The Fast Lane
^^^^^^^^^^^^^

Maybe you already have ``tidy3d`` installed in some form. After installing version ``tidy3d>=2.6``, you can use a few terminal commands to set you up on the correct environment and perform common development tasks. Just run in your terminal, :code:`tidy3d develop` to get the latest list of commands.

It does not matter how you have installed ``tidy3d`` before as long as you have any form of `tidy3d>=2.6`` in your environment. This can help you transition from a standard user installation to a development environment installation.

Beta instructions for verification (REMOVE pre 2.6 release)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Instructions for anyone who wants to test the new development flow before it gets included as part of the pre-release:

For ubuntu:

.. code-block:: bash

    git clone https://github.com/flexcompute/tidy3d.git
    cd tidy3d
    git fetch origin repo_merge_no_history
    git checkout repo_merge_no_history
    # Create and activate a virtual environment here based on your python installation
    python3 -m pip install -e . # Follow standard pip development install
    python3 -m tidy3d develop # list all new development commands
    python3 -m tidy3d develop configure-dev-environment

Now you can run the following ``tidy3d`` cli commands to test them.


Automatic Environment Installation (Beta)
'''''''''''''''''''''''''''''''''''''''''''''

If you are transitioning from the old development flow, to this new one, there are a list of commands you can run to make your life easier and set you up well:

.. code::

    # Automatically check and install requirements like pipx, poetry, pandoc
    tidy3d develop configure-dev-environment

Note that this is just a automatic script implementation of the `The Detailed Lane`_ instructions. It is intended to help you and raise warnings with suggestions of how to fix an environment setup issue. You do not have to use this helper function and can just follow the instructions in  `The Detailed Lane`_. All commands are echo-ed in the terminal so you will be able to observe and reproduce what is failing if you desire.

The way this command works is dependent on the operating system you are running. There are some prerequisites for each platform, but the command line tool will help you identify and install the tools it requires. You should rerun the command after you have installed any prerequisite as it will just progress with the rest of the tools installation. If you already have the tool installed, it will verify that it conforms to the supported versions.

This command will first check if you already have installed the development requirements, and if not, it will run the installation scripts for ``pipx``, ``poetry``, and ask you to install the required version of ``pandoc``. It will also install the development requirements and ``tidy3d`` package in a specific ``poetry`` environment.

Environment Verification
''''''''''''''''''''''''

If you rather install ``poetry``, ``pipx`` and ``pandoc`` yourself, you can run the following command to verify that your environment conforms to the reproducible development environment which would be equivalent to the one installed automatically above and described in `The Detailed Lane`_.

.. code::

    tidy3d develop verify-dev-environment


.. _detailed_lane:

The Detailed Lane
^^^^^^^^^^^^^^^^^

If you do not have any of the above tools already installed and want to install them manually, let's go through the process of setting things up from scratch:


Environment Requirements
''''''''''''''''''''''''''

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
            sudo apt-get install pandoc="2.9"

   .. group-tab:: macOS

        Further instructions in the `poetry installation instructions <https://pandoc.org/installing.html#macos>`_

        .. code-block:: bash

            brew install pandoc@2.9

   .. group-tab:: Windows

        This installation flow uses `Chocolatey <https://chocolatey.org/>`_. Further instructions in the `poetry installation instructions <https://pandoc.org/installing.html#windows>`_

        .. code-block:: bash

           choco install pandoc --version="2.9"

Congratulations! Now you have all the required tools installed, you can now use all the `poetry run tidy3d develop` commands reproducibly.


Packaging Equivalent Functionality
'''''''''''''''''''''''''''''''''''

This package installation process should be  approximately equivalent to the previous ``setup.py`` installation flow. Independent of the ``poetry`` development flow, it is possible to run any of the following commands in any particular virtual environment you have configured:

.. code::

    pip install tidy3d[dev]
    pip install tidy3d[docs]
    pip install tidy3d[web]
    ...
    pip install tidy3d[jax]

All these options can be found inside the ``pyproject.toml`` ``tool.poetry.extras`` section. Each has a corresponding list of dependencies whose versions are defined on the ``tool.poetry.dependencies`` section of the file.