Using `poetry` for package management
--------------------------------------

What is Poetry
^^^^^^^^^^^^^^^^^^^^^^^^^^

`Poetry <https://python-poetry.org>`_ is a package management tool for Python.

Among other things, it provides a nice way to:

- Manage dependencies
- Publish packages
- Set up and use virtual environments

Effectively, it is a command line utility (similar to ``pip``) that is a bit more convenient and allows more customization.

Why do we want to use it
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. To improve our dependency management, which is used to be all over the place. We have several ``requirements.txt`` files that get imported into ``setup.py`` and parsed depending on the extra arguments passed to ``pip install``. ``Poetry`` handles this much more elegantly through a ``pyproject.toml`` file that defines the dependency configuration very explicitly in a simple data format.
2. Reproducible development virtual environments means that everyone is using the exact same dependencies, without conflicts. This also improves our packaging and release flow.

How to install it?
^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide custom installation instructions and an installation script on TODO ADD LINK SECTION. However, you can read more information here: see the poetry documentation for a guide to `installation <https://python-poetry.org/docs/#installation>`_ and `basic use <https://python-poetry.org/docs/basic-usage/>`_.


Usage Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

To add poetry to a project
""""""""""""""""""""""""""""

To initialize a new basic project with poetry configured, run:

.. code-block:: bash

   poetry new poetry-demo

To add poetry to an existing project, ``cd`` to the project directory and run:

.. code-block:: bash

   poetry init

Configuring dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

The dependency configuration is in the editable file called ``pyproject.toml``. Here you can specify whatever dependencies you want in your project, their versions, and even different levels of dependencies (e.g., ``dev``).

To add a dependency to the project (e.g., ``numpy``), run:

.. code-block:: bash

   poetry add numpy

You can then verify that it was added to the ``tool.poetry.dependencies`` section of ``pyproject.toml``.

For many more options on defining dependencies, see `here <https://python-poetry.org/docs/dependency-specification/>`_.

Virtual environments
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that the project has had poetry configured and the correct dependencies are specified, we can use poetry to run our scripts/shell commands from a virtual environment without much effort. There are a few ways to do this:

**Poetry run**: One way is to precede any shell command you’d normally run with ``poetry run``. For example, if you want to run ``python tidy_script.py`` from the virtual environment set up by poetry, you’d do:

.. code-block:: bash

   poetry run python tidy3d_script.py

**Poetry shell**:

If you want to open up a shell session with the environment activated, you can run:

.. code-block:: bash

   poetry shell

And then run your commands. To return to the original shell, run ``exit``.

There are many more advanced options explained `here <https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment>`_.

Publishing Package
^^^^^^^^^^^^^^^^^^^^^^^^^^

To upload the package to PyPI:

.. code-block:: bash

   poetry build

   poetry publish

Note that some `configuration <https://python-poetry.org/docs/cli/#publish>`_ must be set up before this would work properly.
