Develop
=================

Welecome to the ``tidy3d`` developer's guide! These are just some recommendations I've compiled, but we can change anything as we think might help the development more.

Project Structure
-----------------

As of TODOVERSION, the ``tidy3d`` frontend has been restructured to improve the development cycle. The project follows the following structure, which is derived from some recommended Python project architecture guides https://docs.python-guide.org/writing/structure/ . This is a handy structure because many tools, such as ``sphinx``, integrate quite well with this type of project layout.

.. code::

    docs/
        # sphinx rst files
        ...
        notebooks/
            # Git submodule repository
    tests/
        # pytest source and docs
        # pytest notebooks
    scripts/
        # useful handy scripts
    tidy3d/
        # python source code
    ...
    pyproject.toml # python packaging
    poetry.lock # environment management

It is important to note the new tools we are using to manage our development environment and workflow.

- ``poetry``
- ``pipx``


The Fast Lane
--------------

Maybe you already have ``tidy3d`` installed in some form. After TODOVERSION, you can use a few terminal commands to set you up on the correct environment and perform common development tasks. Just run in your installation, :code:`tidy3d develop` to get the latest list of commands.

It does not matter how you have installed ``tidy3d`` before, this will set up the development environment you require to continue the development process.

To the New Development Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are transitioning from the new development flow, to this new one, there are a list of commands you can run to make your life easier.

.. code::

    # Check and install requirements like pipx, poetry, pandoc
    tidy3d develop configure-dev-environment

This command should run successfully. It will first check if you already have installed the development requirements, and if not, it will run the installation scripts for ``pipx``, ``poetry``, and ask you to install the required version of ``pandoc``. It will also install the development requirements and ``tidy3d`` package in a specific ``poetry`` environment.



The Detailed Lane
------------------

If you do not have any of the above tools already installed, let's go through the process of setting things up from scatch.


Environment Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

TODO implement platform specific stuff and add links.
Make sure you have installed ``pipx``:

.. code::

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath


Then install `poetry`:

.. code::

    python3 -m pipx install poetry

After restarting the bash terminal, you should be able to find `poetry` in your `PATH` if it has been installed correctly:

.. code::

    poetry --version
    poetry # prints all commands


Developing ``tidy3d`` with ``poetry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``poetry`` is an incredibly powerful tool for reproducible package development envrionments and dependency management.

If you are developing ``tidy3d``, we recommend you work within the configured ``poetry`` environment defined by ``poetry.lock``. The way to install this envrionment is simple:

.. code::

    cd tidy3d/
    poetry install -E dev

This function will install the package with all the development dependencies automatically. This means you should be able to run any functionality that is possible with ``tidy3d`` reprodicibly.

It is important to note the function above is equivalent to ``pip install tidy3d[dev]``, but by using ``poetry`` there is a guarantee of using the reproducible locked environment.


Get Started
------------

There are a range of handy development functions that you might want to use to streamline your development experience.
