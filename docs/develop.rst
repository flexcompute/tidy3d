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

Maybe you already have ``tidy3d`` installed in some form. After TODOVERSION, you can use a few terminal commands to set you up and perform common development tasks. Just run in your installation, :code:`tidy3d develop` to get the latest list of commands.

It does not matter how you have installed ``tidy3d`` before, this will set up the development environment you require to continue the development process.

To the New Development Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are transitioning from the new development flow, to this new one, there are a list of commands you can run to make your life easier.

.. code::

    # Check and install requirements like pipx, poetry, pandoc
    tidy3d develop configure-dev-environment

This command should run successfully. It will first check if you already have installed the development requirements, and if not, it will run the installation scripts for ``pipx``, ``poetry``, and ask you to install the required version of ``pandoc``. It will also install the development requirements and ``tidy3d`` package in a specific ``poetry`` environment.



The Slow Lane
--------------

If you do not have any of the above tools already installed, let's go through the process of setting things up from scatch.


Get Started
------------

There are a range of handy development functions that you might want to use to streamline your development experience.
