
Project Structure
=================

As of ``tidy3d>=2.6``, the frontend has been restructured to improve the development cycle. The project directories follow the following structure, which is derived from some recommended `Python project architecture guides <https://docs.python-guide.org/writing/structure/>`_. This is a handy structure because many tools, such as ``sphinx``, integrate quite well with this type of project layout.

.. code::

    docs/
        # sphinx rst files
        ...
        notebooks/
            # Git submodule repository
            # Checks out github.com/flexcompute/tidy3d-notebooks
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

Release Flow
^^^^^^^^^^^^^^^

This is very straightforward. You just need to make sure that the ``develop`` branches of both ``tidy3d/`` and ``tidy3d-notebooks/`` repositories within the ``./`` and ``./docs/notebooks/`` directories are updated. The ``readthedocs`` documentation will be automatically updated through the ``sync-readthedocs-repo`` Github action.


