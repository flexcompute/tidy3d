
Project Structure
=================

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

Release Flow
^^^^^^^^^^^^^^^

This is very straightforward. You just need to make sure that the `develop` branches of both `tidy3d/` and `tidy3d-notebooks/` are updated. Then these will be automatically updated on the `readthedocs` documentation through the Github actions.


Utilities
^^^^^^^^^^

There are a range of handy development functions that you might want to use to streamline your development experience.

.. list-table:: Use Cases
   :header-rows: 1

    * - Description
      - Caveats
      - Command
    * - Dual snapshot between the ``tidy3d`` and ``notebooks`` source and submodule repository.
      - Make sure you are on the correct git branches you wish to commit to on both repositories, and all `non-git-ignored` files will be added to the commit.
      - ``tidy3d develop commit <your message>``