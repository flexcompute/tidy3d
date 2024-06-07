``tidy3d`` Project Structure
-----------------------------

As of ``tidy3d>=2.6``, the frontend has been restructured to improve the development cycle. The project directories follow the following structure, which is derived from some recommended `Python project architecture guides <https://docs.python-guide.org/writing/structure/>`_. This is a handy structure because many tools, such as ``sphinx``, integrate quite well with this type of project layout.

.. code::

    docs/
        # sphinx rst files
        ...
        notebooks/
            # Git submodule repository
            # Checks out github.com/flexcompute/tidy3d-notebooks
        faq/
            # Git submodule repository
            # Checks out github.com/flexcompute/tidy3d-faq
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

Important Branches
^^^^^^^^^^^^^^^^^^^

We currently have *three* main branches that have to be kept track of when creating a release, each with different functionality.

.. list-table:: Project Branches
    :header-rows: 1
    :widths: 10 45 45

    * - Name
      - Description
      - Caveats
    * - ``latest``
      - Contains the latest version of the docs. Version release tags are created from this branch.
      - Feature PRs should not be made to this branch as will cause divergence. Only in important documentation patches.
    * - ``develop``
      - Contains the "staging" version of the project. Patch versions and development occurs from these branches.
      - Docs PRs that are non-crucial for the current version should be made to this branch.
    * - ``pre/^*``
      - Contains the next version of the project.
      - Documentation and source code that will only go live in the next version should be updated here.

Sometimes, hopefully infrequently, the `latest` and `develop` branches might diverge.
It is important to bring them back together. However, what happens if we rebase develop into latest?

It could be argued that all the commits in the `latest` branch should have constructed within the `develop` branch.
Then, there is the question if we want to maintain the commit history accordingly. If we just want to maintain the content,
then rebasing and fixing up all the branches works fine. The problem with a merge commit is that it inserts the commits at the historical period in which they were made, rather than the commit period in which we desire to add them.
Hence, it makes sense to merge the `develop` and `latest` branches in order to maintain the same history, assuming the commits should in theory have been in both branches.




