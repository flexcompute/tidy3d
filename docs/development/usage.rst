Using the Development Flow
==========================

Developing ``tidy3d`` with ``poetry``
--------------------------------------------------

`poetry <https://python-poetry.org/>`_ is an incredibly powerful tool for reproducible package development environments and dependency management.

If you are developing ``tidy3d``, we recommend you work within the configured ``poetry`` environment defined by ``poetry.lock``. The way to install this environment is simple:

.. code::

    cd tidy3d/
    poetry install -E dev

This function will install the package with all the development dependencies automatically. This means you should be able to run any functionality that is possible with ``tidy3d`` reproducibly.

It is important to note the function above is equivalent to ``pip install tidy3d[dev]``, but by using ``poetry`` there is a guarantee of using the reproducible locked environment.


``poetry`` with an external virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to use ``poetry`` for package development. However, there are some cases where you might need to use an external virtual environment for some operations. There are a few workarounds where you can leverage the reproducibility of the ``poetry`` managed environment with the freedom of a standard virtual environment. There are a few more instructions and explanations in `the poetry env docs <https://python-poetry.org/docs/managing-environments/>`_ . F See the following example:

.. code::

    mamba create -n tidy3denv python==3.10 # create venv with mamba
    mamba activate tidy3denv # activate the venv
    poetry env use python # using the mamba venv python now
    poetry env info # verify the venvs used by poetry and mamba
    cd anywhere
    # you can use the python activated venv anywhere.

There are also other methodologies of implementing common dependencies management.

Common Utilities
""""""""""""""""""""

There are a range of handy development functions that you might want to use to streamline your development experience.

.. list-table:: Use Cases
    :header-rows: 1
    :widths: 25 25 50

    * - Description
      - Caveats
      - Command
    * - Benchmark timing import of ``tidy3d``
      - Verify the available timing tests by running the command without any arguments.
      - ``poetry run tidy3d develop benchmark-timing-operations -c <timing_command>``
    * - Build documentation on reproducible environment
      -
      - ``poetry run tidy3d develop build-docs``
    * - Build documentation with latest remote notebooks
      - It is defaulted to the  ``develop`` branch of the ``tidy3d-notebooks`` repository.
      - ``poetry run tidy3d develop build-docs-remote-notebooks``
    * - Complete notebooks + base testing of the ``tidy3d``
      - Make sure you have the notebooks downloaded.
      - ``poetry run tidy3d develop test-all``
    * - Dual snapshot between the ``tidy3d`` and ``notebooks`` source and submodule repository.
      - Make sure you are on the correct git branches you wish to commit to on both repositories, and all `non-git-ignored` files will be added to the commit.
      - ``tidy3d develop commit <your message>``
    * - Interactively convert all markdown files to rst (replacement for m2r2)
      -
      - ``poetry run tidy3d develop convert-all-markdown-to-rst``
    * - Running ``pytest`` commands inside the ``poetry`` environment.
      - Make sure you have already installed ``tidy3d`` in ``poetry`` and you are in the root directory.
      - ``poetry run pytest``
    * - Run ``coverage`` testing from the ``poetry`` environment.
      -
      - ``poetry run coverage run -m pytest``
    * - Standard testing of the ``tidy3d`` frontend
      - Make sure you have already installed ``tidy3d`` in ``poetry`` and you are in the root directory.
      - ``poetry run tidy3d develop test-base``
    * - Using ``tidy3d develop`` commands inside the ``poetry`` environment.
      - Make sure you have already installed ``tidy3d`` in ``poetry``
      - ``poetry run tidy3d develop <your command>``
    * - Update lockfile after updating a dependency in ``pyproject.toml``
      - Remember to install after this command.
      - ``poetry lock``
    * - Update and replace all the docstrings in the codebase between versions
      -
      - ``poetry run tidy3d develop replace-in-files``



