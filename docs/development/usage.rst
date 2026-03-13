Using the Development Flow
==========================

Developing ``tidy3d`` with ``uv``
--------------------------------------------------

`uv <https://docs.astral.sh/uv/>`_ is an incredibly powerful tool for reproducible package development environments and dependency management.

If you are developing ``tidy3d``, we recommend you work within the configured ``uv`` environment defined by ``uv.lock``. The way to install this environment is simple:

.. code::

    cd tidy3d/
    uv sync --frozen --extra dev

This function will install the package with all the development dependencies automatically. This means you should be able to run any functionality that is possible with ``tidy3d`` reproducibly.

It is important to note the function above is equivalent to ``pip install tidy3d[dev]``, but by using ``uv`` there is a guarantee of using the reproducible locked environment.


``uv`` with an external virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to use ``uv`` for package development. However, there are some cases where you might need to use an external virtual environment for some operations. There are a few workarounds where you can leverage the reproducibility of the ``uv`` managed environment with the freedom of a standard virtual environment. For more details, see the `uv documentation <https://docs.astral.sh/uv/>`_. See the following example:

.. code::

    mamba create -n tidy3denv python==3.10 # create venv with mamba
    mamba activate tidy3denv # activate the venv
    cd tidy3d
    uv sync --active --frozen --extra dev # sync to the currently active venv
    uv run python -V # confirm commands run in that environment

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
      - ``uv run tidy3d develop benchmark-timing-operations -c <timing_command>``
    * - Build documentation on reproducible environment
      -
      - ``uv run tidy3d develop build-docs``
    * - Build documentation with latest remote notebooks
      - It is defaulted to the  ``develop`` branch of the ``tidy3d-notebooks`` repository.
      - ``uv run tidy3d develop build-docs-remote-notebooks``
    * - Complete notebooks + base testing of the ``tidy3d``
      - Make sure you have the notebooks downloaded.
      - ``uv run tidy3d develop test-all``
    * - Dual snapshot between the ``tidy3d`` and ``notebooks`` source and submodule repository.
      - Make sure you are on the correct git branches you wish to commit to on both repositories, and all `non-git-ignored` files will be added to the commit.
      - ``tidy3d develop commit <your message>``
    * - Interactively convert all markdown files to rst (replacement for m2r2)
      -
      - ``uv run tidy3d develop convert-all-markdown-to-rst``
    * - Running ``pytest`` commands inside the ``uv`` environment.
      - Make sure you have already installed ``tidy3d`` in ``uv`` and you are in the root directory.
      - ``uv run pytest``
    * - Analyze slow ``pytest`` runs with durations / cProfile / debug subset helpers.
      - Use ``--debug`` to run only the first N collected tests or ``--profile`` to capture call stacks.
      - ``python scripts/profile_pytest.py [options]``
    * - Track ``pytest`` RAM usage over time and per test.
      - Defaults to pytest's configured parallelism; use ``--single-process`` to disable xdist.
      - ``uv run python scripts/pytest_ram_profile.py [options]``
    * - Run ``coverage`` testing from the ``uv`` environment.
      -
      - ``uv run coverage run -m pytest``
    * - Standard testing of the ``tidy3d`` frontend
      - Make sure you have already installed ``tidy3d`` in ``uv`` and you are in the root directory.
      - ``uv run tidy3d develop test-base``
    * - Using ``tidy3d develop`` commands inside the ``uv`` environment.
      - Make sure you have already installed ``tidy3d`` in ``uv``
      - ``uv run tidy3d develop <your command>``
    * - Update lockfile after updating a dependency in ``pyproject.toml``
      - Remember to install after this command.
      - ``uv lock``
    * - Update and replace all the docstrings in the codebase between versions
      -
      - ``uv run tidy3d develop replace-in-files``
