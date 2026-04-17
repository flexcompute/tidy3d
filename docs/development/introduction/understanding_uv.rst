Using `uv` for package management
--------------------------------------

What is uv
^^^^^^^^^^^^^^^^^^^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ is the package and environment tool used in this repository.

We use it to:

- Resolve and lock dependencies.
- Create reproducible development environments.
- Run commands in the project environment.
- Build distribution artifacts.

Why we use it in ``tidy3d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Dependency metadata lives in one place: ``pyproject.toml``.
2. ``uv.lock`` captures exact resolved versions for reproducible installs.
3. ``uv sync --frozen`` and ``uv run --frozen`` keep local and CI behavior aligned.

How to install it
^^^^^^^^^^^^^^^^^^^^^^^^^^

See the development installation guide in this documentation, or follow the official uv docs for
`installation <https://docs.astral.sh/uv/getting-started/installation/>`_ and
`project workflows <https://docs.astral.sh/uv/guides/projects/>`_.

Project workflow in this repo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create/update the development environment:

.. code-block:: bash

   uv sync --frozen --extra dev

Run commands inside the managed environment:

.. code-block:: bash

   uv run --frozen pytest
   uv run --frozen pre-commit run --all-files

Update locked dependencies after changing dependency metadata:

.. code-block:: bash

   uv lock

Dependency metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^

Dependencies are defined in ``pyproject.toml``:

- Runtime dependencies under ``[project.dependencies]``.
- Optional dependency sets under ``[project.optional-dependencies]``.

Publishing/building
^^^^^^^^^^^^^^^^^^^^^^^^^^

Build local distribution artifacts:

.. code-block:: bash

   uv build --sdist --wheel

For publish configuration and authentication details, refer to the
`uv publishing guide <https://docs.astral.sh/uv/guides/publish/>`_.
