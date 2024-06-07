An Introduction to the Development Flow
========================================

This page hopefully will get you started to develop Tidy3D.

**TLDR:**

- Branch off of the target branch (usually ``develop`` or ``pre/x.x``), work on your branch, and submit a PR when ready.
- Use isolated development environments with ``poetry``.
- Use ``ruff`` to lint and format code, and install the pre-commit hook via ``pre-commit install`` to automate this.
- Document code using NumPy-style docstrings.
- Write unit tests for new features and try to maintain high test coverage.

.. toctree::
    :maxdepth: 1

    understanding_virtual_environments
    understanding_poetry
    code_quality_principles
    project_structure


.. include:: /development/introduction/understanding_virtual_environments.rst
.. include:: /development/introduction/understanding_poetry.rst
.. include:: /development/introduction/code_quality_principles.rst
.. include:: /development/introduction/project_structure.rst