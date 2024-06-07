Code Quality Principles
------------------------

When writing a code snippet, remember the saying: "code is read more than written". We want to maintain our code maintainable, readable and high quality.

Linting & Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^

To maintain code quality, we use `Ruff <https://github.com/astral-sh/ruff>`_ as a linter and code formatter. A linter analyzes code to identify and flag potential errors, stylistic issues, and code that doesn't adhere to defined standards (such as `PEP8 <https://peps.python.org/pep-0008/>`_). A code formatter automatically restructures the code to ensure it is consistently styled and properly formatted, making it consistent across the code base.

Run ``ruff format`` to format all Python files:

.. code-block:: bash

   poetry run ruff format .

Run ``ruff check`` to check for style and other issues. Many common warnings can be automatically fixed with the ``--fix`` flag:

.. code-block:: bash

   poetry run ruff check tidy3d --fix

The configuration defining what ``ruff`` will correct lives in ``pyproject.toml`` under the ``[tool.ruff]`` section.

When submitting code, for tests to pass, ``ruff`` should give no warnings.

Documentation
^^^^^^^^^^^^^^^

Document all code you write using `NumPy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Testing
-------

Here we will discuss how tests are defined and run in Tidy3d.

Unit Testing
^^^^^^^^^^^^^^

The tests live in ``tests/`` directory.

We use `pytest <https://docs.pytest.org/en/6.2.x/>`_ package for our testing.

To run all of the tests, call:

.. code-block:: bash

   poetry run pytest -rA tests

This command will trigger ``pytest`` to go through each file in ``tests/`` called ``test*.py`` and run each function in that file with a name starting with ``test``.

If all of these functions run without any exceptions being raised, the tests pass!

The specific configuration we use for ``pytest`` lives in the ``[tool.pytest.ini_options]`` section of ``pyproject.toml``.

These tests are automatically run when code is submitted using GitHub Actions, which tests on Python 3.9 through 3.12 running on Ubuntu, MacOS, and Windows operating systems, as well as Flexcompute's servers.

Note: The ``-rA`` flag is optional but produces output that is easily readable.

Note: You may notice warnings and errors in the ``pytest`` output, this is because many of the tests intentionally trigger these warnings and errors to ensure they occur in certain situations. The important information about the success of the test is printed out at the bottom of the ``pytest`` output for each file.

To get a code coverage report:

.. code-block:: bash

   pip install pytest-cov

if not already installed

To run coverage tests with results printed to STDOUT:

.. code-block:: bash

   pytest tests --cov-report term-missing --cov=tidy3d

To run coverage tests and get output as .html (more intuitive):

.. code-block:: bash

   pytest tests --cov-report=html --cov=tidy3d
   open htmlcov/index.html

Automated Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^

We use GitHub Actions to perform these tests automatically and across different operating systems.

On commits, each of the ``pytest`` tests are run using Python 3.9 - 3.12 installed on Ubuntu, MacOS, and Windows operating systems.

See the "actions" tab for details on previous tests and ``.github/workflows/run_tests.yml`` for the configuration and to see the specific tests run.

See `this <https://docs.github.com/en/actions/learn-github-actions>`_ for more explanation.

Other Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are additional tests in both the `documentation <https://github.com/flexcompute/tidy3d-docs/tree/main/docs>`_ and our private backend code. The same practices outlined here apply to those tests.

More Resources on Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^

A useful explanation for those curious to learn more about the reasoning behind these decisions:

`https://www.youtube.com/watch?v=DhUpxWjOhME <https://www.youtube.com/watch?v=DhUpxWjOhME>`