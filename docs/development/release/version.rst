Release Workflow
-----------------

The release process is automated through the ``tidy3d-python-client-release`` GitHub Actions workflow.

Triggering a Release
^^^^^^^^^^^^^^^^^^^^

Releases are triggered manually via GitHub Actions:

1. Go to **Actions** ? **public/tidy3d/python-client-release**
2. Click **Run workflow**
3. Configure the release parameters

Release Parameters
^^^^^^^^^^^^^^^^^^

**Required:**

- ``release_tag``: Version tag (e.g., ``v2.10.0``, ``v2.10.0rc1``)

**Optional:**

- ``release_type``: 
  
  - ``draft`` (default): Test release, no PyPI publish
  - ``final``: Official release, publishes to PyPI

- ``workflow_control``: Stage control for resuming partial releases
  
  - ``start-tag``: Full release from tag creation
  - ``start-tests``: Resume from tests
  - ``start-deploy``: Resume from deployment
  - ``only-tag``: Create tag only
  - ``only-tests``: Run tests only
  - ``only-tag-tests``: Tag + tests only
  - ``only-tag-deploy``: Tag + deploy only

- ``client_tests``, ``cli_tests``, ``submodule_tests``: Enable/disable specific test suites (default: all ``true``)

Release Workflow Stages
^^^^^^^^^^^^^^^^^^^^^^^^

**1. Tag Creation**

Creates and pushes a git tag for the release version.

**2. Tests**

Runs comprehensive test suite including:

- **Local tests**: Fast tests on self-hosted runners (Python 3.10, 3.13)
- **Remote tests**: Full matrix tests on GitHub runners (Python 3.10-3.13, Windows/Linux/macOS)
- **CLI tests**: Develop CLI functionality tests
- **Submodule tests**: Integration tests with dependent packages
- **Code quality**: Linting, type checking, commit message validation, security analysis

.. note::
   Submodule tests are **automatically enabled** for final non-RC releases (versions that push to ``latest``).

**3. Deployment** (if tests pass)

- Creates GitHub release (draft or final)
- Publishes to PyPI (final releases only)
- Syncs documentation to ReadTheDocs
- Syncs branches (final releases only)

Test Validation
^^^^^^^^^^^^^^^

All tests are validated through a centralized ``workflow-validation`` job in the tests workflow. The release gatekeeper checks this validation before allowing deployment.

If any required test fails, deployment is automatically blocked.

Release Types
^^^^^^^^^^^^^

**Draft Release** (``release_type: draft``)

- For testing and verification
- Creates GitHub release as draft
- Does **not** publish to PyPI
- Does **not** sync branches
- Tag format not strictly enforced

**Final Release** (``release_type: final``)

- For production deployment
- Creates public GitHub release
- Publishes to PyPI
- Syncs branches to maintain consistency
- Tag format strictly enforced: ``v{major}.{minor}.{patch}[rc{num}]``

**Non-RC Final Release** (e.g., ``v2.10.0``)

- Pushes to ``latest`` branch
- **Automatically runs submodule tests**
- Represents stable production release

**RC Release** (e.g., ``v2.10.0rc1``)

- Pre-release candidate
- Does **not** push to ``latest``
- Submodule tests run only if explicitly enabled

Examples
^^^^^^^^

**Standard Release:**

.. code-block:: yaml

   release_tag: v2.10.0
   release_type: final
   workflow_control: start-tag
   # All tests enabled by default
   # Submodule tests auto-enabled (non-RC)

**RC Release:**

.. code-block:: yaml

   release_tag: v2.10.0rc1
   release_type: final
   workflow_control: start-tag
   # Submodule tests not auto-enabled (RC)

**Test-Only Run:**

.. code-block:: yaml

   release_tag: v2.10.0
   release_type: draft
   workflow_control: only-tests
   client_tests: true

**Resume Failed Release (from deployment):**

.. code-block:: yaml

   release_tag: v2.10.0
   release_type: final
   workflow_control: start-deploy
   # Skips tag creation and tests

Release Changelog Build
^^^^^^^^^^^^^^^^^^^^^^^

Before running the release workflow, generate release notes from ``changelog.d`` fragments:

.. code-block:: bash

   RELVER=$(poetry version -s | sed -E 's/\.dev[0-9]+$//')
   RELDATE=$(date -u +%F)
   poetry run towncrier build --yes --version "${RELVER}" --date "${RELDATE}"
   poetry run python scripts/changelog_refs.py --version "${RELVER}"

This sequence:

- Derives the release version from ``pyproject.toml`` (for example ``2.11.0.dev0`` -> ``2.11.0``).
- Builds ``CHANGELOG.md`` and consumes fragment files.
- Adds or updates the reference-style compare link for ``[RELVER]`` using the latest reachable stable ``vX.Y.Z`` tag as the previous version.

Commit the updated ``CHANGELOG.md`` and removed fragment files in the same release commit.

Automated Changelog PR Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can automate changelog generation and PR creation with GitHub Actions workflow ``public/tidy3d/python-client-build-changelog-pr``.

- Default source branch: ``develop``
- Default target branch: ``develop``
- Optional overrides: ``source_branch``, ``target_branch``, ``release_version``, ``release_date``, ``previous_version``

This workflow runs Towncrier, updates compare reference links, and opens a PR with the resulting changes.

Best Practices
^^^^^^^^^^^^^^

1. **Always test first**: Run a ``draft`` release before ``final``
2. **Use RC versions**: Test with ``v2.10.0rc1`` before releasing ``v2.10.0``
3. **Monitor test results**: Check the ``workflow-validation`` job for detailed test status
4. **Resume on failure**: Use ``workflow_control`` stages to resume from failure points
5. **Verify submodules**: For final releases, ensure submodule tests pass (auto-enabled for non-RC)
