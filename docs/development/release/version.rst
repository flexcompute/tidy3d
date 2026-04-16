Release Workflow
-----------------

The GitHub Actions release workflows that previously lived in this repository have been removed.

Coordinate with the maintainers for the current release and publication path. The only documented steps that still apply inside this repository are the local changelog preparation commands below.

Release Changelog Build
^^^^^^^^^^^^^^^^^^^^^^^

Before preparing a release, generate release notes from ``changelog.d`` fragments:

.. code-block:: bash

   RELVER=$(uv version --short | sed -E 's/\.dev[0-9]+$//')
   RELDATE=$(date -u +%F)
   uv run towncrier build --yes --version "${RELVER}" --date "${RELDATE}"
   uv run python scripts/changelog_refs.py --version "${RELVER}"

This sequence:

- Derives the release version from ``pyproject.toml`` (for example ``2.11.0.dev0`` -> ``2.11.0``).
- Builds ``CHANGELOG.md`` and consumes fragment files.
- Adds or updates the reference-style compare link for ``[RELVER]`` using the latest reachable stable ``vX.Y.Z`` tag as the previous version.

Commit the updated ``CHANGELOG.md`` and removed fragment files in the same release commit.

Best Practices
^^^^^^^^^^^^^^

1. **Build changelog updates locally** before cutting a release.
2. **Commit ``CHANGELOG.md`` updates and consumed fragments together** so the release diff stays reviewable.
3. **Coordinate tagging and publication outside this repository's removed GitHub Actions workflow set.**
