Documentation Release
---------------------

The `tidy3d-docs` repository is no longer synchronized through this repository's GitHub Actions workflows.

Any documentation publishing or mirroring now needs to be handled outside this repo's workflow set.

The `latest` branch holds the state of the docs that we want to host in `latest` version on the website. These are the latest docs (including new notebooks, typo fixes, etc.) related to the last official release (not pre-release).

The `stable` version of the docs on our website is built based on the last version tag which is not a pre-release tag (no `rc`  ending).

Hot Fix & Submodule Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If documentation hot-fix handling is still needed, it should be managed in the docs hosting repository or its own delivery pipeline rather than from `tidy3d`.
