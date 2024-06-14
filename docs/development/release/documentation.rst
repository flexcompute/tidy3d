Documentation Release
---------------------

The `tidy3d-docs` repository automatically mirrors the `tidy3d` repository. Specifically, these branches are automatically synced.

- main
- latest
- develop
- 'pre/*'
- 'v*'

These branches are synced to the tidy3d-docs repo through the sync-readthedocs-repo Github action.
You can read the latest versions synced in the action file.
However, you need to configure how they appear in the documentation build in the readthedocs admin page.
Only latest is the public version, others are private.

The `latest` branch holds the state of the docs that we want to host in `latest` version on the website. These are the latest docs (including new notebooks, typo fixes, etc.) related to the last official release (not pre-release).

The `stable` version of the docs on our website is built based on the last version tag which is not a pre-release tag (no `rc`  ending).

Hot Fix & Submodule Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make a “hot fix” (eg fix a typo, add a notebook, update the release FAQ), just update the ``latest`` branch in ``tidy3d`` repo. This should automatically sync to `tidy3d-docs`, and trigger a docs rebuild. **However, we should avoid this as this will cause the ``develop`` and ``latest branches`` to diverge.** Ideally, these hot fixes could wait until the next pre/post-release to be propagated through.

NOTE: To avoid conflicts, ideally we should only update ``latest`` by merging ``develop`` in it, or at the very least we should make sure changes are propagated to both branches.