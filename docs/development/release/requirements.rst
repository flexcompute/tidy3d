Release Deployment Requirements
-------------------------------

When a new release is created, it is necessary to create a "mirror" branch under the `tidy3d-docs` repository in order for it to build. If this is not there an error such as the following might appear under the `sync-readthedocs-repo` Github action. You will also need to do this for full releases.

.. code-block::

    /usr/bin/git worktree remove github-pages-deploy-action-temp-deployment-folder --force
    Error: The deploy step encountered an error: There was an error creating the worktree: The process '/usr/bin/git' failed with exit code 128 ❌ ❌
    Notice: Deployment failed! ❌