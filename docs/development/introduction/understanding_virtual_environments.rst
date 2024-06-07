Understanding Virtual Environments
----------------------------------

Introduction
^^^^^^^^^^^^^

In larger projects, it's crucial to have a *separate* Python environment for each feature or branch you work on. This practice ensures isolation and reproducibility, simplifying testing and debugging by allowing issues to be traced back to specific environments. It also facilitates smoother integration and deployment processes, ensuring controlled and consistent development.
Managing multiple environments might seem daunting, but it's straightforward with the right tools. Follow the steps below to set up and manage your environments efficiently.

Benefits
^^^^^^^^^^^^^

- **Isolation**: Avoids conflicts between dependencies of different features.
- **Reproducibility**: Each environment can be easily replicated.
- **Simplified Testing**: Issues are contained within their respective environments.
- **Smooth Integration**: Ensures features are developed in a consistent setting.

Prerequisites
^^^^^^^^^^^^^^

Make sure that you have ``poetry`` installed. This can be done system-wide with ``pipx`` or within a ``conda`` environment. Note that we use ``conda`` only for setting up the interpreter (Python version) and ``poetry``, not for managing dependencies.
Refer to the official development guide for detailed instructions:

`https://docs.flexcompute.com/projects/tidy3d/en/stable/development/index.html#installation <https://docs.flexcompute.com/projects/tidy3d/en/stable/development/index.html#installation>`_

Setting Up a New Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Check out the branch:

   .. code-block:: bash

      git checkout branch

2. Set up the environment with ``conda`` (skip this step if you don’t use ``conda``):

   .. code-block:: bash

      conda create -n branch_env python=3.11 poetry
      conda activate branch_env
      poetry env use system
      poetry env info # verify you're running the right environment now

3. Install dependencies with ``poetry``:

   .. code-block:: bash

      poetry install -E dev
      poetry run pre-commit install

4. Update the environment when switching to a different branch:

   .. code-block:: bash

      poetry install -E dev



Multiple Folders or Worktrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have multiple folders (e.g., multiple clones or git worktrees), you will need to repeat the environment setup for each folder. Ensure that each folder has its own isolated environment.

By following these steps, you can maintain isolated and reproducible environments for each branch and feature, leading to a more efficient and error-free development process.