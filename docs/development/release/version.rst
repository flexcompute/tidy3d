Releasing a new ``tidy3d`` version
----------------------------------

This document contains the relevant information to create and publish a new tidy3d version.

Version Information Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pyproject.toml`` is declarative (ie static) and provides information to the packaging tools like PyPi on what version is ``tidy3d``. However, we also have a ``version.py`` file so that we can dynamically query ``tidy3d.__version__`` within our python version. These two files need to be kept with the same version. This is achieved by using the ``bump-my-version`` utility as described in the following section. **These files should not be manually updated.**

The configuration of the way the version bumping occurs is described in the ``pyproject.toml``.