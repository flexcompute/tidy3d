Releasing a new ``tidy3d`` version
----------------------------------

This document contains the relevant information to create and publish a new tidy3d version.

Version Information Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pyproject.toml`` is declarative (ie static) and provides information to the packaging tools like PyPi on what version is ``tidy3d``. However, we also have a ``version.py`` file so that we can dynamically query ``tidy3d.__version__`` within our python version. These two files need to be kept with the same version. This is achieved by using the ``bump-my-version`` utility as described in the following section. **These files should not be manually updated.**

The configuration of the way the version bumping occurs is described in the ``pyproject.toml``.

Bumping Versions
''''''''''''''''

There's a really nice tool to manage the releases which is called ``bump-my-version``, which has already been configured.
It's really easy to use.

Docs of the tool available here https://callowayproject.github.io/bump-my-version/#create-a-default-configuration

You need to have installed the development installation of ``tidy3d``:

.. code-block:: bash

    poetry install -E dev

Now, make sure the git index is clean, and you're ready to release. Test that the release would update the correct files:

.. code-block:: bash

    poetry run bump-my-version show-bump

When you want to bump the version, you only have to do:

.. code-block:: bash

    poetry run bump-my-version bump <patch, or similar according to the show-bump commands>

An example of the ``.bump-my-version.toml`` is as below, and just configures the files to be updated:

.. code-block:: bash

    2024-03-20 16:58:02 ⌚  dxps in ~/flexcompute/tidy3d
    ± |dario/2.6.2/fix_versioning S:1 U:3 ?:2 ✗| → poetry run bump-my-version show-bump
      Specified version (2.6.1) does not match last tagged version (2.6.0)
    2.6.1 ── bump ─┬─ major ─ 3.0.0
                   ├─ minor ─ 2.7.0
                   ├─ patch ─ 2.6.2
                   ├─ pre_l ─ 2.6.1-rc0
                   ╰─ pre_n ─ 2.6.1-dev1


The Release Process
^^^^^^^^^^^^^^^^^^^

Generic Release Process
''''''''''''''''''''''''

When it is time to release version ``x.y.z``, we must do the following:

- Merge any final ``x.y.z`` PRs into the develop branch of `tidy3d-notebooks <https://github.com/flexcompute/tidy3d-notebooks>`_
- Rebase any final ``x.y.z`` PRs into the develop branch of `tidy3d-faq <https://github.com/flexcompute/tidy3d-faq>`_
- Rebase any final ``x.y.z`` PRs into the develop branch of `tidy3d <https://github.com/flexcompute/tidy3d>`_
- Checkout develop branch of tidy3d-frontend and git pull to get any last changes.
- Update and add the ``tidy3d-notebooks`` and ``tidy3d-faq`` submodules by running `git submodule update --recursive`. Make sure both are in the `develop` branches.
- Make a commit to track the added submodule states.
- Do the corresponding backend/web checks.
- Do final updates to ``CHANGELOG.md`` including the links at the bottom of the file. Review everything.
- Ensure ``tidy3d-faq`` and ``tidy3d-notebooks`` submodules are in the right state. This should be the case if the ``test-latest-submodules`` github test passes.
- Push the state to develop.

Note that when a tag is created, the repo state at that tag will be merged into the ``latest`` branch automatically by the ``release.yaml`` GH action.

Pre-Release Process
'''''''''''''''''''

A pre-release would follow the same steps as above except:

- Work on a ``pre/x.y`` branch. Do not merge things to develop or main.
- The tag will be of the form ``vx.y.0.rcn``
- Find the release on github and check the “pre-release” box manually.

Official Release Process
''''''''''''''''''''''''

Make sure to follow the generic release process instructions before. Now that everything looks good, we need to release the package.

ONLY If this is an official release (not a pre-release) we’ll first need to propagate our changes into latest branch using the following comments.

.. code-block:: bash

    git checkout latest
    git merge develop
    git push

Now that we’ve done that, for BOTH PRE AND OFFICIAL we perform the release by pushing a tag for the version number.

.. code-block:: bash

    git tag vx.y.z
    git push origin vx.y.z

Extra note, unlike the version string, the tag has ``v`` in front of version number ``x.y.z``, which is standard practice.

Note: at this point, the GitHub action should take over to make a public GitHub release and push the newly released version to PyPI.

