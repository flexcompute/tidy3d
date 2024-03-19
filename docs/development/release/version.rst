Version Release
----------------

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

