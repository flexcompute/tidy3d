Configuration Guide |:gear:|
============================

.. highlight:: python

Working with cloud simulations usually requires a handful of settings such as
your API key, the active environment, and any local tweaks you make while
experimenting. The ``config`` object (available via ``from tidy3d import config``)
keeps all of this in one place through a single interface. This page explains
how it behaves, where values are stored, and how to keep your changes
consistent across sessions. For a catalog of every option, see :doc:`reference`.

Getting Started
---------------

Most users only need the following import::

    from tidy3d import config

You can then read or update sections just like attributes::

    # read values
    print(config.web.api_endpoint)
    print(config.logging.level)

    # update values
    config.logging.level = "DEBUG"
    config.web.timeout = 60
    config.save()

The ``save()`` call writes your edits to disk so the same settings load the
next time you import ``tidy3d``.

Where Settings Are Stored
-------------------------

Tidy3D chooses a configuration directory the first time you import the module.
The location depends on your operating system:

.. list-table:: Default configuration directory
   :widths: 30 70
   :header-rows: 1

   * - Platform
     - Path
   * - macOS / Linux
     - ``~/.config/tidy3d``
   * - Windows
     - ``C:\Users\<you>\.config\tidy3d``

You can override this by setting the ``TIDY3D_BASE_DIR`` environment variable
*before* importing ``tidy3d``. When it is present, the config files are kept in
``<base>/.tidy3d``. If the chosen location is not writable, Tidy3D falls back to
a temporary directory and warns that the settings will not persist.

Files Inside the Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``config.toml`` – base settings shared by all profiles.
- ``profiles/<name>.toml`` – optional overrides for custom profiles. Each file
  only contains the differences from the base settings.
- ``default_profile`` is base metadata and is only valid in ``config.toml``.

Each configuration file includes a root ``config_version`` key. When it is
missing, Tidy3D treats the file as version 0 and migrates it in memory to the
current schema on load.

Priority Order
--------------

Whenever you read ``config.<section>.<field>``, the value comes from the highest
priority source in the list below. Lower entries only apply when the ones above
them do not set a value.

1. Runtime changes you make in the current Python session.
2. Environment variables (``TIDY3D_<SECTION>__<FIELD>``).
3. Profile overrides from ``profiles/<name>.toml``.
4. The base ``config.toml`` file.
5. Built-in profiles (for example ``prod`` and ``dev``) bundled with Tidy3D.
6. Default values defined by the software.

This means environment variables always win over ``config.toml``, and any
attribute you set in code wins over everything else until you discard it or
call ``save()``.

Making Changes That Last
------------------------

Runtime Updates
~~~~~~~~~~~~~~~

Assignments like ``config.logging.level = "INFO"`` apply immediately but only
live in memory. They affect new simulations started in the same interpreter but
disappear when the process exits.

For short-lived overrides, use ``config`` as a context manager. Changes made
inside the block are restored automatically when the block exits::

    from tidy3d import config, web

    with config as tmp_config:
        tmp_config.logging.level = "DEBUG"
        web.run(sim)

    # runtime overrides from the block are cleared here

Saving to Disk
~~~~~~~~~~~~~~

Call ``config.save()`` to write the current profile to disk. The method removes
environment-variable overrides automatically so you never persist an API key or
other secret that was loaded from the shell. To store the full set of values,
including defaults, pass ``include_defaults=True``::

    config.save(include_defaults=True)

Profiles
--------

Tidy3D ships with built-in profiles such as ``prod`` and ``dev``. Switch between
them with::

    config.switch_profile("dev")

To create your own profile, switch to a new name, edit settings, then call
``save()``::

    config.switch_profile("my_profile")
    config.web.api_endpoint = "https://example.com"
    config.save()

This writes ``profiles/my_profile.toml`` containing only the adjustments you
made. Use ``config.profiles.list()`` to discover available built-in and user
profiles.

Environment Variables
---------------------

Environment variables let you override individual options without touching any
files. The naming pattern is ``TIDY3D_<SECTION>__<FIELD>``, for example::

    export TIDY3D_LOGGING__LEVEL=WARNING

Supported variables take effect the next time you import ``tidy3d``. Remove a
variable or clear the shell environment to restore the lower priority setting.

Command Line Helpers
--------------------

Use ``tidy3d configure`` to store your API key in ``config.toml``. The command
creates the directory if it is missing and updates only the ``web`` section.

Use ``tidy3d config upgrade`` to preview or apply schema migrations for
``config.toml`` and profile files. Pass ``--dry-run`` to inspect diffs or
``--check`` in CI. Automatic write-back can be disabled with
``TIDY3D_CONFIG_AUTO_MIGRATE=0``; use ``TIDY3D_CONFIG_FORWARD_COMPAT=strict`` to
error on newer schema versions instead of best-effort parsing.

.. _refreshing-local-license-entitlements:

Refreshing Local License Entitlements
-------------------------------------

Some local features, including ``tidy3d-extras`` features, cache license
entitlements locally to avoid repeated license server calls. If your account
licenses change while a Python process is still running, clear this cache
before retrying the local feature. This applies to the web-authenticated local
license checks used by the standard client and custom endpoint profiles. It does
not affect the separate compile-time on-prem release license-file path::

    tidy3d configure --refresh-licenses

The same operation is available from Python::

    import tidy3d.web as web

    web.refresh_licenses()

If ``--refresh-licenses`` or ``refresh_licenses=True`` is combined with a
configuration update that is rejected, the license cache is not refreshed and
the command or call reports the failed configuration update. Resolve the
configuration error first, or run ``tidy3d configure --refresh-licenses`` /
``web.refresh_licenses()`` separately.

Successful API key or API endpoint configuration updates through
``tidy3d configure ...`` or ``web.configure(...)`` also clear the local
license cache after the configuration is saved. This operation clears cached
entitlements and writes a cache generation marker. Running Python processes
observe that marker on their next native license check and invalidate their
native in-process cache. A shell ``tidy3d configure ...`` command cannot clear
an already-cached failed ``tidy3d_extras`` import in an existing Python process;
call ``web.refresh_licenses()`` in that interpreter or restart the process in
that case. The refresh is cache-wide within the active Tidy3D config directory,
so it clears persisted license auth files for all API key and endpoint hashes
stored there. It does not contact the license server immediately; the next
licensed local feature check fetches current entitlements and caches them again.

If this automatic refresh fails after a configuration update is saved,
``tidy3d configure`` exits nonzero and ``web.configure(...)`` raises an error
that starts with ``Configuration saved, but license cache refresh failed``. In
that partial-success case, the configuration change remains persisted; rerun
``tidy3d configure --refresh-licenses`` or call ``web.refresh_licenses()`` after
resolving the refresh failure.

The command-line refresh updates the on-disk cache used by future license
checks, but it runs in a separate process and cannot reset imports inside an
already-running Python interpreter. If ``tidy3d_extras`` previously failed to
initialize in the current interpreter, call ``web.refresh_licenses()`` in that
interpreter before importing it again. Existing Python variables bound to the
old ``tidy3d_extras`` module are not updated; import the package again before
using it directly.

Legacy Access Points
--------------------

Older code paths such as ``tidy3d.config.logging_level`` and ``tidy3d.config.Env``
were removed in Tidy3D 2.12. Accessing these names now raises an error with
replacement guidance.

Next Steps
----------

- :doc:`reference`
- :doc:`../api/configuration`

.. toctree::
   :hidden:

   reference
   nexus
