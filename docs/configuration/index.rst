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

If you have older files in ``~/.tidy3d``, run ``tidy3d config migrate`` to move
them into the new location described above. The command copies the legacy files
into the canonical directory, leaving the originals untouched unless you pass
``--delete-legacy``. Use ``--overwrite`` if you have already started using the
new location and want to replace those files with the legacy versions.

Legacy Access Points
--------------------

Older code paths such as ``tidy3d.config.logging_level`` and ``tidy3d.config.Env``
still work. They emit a ``DeprecationWarning`` each time you use them to help
you transition to the modern interface. See :doc:`migration` for advice on
updating scripts that depend on these names.

Next Steps
----------

- :doc:`reference`
- :doc:`migration`
- :doc:`../api/configuration`
