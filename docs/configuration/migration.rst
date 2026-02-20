Upgrading Existing Setups
=========================

This short note highlights the differences you may notice when moving from
earlier versions of Tidy3D to the current configuration manager.

File Locations
--------------

- Previous releases stored settings in ``~/.tidy3d`` on all platforms. The new
  manager now prefers the platform-specific paths described in
  :doc:`index`.
- Your existing ``~/.tidy3d/config`` is still respected. Run
  ``tidy3d config migrate`` if you would like to copy it into the new directory.
  Append ``--overwrite`` to replace any files that already exist in the new
  location, and ``--delete-legacy`` to remove ``~/.tidy3d`` after the copy.

Environment Switching
---------------------

- The ``Env`` helper remains available. Calls such as ``Env.dev.active()`` now
  forward to the new manager and produce a ``DeprecationWarning`` to encourage
  the modern API, e.g. ``config.switch_profile("dev")``.

Legacy Attributes
-----------------

- Shorthand properties ``config.logging_level``, ``config.log_suppression``,
  and ``config.use_local_subpixel`` still work and set the equivalent fields in
  ``config.logging`` or ``config.simulation``. Each call raises a warning so
  you can update scripts at your own pace.

Schema Versioning
-----------------

- Config files now include a root ``config_version`` to support incremental
  schema migrations. Missing versions are treated as ``0``.
- The loader migrates configs in memory and writes back upgraded files by
  default. Set ``TIDY3D_CONFIG_AUTO_MIGRATE=0`` to disable auto write-back.
- Use ``tidy3d config upgrade --dry-run`` to inspect schema diffs or
  ``tidy3d config upgrade --check`` in CI to verify files are current.
- Set ``TIDY3D_CONFIG_FORWARD_COMPAT=strict`` to error on newer schema versions
  instead of best-effort parsing.
