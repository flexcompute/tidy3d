Configuration API
=================

.. currentmodule:: tidy3d.config

The objects and helpers below expose the public configuration interface.

Manager and Helpers
-------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.config.ConfigManager
   tidy3d.config.get_manager
   tidy3d.config.reload_config

Legacy Compatibility
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.config.LegacyConfigWrapper
   tidy3d.config.Environment
   tidy3d.config.EnvironmentConfig

Registration Utilities
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.config.register_section
   tidy3d.config.register_plugin
   tidy3d.config.register_handler
   tidy3d.config.get_sections
   tidy3d.config.get_handlers
