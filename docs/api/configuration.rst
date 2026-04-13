Configuration API
=================

.. currentmodule:: tidy3d.config

The objects and helpers below expose the public configuration interface.

Manager and Helpers
-------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   ConfigManager
   get_manager
   reload_config

Legacy Compatibility
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   LegacyConfigWrapper
   Environment
   EnvironmentConfig

Registration Utilities
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   register_section
   register_plugin
   register_handler
   get_sections
   get_handlers

Schema Versioning
-----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   CURRENT_CONFIG_VERSION
   register_migration
