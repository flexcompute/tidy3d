.. currentmodule:: tidy3d

KLayout Integration
-------------------

.. toctree::

    ./../../../tidy3d/plugins/klayout/README.md

DRC
~~~

.. toctree::

    ./../../../tidy3d/plugins/klayout/drc/README.md

DRC Configuration
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.DRCConfig

DRC Runner
^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.DRCRunner
    tidy3d.plugins.klayout.run_drc_on_gds

DRC Results
^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.DRCResults
    tidy3d.plugins.klayout.drc.DRCViolation

DRC Markers
^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.drc.EdgeMarker
    tidy3d.plugins.klayout.drc.EdgePairMarker
    tidy3d.plugins.klayout.drc.MultiPolygonMarker

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.check_installation 