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
    tidy3d.plugins.klayout.drc.results.DRCViolation

DRC Markers
^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.drc.results.EdgeMarker
    tidy3d.plugins.klayout.drc.results.EdgePairMarker
    tidy3d.plugins.klayout.drc.results.MultiPolygonMarker

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.klayout.check_installation 
