.. currentmodule:: tidy3d

KLayout Integration
-------------------

.. include:: ../../../tidy3d/plugins/klayout/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2

DRC
---

.. include:: ../../../tidy3d/plugins/klayout/drc/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2

DRC Configuration
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.DRCConfig

DRC Runner
~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.DRCRunner
    plugins.klayout.run_drc_on_gds

DRC Results
~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.DRCResults
    plugins.klayout.drc.results.DRCViolation

DRC Markers
~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.drc.results.EdgeMarker
    plugins.klayout.drc.results.EdgePairMarker
    plugins.klayout.drc.results.PolygonMarker
    plugins.klayout.drc.results.MultiPolygonMarker

Utilities
---------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.check_installation
