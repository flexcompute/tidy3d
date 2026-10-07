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

Batched DRC
~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.BatchedDRCChecker

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

LVS
---

.. include:: ../../../tidy3d/plugins/klayout/lvs/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2

LVS Configuration
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.LVSConfig

LVS Runner
~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.LVSRunner
    plugins.klayout.run_lvs_on_gds

LVS Results
~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.LVSResults
    plugins.klayout.lvs.results.LVSComparison
    plugins.klayout.lvs.results.LVSMessage
    plugins.klayout.lvs.results.LVSGeometry

LVS Pairs
~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.lvs.results.LVSPair
    plugins.klayout.lvs.results.LVSNetPair
    plugins.klayout.lvs.results.LVSPinPair
    plugins.klayout.lvs.results.LVSDevicePair
    plugins.klayout.lvs.results.LVSSubcircuitPair

LVS Netlists
~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.lvs.netlist.LVSNetlist
    plugins.klayout.lvs.netlist.LVSCircuit
    plugins.klayout.lvs.netlist.LVSNet
    plugins.klayout.lvs.netlist.LVSConnection
    plugins.klayout.lvs.netlist.LVSPin
    plugins.klayout.lvs.netlist.LVSSubcircuit
    plugins.klayout.lvs.netlist.LVSPlacement
    plugins.klayout.lvs.netlist.LVSLabel
    plugins.klayout.lvs.netlist.LVSDevice
    plugins.klayout.lvs.netlist.LVSDevicePart
    plugins.klayout.lvs.netlist.LVSDeviceClass
    plugins.klayout.lvs.netlist.LVSParameterDefinition
    plugins.klayout.lvs.netlist.LVSDeviceAbstract
    plugins.klayout.lvs.netlist.LVSLayer

Utilities
---------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.klayout.check_installation
