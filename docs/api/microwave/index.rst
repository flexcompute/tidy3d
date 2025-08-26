Microwave & RF |:satellite:|
============================

This page contains an overview of tidy3d microwave and RF simulation capabilities.

.. warning::

    RF simulations are subject to new license requirements in the future. These components are within the RF scope.

Modelling Components
--------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler


Component Modeler Data
----------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModelerData


Ports
-----

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.smatrix.Port
    tidy3d.plugins.smatrix.LumpedPort
    tidy3d.plugins.smatrix.CoaxialLumpedPort
    tidy3d.plugins.smatrix.WavePort


Voltage & Current Integrals
----------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.microwave.AxisAlignedPathIntegral
   tidy3d.plugins.microwave.VoltageIntegralAxisAligned
   tidy3d.plugins.microwave.CurrentIntegralAxisAligned
   tidy3d.plugins.microwave.CustomPathIntegral2D
   tidy3d.plugins.microwave.CustomVoltageIntegral2D
   tidy3d.plugins.microwave.CustomCurrentIntegral2D


Metrics Analysis
----------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.microwave.ImpedanceCalculator
   tidy3d.plugins.microwave.RectangularAntennaArrayCalculator
   tidy3d.plugins.microwave.LobeMeasurer


Data Array
----------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.PortDataArray
   tidy3d.plugins.smatrix.TerminalPortDataArray


Performance Metrics Data
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.AntennaMetricsData


Migration 2.9 -> 2.10 onwards
-----------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.run.
   tidy3d.plugins.smatrix.TerminalPortDataArray


