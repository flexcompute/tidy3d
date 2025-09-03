Microwave and RF |:satellite:|
==============================

Overview
--------

This page contains an overview of tidy3d microwave and RF simulation capabilities.

.. warning::

   RF simulations are subject to new license requirements in the future. These components are within the RF scope.

~~~~

TerminalComponentModeler and Data
---------------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.plugins.smatrix.TerminalPortDataArray   


~~~~

RF Materials
------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst
	      
   tidy3d.PECMedium
   tidy3d.PMCMedium
   tidy3d.LossyMetalMedium
   

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst
	      
   tidy3d.plugins.dispersion.FastDispersionFitter

~~~~

Layer-based Grid Refinement
---------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst
	      
   tidy3d.LayerRefinementSpec
   tidy3d.CornerFinderSpec


~~~~

Lumped Port and Elements
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst
	      
   tidy3d.plugins.smatrix.LumpedPort
   tidy3d.plugins.smatrix.CoaxialLumpedPort


.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.LumpedResistor
   tidy3d.CoaxialLumpedResistor
   tidy3d.RLCNetwork
   tidy3d.AdmittanceNetwork
   tidy3d.LinearLumpedElement

	      
~~~~

Wave Port
---------

.. autosummary::
    :toctree: ../_autosummary/
    :template: module.rst

    tidy3d.plugins.smatrix.WavePort
    tidy3d.ModeSpec


.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.microwave.AxisAlignedPathIntegral
   tidy3d.plugins.microwave.VoltageIntegralAxisAligned
   tidy3d.plugins.microwave.CurrentIntegralAxisAligned
   tidy3d.plugins.microwave.CustomPathIntegral2D
   tidy3d.plugins.microwave.CustomVoltageIntegral2D
   tidy3d.plugins.microwave.CustomCurrentIntegral2D
   tidy3d.plugins.microwave.ImpedanceCalculator
    

~~~~

Radiation and Scattering
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.DirectivityMonitor
   tidy3d.plugins.microwave.RectangularAntennaArrayCalculator
   tidy3d.plugins.microwave.LobeMeasurer
   tidy3d.AntennaMetricsData

~~~~

Migration 2.9 -> 2.10 onwards
-----------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.run.run
   tidy3d.plugins.smatrix.run.create_batch
   tidy3d.plugins.smatrix.run.compose_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_component_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_terminal_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_modeler_data
   tidy3d.plugins.smatrix.run.compose_modeler
   tidy3d.plugins.smatrix.run.compose_modeler_data
   tidy3d.plugins.smatrix.run.compose_component_modeler_data
   tidy3d.plugins.smatrix.run.compose_terminal_modeler_data
   tidy3d.plugins.smatrix.run.compose_simulation_data_index


~~~~
