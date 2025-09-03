Microwave and RF |:satellite:|
==============================

Overview
--------

This page consolidates Tidy3D features related to microwave and RF simulation. While microwave/RF and optical simulations have many properties in common, there are some differences in the typical RF user workflow that deserve special consideration.

The following sections discuss:

* `TerminalComponentModeler and Data`_: The core simulation object in microwave/RF models
* `RF Materials`_: Typical material types in microwave/RF simulation
* `Layer-based Grid Refinement`_: Automated grid refinement strategy for planar structures (e.g. printed circuit boards)
* `Lumped Port and Elements`_: Lumped excitations and terminations
* `Wave Port`_: Port excitation based on mode solutions
* `Radiation and Scattering`_: Useful features for antenna and scattering problems
* `Backwards Compatibility`_: Migration from 2.9 to 2.10 onwards

.. seealso::

   If you are completely new to Tidy3D, we recommend checking out the following beginner resources:

   + `Quickstart <../notebooks/StartHere.html>`_
   + `Tidy3D first walkthrough <../notebooks/Simulation.html>`_
   + `Introduction to Tidy3D working principles <../notebooks/Primer.html>`_

.. warning::

   RF simulation will be subject to new license requirements in the future.

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
   tidy3d.SurfaceImpedanceFitterParam
   tidy3d.HammerstadSurfaceRoughness
   tidy3d.HuraySurfaceRoughness

To model lossless metal, use ``PECMedium``.

.. code-block:: python

   # lossless metal
   my_pec = PECMedium()

To model lossy metal, use ``LossyMetalMedium``.

.. code-block:: python

   # lossy metal (conductivity in S/um)
   my_lossy_metal = LossyMetalMedium(conductivity=58, freq_range=(1e9, 10e9))

Note that the unit of ``conductivity`` is ``S/um`` and ``freq_range`` is ``Hz``. The ``LossyMetalMedium`` class implements the surface impedance boundary condition (SIBC). It can also accept surface roughness specifications using the Hammerstad or Huray models. Please refer to their respective documentation pages for details.

.. note::
   
   When modeling lossy metals, always be sure to check the skin depth --- if the skin depth is significant compared to the geometry size, then ``LossyMetalMedium`` may be not accurate. In that case, use a regular dispersive medium instead.
   

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.Medium
   tidy3d.plugins.dispersion.FastDispersionFitter

To model lossless dielectrics, use the regular ``Medium``.

.. code-block:: python

   # lossless dielectric
   my_lossless_dielectric = Medium(permittivity=2.2)

To model a lossy dielectric with constant loss tangent, use the ``constant_loss_tangent_model()`` method of the ``FastDispersionFitter`` utility class.

.. code-block:: python

   # lossy dielectric (constant loss tangent)
   my_lossy_dielectric = FastDispersionFitter.constant_loss_tangent_model(
       eps_real=4.4,
       loss_tangent=0.002,
       frequency_range=(1e9, 5e9)
   )

More advanced material models, including frequency dependence and anisotropy, are available in Tidy3D. For more details, please refer to the `EM Mediums <../mediums.html>`_ documentation page. 


.. seealso::

   For a more comprehensive discussion of the different EM mediums available in Tidy3D, please refer to the EM Mediums page:

   + `EM Mediums <../mediums.html>`_

~~~~

Layer-based Grid Refinement
---------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst
	      
   tidy3d.LayerRefinementSpec
   tidy3d.CornerFinderSpec



.. seealso::

   For a more comprehensive discussion of grid discretization in Tidy3D, please refer to the following page:

   + `Grid Discretization <../discretization.html>`_

   Example applications:

   + `Edge feed patch antenna benchmark <../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../notebooks/HybridMicrostripCPWBandpassFilter.html>`_


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


.. seealso::

   For more in-depth discussion and examples, please see the following learning center article:

   + `Using lumped elements in Tidy3D simulations <../notebooks/LinearLumpedElements.html>`_

   Example applications:

   + `Hybrid microstrip/co-planar waveguide bandpass filter <../notebooks/HybridMicrostripCPWBandpassFilter.html>`_
   + `Designing a power divider (part 3) <../notebooks/WPDHarmonicSuppression3.html>`_

	      
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


.. seealso::

   Example applications:

   + `Differential stripline benchmark <../notebooks/DifferentialStripline.html>`_
    

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

.. seealso::

   For more in-depth discussion and examples, please see the following learning center article:

   + `Introduction to Antenna Simulation <../notebooks/AntennaCharacteristics.html>`_

   Example applications:

   + `Edge feed patch antenna benchmark <../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_

~~~~

Backwards Compatibility
-----------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.run.run
   tidy3d.plugins.smatrix.run.create_batch
   tidy3d.plugins.smatrix.run.compose_modeler   
   tidy3d.plugins.smatrix.run.compose_modeler_data   
   tidy3d.plugins.smatrix.run.compose_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_component_modeler_data   
   tidy3d.plugins.smatrix.run.compose_component_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_terminal_modeler_data   
   tidy3d.plugins.smatrix.run.compose_terminal_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_simulation_data_index


~~~~
