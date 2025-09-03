Microwave and RF |:satellite:|
==============================

Overview
--------

.. warning::

   RF simulation will be subject to new license requirements in the future.

This page consolidates Tidy3D features related to microwave and RF simulation. While microwave/RF and optical simulations have many properties in common, there are some differences in the typical RF user workflow that deserve special consideration.

The following sections discuss:

* `TerminalComponentModeler and Data`_: The core simulation object in microwave/RF models
* `RF Materials`_: Typical material types in microwave/RF simulation
* `Layer-based Grid Refinement`_: Automated grid refinement strategy for planar structures (e.g. printed circuit boards)
* `Lumped Port and Elements`_: Lumped excitations and terminations
* `Wave Port`_: Port excitation based on modal fields
* `Radiation and Scattering`_: Useful features for antenna and scattering problems

.. seealso::

   If you are completely new to Tidy3D, we recommend first checking out the following beginner resources:

   + `Quickstart <../notebooks/StartHere.html>`_
   + `Tidy3D first walkthrough <../notebooks/Simulation.html>`_
   + `Introduction to Tidy3D working principles <../notebooks/Primer.html>`_

~~~~

TerminalComponentModeler and Data
---------------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.plugins.smatrix.TerminalPortDataArray

The ``TerminalComponentModeler`` is the core simulation object for 3D RF/microwave simulations in Tidy3D. Its primary function is to simulate the system over ``N`` number of ports and ``M`` number of frequencie points, with the end result being a ``MxNxN`` S-parameter matrix.

.. code-block:: python

   my_tcm = TerminalComponentModeler(
       simulation=base_sim,
       ports=[port1, port2],
       freqs=my_frequencies,
       ...
   )

The key parts of a ``TerminalComponentModeler`` are as follows:

* The ``simulation`` parameter defines the underlying Tidy3D ``Simulation`` object. This base ``Simulation`` object contains information about the simulation domain such as structures, boundary conditions, grid specifications, monitors and so on. Note that sources are not defined in the base simulation.
* The ``ports`` parameter defines a list of all the possible system excitations. These are commonly of type ``LumpedPort`` or ``WavePort``. The number of ports determines the number of batch jobs in the ``TerminalComponentModeler`` and the dimensionality of the S-parameter matrix.
* The ``freqs`` parameter defines the list of frequency points for the simulation.

More information and explanation for additional parameters can be found in the documentation page for the ``TerminalComponentModeler``.

.. seealso::

   Please refer to the following example models to see the ``TerminalComponentModeler`` in action:

   + `Differential stripline benchmark <../notebooks/DifferentialStripline.html>`_
   + `Edge feed patch antenna benchmark <../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../notebooks/HybridMicrostripCPWBandpassFilter.html>`_

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

The ``PECMedium`` and ``LossyMetalMedium`` classes can be used to model metallic materials. 

.. code-block:: python

   # lossless metal
   my_pec = PECMedium()

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
   tidy3d.GridRefinement

The ``LayerRefinementSpec`` class allows the user to specify automated refinement within a layered region, for instance, the metallic trace plane of a printed circuit board. The grid will be automatically refined near any metallic corners and edges in that layer. 

.. code-block:: python

   # Define layer refinement spec
   my_layer_refinement_spec = LayerRefinementSpec(
       axis=2,  # layer normal axis
       center=(0, 0, 0),  # layer center
       size=(3, 2, 0.1),  # layer size
       min_steps_along_axis=2,  # minimum number of grid points along normal axis
       corner_refinement=td.GridRefinement(dl=100, num_cells=2)  # metal corner refinement specification
   )

   # Add layer refinement spec to overall grid specification
   my_grid_spec = GridSpec(
       ...,
       layer_refinement_specs = [my_layer_refinement_spec]
   )

More than one ``LayerRefinementSpec`` is permitted. In addition to manually defining the ``center`` and ``size`` of the ``LayerRefinementSpec``, one can alternatively use the ``from_bounds()``, ``from_layer_bounds()``, or ``from_structures()`` convenience methods.

.. code-block:: python

   my_layer_refinement_spec_2 = LayerRefinementSpec.from_structures(
       structures=[my_planar_structure],   # position, size, and axis automatically determined based on structure
       ...
   )

Note that different ``LayerRefinementSpec`` instances are required for structures on different physical layers.

.. seealso::

   For more explanation and examples, please refer to the following pages:

   + `Grid Discretization <../discretization.html>`_
   + `Automatic mesh refinement in layered structures <../notebooks/LayerRefinement.html>`_  

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

The ``LumpedPort`` feature represents a planar, uniform current excitation with a fixed impedance termination.

.. code-block:: python

   # Define a lumped port
   my_port_1 = LumpedPort(
       name='My Port 1',   
       center=(0,0,0),
       size=(0, port_width, port_height),
       voltage_axis=2,   # z-axis aligned excitation
       impedance=50,   # port impedance
   )

The ``LumpedPort`` can be 1D (line) or 2D (plane). For 2D, only axis-aligned planes are supported at this time. The port ``impedance`` value can be complex.

.. note::

   Lumped ports and elements are fundamentally approximations and thus should only be used when the port/element size is much smaller than the wavelength of interest (typically ``lambda/10``). For more accurate results, especially when the port is adjacent to an intentional waveguide, consider using the ``WavePort`` excitation instead. 

The ``CoaxialLumpedPort`` represents an analytical coaxial field source.

.. code-block:: python

   # Define coaxial lumped port
   my_coaxial_port_1 = CoaxialLumpedPort(
       name='My Coaxial Port 1',
       center=(0,0,0),
       inner_diameter=1000,   # inner diameter in um
       outer_diameter=2000,   # outer diameter in um
       normal_axis=0,   # normal axis to port plane
       direction='+',   # direction of signal along normal axis
       impedance=50,   # port impedance
   )

.. note::

   Because the ``CoaxialLumpedPort`` injects an analytical field source, the structure connected to this port must match the physical port dimensions. Any deviation will result in signal reflection and potential inaccuracies. One common source of this issue is in imported geometries with faceted cylinders. 

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.LumpedResistor
   tidy3d.CoaxialLumpedResistor
   tidy3d.LinearLumpedElement
   tidy3d.RLCNetwork
   tidy3d.AdmittanceNetwork

For a simple resistive lumped element, use ``LumpedResistor``.

.. code-block:: python

   my_resistor = LumpedResistor(
       name='My resistor',
       center=(0,0,0),
       size=(0, element_width, element_height),
       voltage_axis=2,   # z-axis aligned 
       resistance=50,   # real-valued impedance
   )

For more complicated RLC networks, use the general ``LinearLumpedElement`` class.

.. code-block:: python

   my_lumped_element = LinearLumpedElement(
       name='My lumped element',
       center=(0,0,0),
       size=(0, element_width, element_height),
       voltage_axis=2,   # z-axis aligned
       network=RLCNetwork(resistance=50, inductance=1e-9)  # RLC network
   )

All lumped elements should be added to the ``lumped_elements`` parameter of the base ``Simulation`` instance.

.. code-block:: python

   my_simulation = Simulation(
       lumped_elements=[my_resistor, my_lumped_element],
       ...,
   )


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

The ``WavePort`` represents a modal source port. The port mode is first calculated in the 2D mode solver, then injected into the 3D simulation. The ``WavePort`` is also automatically terminated with a modal absorbing boundary ``ModeABCBoundary`` that perfectly absorbs the outgoing mode. Any non-matching modes are subject to PEC reflection. 

.. code-block:: python

   my_wave_port_1 = WavePort(
       center=(0,0,0),
       size=(port_width, port_height, 0),
       name='My Wave Port 1',
       direction='+',  # direction of signal
       mode_spec=ModeSpec(target_neff=1.5),  # specification for mode solver
       current_integral=my_current_integral,  # current integration curve for port impedance calculation
   )

Most parameters are self explanatory. Some additional notes:

* ``mode_spec`` is used to specify the effective index search value for the mode solver
* ``current_integral`` and/or ``voltage_integral`` are used to specify the integration paths for port impedance calculation. If only one of the two is specified, then the port power is also used (automatically determined). 

If it is desired to only solve for the 2D port mode, one can use the ``to_mode_solver()`` convenience method to generate a ``ModeSolver`` simulation object. 

.. code-block:: python

   # Define a mode solver from the wave port
   my_mode_solver = my_wave_port_1.to_mode_solver(
       simulation=base_sim,   # base Simulation object
       freqs=my_frequencies,   # frequencies for 2D mode solver
   )

   # Execute mode solver
   my_mode_data = web.run(my_mode_solver, task_name='mode solver')
		

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.microwave.VoltageIntegralAxisAligned
   tidy3d.plugins.microwave.CurrentIntegralAxisAligned
   tidy3d.plugins.microwave.CustomVoltageIntegral2D
   tidy3d.plugins.microwave.CustomCurrentIntegral2D
   tidy3d.plugins.microwave.AxisAlignedPathIntegral
   tidy3d.plugins.microwave.CustomPathIntegral2D
   tidy3d.plugins.microwave.ImpedanceCalculator

The classes above are used to define the voltage/current integration paths for impedance calculation.

.. code-block:: python

   # Define voltage integration line
   my_voltage_integral = VoltageIntegralAxisAligned(
       center=(0,0,0),  # center of integration line
       size=(5, 0, 0),  # length of integration line
       sign='+',  # sign of integral
   )

   # Define current integration loop
   my_current_integral = CurrentIntegralAxisAligned(
       center=(0,0,0),  # center of integration loop
       size=(20, 20, 0),  # size of integration loop
       sign='+', # sign of integral (should match wave port direction)
   )

In addition to being used in the ``WavePort`` definition, the current/voltage integration objects can also be manually performed on arbitrary EM field data (2D and 3D). This is most commonly used in conjunction with the ``ImpedanceCalculator`` to calculate the line impedance of a 2D mode.

.. code-block:: python

   # Define impedance calculator
   my_Z_calculator = ImpedanceCalculator(
       voltage_integral = my_voltage_integral,
       current_integral = my_current_integral,
   )

   # Calculate impedance of 2D mode
   Z_mode = my_Z_calculator.compute_impedance(my_mode_data)

As before, only one of the two integration paths (voltage or current) are strictly necessary. This determines the convention used to calculate the impedance (PI, PV, or VI). 
   
.. seealso::

   For more information, please see the following articles:

   + `Computing the characteristic impedance of transmission lines <../notebooks/CharacteristicImpedanceCalculator.html>`_

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

For radiation and scattering type problems, it is frequently desired to calculate the radiation/scattering pattern. One should use the ``DirectivityMonitor``.

.. code-block:: python

   # Define angular coordinates
   # Theta is the elevation angle relative to global +z axis
   # Phi is the azimuthal angle relative to global +x axis
   my_theta = np.linspace(0, np.pi, 91)
   my_phi = np.linspace(0, 2*np.pi, 181)

   # Define directivity monitor
   my_directivity_monitor = DirectivityMonitor(
       center=(0,0,0),
       size=(100, 100, 100),
       freqs=my_frequencies,
       phi=my_phi,
       theta=my_theta,
       name='My radiation monitor',
       far_field_approx=True,
   )

The ``DirectivityMonitor`` should completely surround the structure of interest. The ``far_field_approx`` flag can be used to set whether the far-field approximation is used (default ``True``).

Once the monitor is defined, it should be added to the ``radiation_monitors`` option of the ``TerminalComponentModeler``.

.. code-block:: python

   # Add directivity monitor to simulation
   my_tcm = TerminalComponentModeler(
       ...,
       radiation_monitor=[my_directivity_monitor],
   )

Once the simulation is completed, the ``get_antenna_metrics_data()`` method of the ``TerminalComponentModeler`` is used to obtain the radiation metrics.

.. code-block:: python

   # Get radiation metrics
   my_antenna_metrics = my_tcm.get_antenna_metrics_data()

   # Get individual metrics
   my_directivity = my_antenna_metrics.directivity
   my_gain = my_antenna_metrics.gain   
   my_radiation_efficiency = my_antenna_metrics.radiation_efficiency
   my_reflection_efficiency = my_antenna_metrics.reflection_efficiency
   my_realized_gain = my_antenna_metrics.realized_gain
   my_supplied_power = my_antenna_metrics.supplied_power
   my_radiated_power = my_antenna_metrics.radiated_power
   my_radiation_intensity = my_antenna_metrics.radiation_intensity
   my_axial_ratio = my_antenna_metrics.axial_ratio
   my_left_pol = my_antenna_metrics.left_polarization
   my_right_pol = my_antenna_metrics.right_polarization

The ``LobeMeasurer`` utility class can be used to calculate radiation lobe statistics.

.. code-block:: python

   # Define lobe measurer
   my_lobes = LobeMeasurer(
       angle=phi,   # Angular axis of interest
       radiation_pattern=my_gain,   # Radiation pattern to measure
   )

   # Get lobe characteristics
   my_lobe_measures = my_lobes.lobe_measures
   my_main_lobe = my_lobes.main_lobe
   my_side_lobes = my_lobes.side_lobe


.. seealso::

   For more in-depth discussion and examples, please see the following learning center article:

   + `Introduction to Antenna Simulation <../notebooks/AntennaCharacteristics.html>`_

   Example applications:

   + `Edge feed patch antenna benchmark <../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_

~~~~
