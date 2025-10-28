.. _radiation_scattering:

Radiation & Scattering
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.DirectivityMonitor
   tidy3d.plugins.smatrix.DirectivityMonitorSpec
   tidy3d.plugins.microwave.RectangularAntennaArrayCalculator
   tidy3d.plugins.microwave.LobeMeasurer
   tidy3d.AntennaMetricsData

When modeling antennas or scattering problems, it is vital to analyze the radiated far-field. For such applications, the :class:`.DirectivityMonitor` should be used.

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
   )

The :class:`.DirectivityMonitor` should completely surround the structure of interest.

Alternatively, a :class:`.DirectivityMonitorSpec` can be used to create a specification for automatic generation of a :class:`.DirectivityMonitor` in the :class:`.TerminalComponentModeler`.

.. code-block:: python

   # Define directivity monitor spec
   my_directivity_monitor_spec = DirectivityMonitorSpec()

Once the monitor or monitor spec is defined, it should be added to the ``radiation_monitors`` option of the :class:`.TerminalComponentModeler`.

.. code-block:: python

   # Add directivity monitor to simulation
   my_tcm = TerminalComponentModeler(
       ...,
       radiation_monitors=[my_directivity_monitor, my_directivity_monitor_spec],
   )

Once the simulation is completed, the ``get_antenna_metrics_data()`` method of the :class:`.TerminalComponentModelerData` object is used to obtain the radiation metrics.

.. code-block:: python

   # Get radiation metrics
   my_antenna_metrics = my_tcm_data.get_antenna_metrics_data()

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

Each metric is in the form of an ``xarray.DataArray`` object that can be used for plotting, export, and further analysis. For examples of how these datasets can be manipulated, please refer to the notebooks in the "See also" section below.

The :class:`.LobeMeasurer` utility class can be used to analyze radiation pattern lobes.

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

Lobe characteristics such as direction, magnitude, and -3 dB beamwidth can be obtained for the main and side lobes. Additionally, the ``LobeMeasurer.plot()`` utility function adds main lobe beam direction and width markers to polar radiation plots.


.. seealso::

   For more in-depth discussion and examples, please see the following learning center article:

   + `Introduction to Antenna Simulation <../../notebooks/AntennaCharacteristics.html>`_

   Example applications:

   + `Edge feed patch antenna benchmark <../../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_

~~~~
