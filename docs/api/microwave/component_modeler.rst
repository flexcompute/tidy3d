.. _TerminalComponentModeler:

TerminalComponentModeler
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.plugins.smatrix.MicrowaveSMatrixData
   tidy3d.plugins.smatrix.TerminalPortDataArray
   tidy3d.plugins.smatrix.PortDataArray

The :class:`.TerminalComponentModeler` is the core simulation object for 3D RF/microwave simulations in Tidy3D. Its primary function is to simulate the system over ``N`` number of ports and ``M`` number of frequency points, with the end result being a ``MxNxN`` S-parameter matrix.

.. code-block:: python

   my_tcm = TerminalComponentModeler(
       simulation=base_sim,
       ports=[port1, port2],
       freqs=my_frequencies,
       ...
   )

The key parts of a :class:`.TerminalComponentModeler` are:

* The :class:`.Simulation` field defines the underlying Tidy3D `Simulation object <../simulation.html>`_. This base :class:`.Simulation` object contains information about the simulation domain such as structures, boundary conditions, grid specifications, and monitors. Note that sources should not be included in the base simulation, but rather in the ``ports`` field instead.
* The ``ports`` field defines the list of source excitations. These are commonly of type :class:`LumpedPort` or :class:`.WavePort`. The number of ports determines the number of batch jobs in the :class:`.TerminalComponentModeler` and the dimensionality of the S-parameter matrix.
* The ``freqs`` field defines the list of frequency points for the simulation.

More information and explanation for additional fields can be found in the documentation page for the :class:`.TerminalComponentModeler`. In order to submit the simulation, use ``tidy3d.web.upload()``, ``tidy3d.web.start()``, ``tidy3d.web.monitor()``, and ``tidy3d.web.load()``.

.. code-block:: python

   # Upload simulation and get cost estimate
   my_task_id = tidy3d.web.upload(my_tcm, task_name='my_task_name')

   # Run simulation
   tidy3d.web.start(my_task_id)

   # Monitor simulation
   tidy3d.web.monitor(my_task_id)

   # Load results after completion
   my_tcm_data = tidy3d.web.load(my_task_id)

   
Alternatively, use the ``tidy3d.web.run()`` method to perform all of the above in one single step.


.. code-block:: python

   # Upload, run simulation, and download data
   my_tcm_data = tidy3d.web.run(my_tcm, task_name='my_task_name', path='my/local/download/path')

To get the S-matrix from the results, use the ``smatrix()`` method of the :class:`.TerminalComponentModelerData` object.

.. code-block:: python

   # Get S-matrix from results
   my_s_matrix = my_tcm_data.smatrix()

The S-matrix is stored as a :class:`.MicrowaveSMatrixData` whose ``data`` property contains a :class:`.TerminalPortDataArray` instance. To obtain a specific ``S_ij`` value, use the ``port_in`` and ``port_out`` coordinates with the corresponding port name. To obtain a specific frequency, use the ``f`` coordinate.

.. code-block:: python

   # Get return loss
   my_S11 = my_s_matrix.data.sel(port_in="my_port_1", port_out="my_port_1")

.. note::

   At this moment, Tidy3D uses the physics phase convention :math:`e^{-i\omega t}`. Other RF simulation software and texts may use the electrical engineering convention :math:`e^{i\omega t}`. This affects the calculated S-parameters and impedance values. To convert between the two, simply use the complex conjugation operation, e.g. ``np.conjugate()``.

To access simulation data for a given port excitation, use the ``data`` attribute of the :class:`.TerminalComponentModelerData`.

.. code-block:: python

   # Get simulation data for a given port "my_port"
   sim_data = my_tcm_data.data["my_port"]

   # Get monitor data from a given monitor "my_monitor"
   my_monitor_data = sim_data["my_monitor"]

The ``data`` attribute holds the simulation data in a dictionary with the respective port name as keys. The data for each monitor can then be accessed from the simulation data object using the monitor name as the dictionary key.

.. seealso::

   To learn more about the web API workflow in Tidy3D, please refer to the following pages:

   + `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
   + `Web API documentation <../submit_simulations.html>`_

   To learn more about data post-processing and visualization, please refer to the following pages:

   + `Performing visualization of simulation data <../../notebooks/VizData.html>`_
   + `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

   Please refer to the following example models to see the :class:`.TerminalComponentModeler` in action:

   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_
   + `Edge feed patch antenna benchmark <../../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../../notebooks/HybridMicrostripCPWBandpassFilter.html>`_

~~~~
