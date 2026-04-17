.. currentmodule:: tidy3d

TerminalComponentModeler
------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   rf.TerminalComponentModeler
   rf.TerminalComponentModelerData
   rf.MicrowaveSMatrixData
   rf.TerminalPortDataArray
   rf.PortDataArray

The :class:`~tidy3d.rf.TerminalComponentModeler` is the core simulation object for 3D RF/microwave simulations in Tidy3D. Its primary function is to simulate the system over ``N`` number of ports and ``M`` number of frequency points, with the end result being a ``MxNxN`` S-parameter matrix.

.. code-block:: python

   my_tcm = TerminalComponentModeler(
       simulation=base_sim,
       ports=[port1, port2],
       freqs=my_frequencies,
       ...
   )

The key parts of a :class:`~tidy3d.rf.TerminalComponentModeler` are:

* The :class:`~tidy3d.Simulation` field defines the underlying Tidy3D `Simulation object <../simulation.html>`_. This base :class:`~tidy3d.Simulation` object contains information about the simulation domain such as structures, boundary conditions, grid specifications, and monitors. Note that sources should not be included in the base simulation, but rather in the ``ports`` field instead.
* The ``ports`` field defines the list of source excitations. These are commonly of type :class:`~tidy3d.rf.LumpedPort`, :class:`~tidy3d.rf.WavePort`, or :class:`~tidy3d.rf.TerminalWavePort`. The number of ports determines the number of batch jobs in the :class:`~tidy3d.rf.TerminalComponentModeler` and the dimensionality of the S-parameter matrix. **Note:** Port names cannot contain the '@' symbol (reserved for internal indexing).
* The ``freqs`` field defines the list of frequency points for the simulation.

More information and explanation for additional fields can be found in the documentation page for the :class:`~tidy3d.rf.TerminalComponentModeler`.

Workflow
~~~~~~~~

In order to submit the simulation, use ``web.upload()``, ``web.start()``, ``web.monitor()``, and ``web.load()``.

.. code-block:: python
   import tidy3d.web as web

   # Upload simulation and get cost estimate
   my_task_id = web.upload(my_tcm, task_name='my_task_name')

   # Run simulation
   web.start(my_task_id)

   # Monitor simulation
   web.monitor(my_task_id)

   # Load results after completion
   my_tcm_data = web.load(my_task_id)

   
Alternatively, use the ``web.run()`` method to perform all of the above in one single step.


.. code-block:: python
   import tidy3d.web as web

   # Upload, run simulation, and download data
   my_tcm_data = web.run(my_tcm, task_name='my_task_name', path='my/local/download/path')

If ``path`` is omitted, the component modeler results download to ``cm_data.hdf5`` in the
working directory by default.

To get the S-matrix from the results, use the ``smatrix()`` method of the :class:`~tidy3d.rf.TerminalComponentModelerData` object.

.. code-block:: python

   # Get S-matrix from results
   my_s_matrix = my_tcm_data.smatrix()

The S-matrix is stored as a :class:`~tidy3d.rf.MicrowaveSMatrixData` whose ``data`` property contains a :class:`~tidy3d.rf.TerminalPortDataArray` instance. To obtain a specific ``S_ij`` value, use the ``port_in`` and ``port_out`` coordinates with the corresponding port name. To obtain a specific frequency, use the ``f`` coordinate.

.. code-block:: python

   # Get return loss
   my_S11 = my_s_matrix.data.sel(port_in="my_port_1", port_out="my_port_1")

.. note::

   At this moment, Tidy3D uses the physics phase convention :math:`e^{-i\omega t}`. Other RF simulation software and texts may use the electrical engineering convention :math:`e^{i\omega t}`. This affects the sign of the imaginary part of calculated S-parameters and impedance values. To convert between the two, simply use the complex conjugation operation, e.g. ``np.conjugate()``.

.. seealso::

   To learn more about the web API workflow in Tidy3D, please refer to the following pages:

   + `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
   + `Web API documentation <../submit_simulations.html>`_

   To learn more about data post-processing and visualization, please refer to the following pages:

   + `Performing visualization of simulation data <../../notebooks/VizData.html>`_
   + `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

   Please refer to the following example models to see the :class:`~tidy3d.rf.TerminalComponentModeler` in action:

   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_
   + `Edge feed patch antenna benchmark <../../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../../notebooks/HybridMicrostripCPWBandpassFilter.html>`_

~~~~
