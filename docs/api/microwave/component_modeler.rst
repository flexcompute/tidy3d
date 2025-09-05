~~~~

TerminalComponentModeler and Data
---------------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.plugins.smatrix.TerminalPortDataArray

The ``TerminalComponentModeler`` is the core simulation object for 3D RF/microwave simulations in Tidy3D. Its primary function is to simulate the system over ``N`` number of ports and ``M`` number of frequency points, with the end result being a ``MxNxN`` S-parameter matrix.

.. code-block:: python

   my_tcm = TerminalComponentModeler(
       simulation=base_sim,
       ports=[port1, port2],
       freqs=my_frequencies,
       ...
   )

The key parts of a ``TerminalComponentModeler`` are:

* The ``simulation`` field defines the underlying Tidy3D `Simulation object <../simulation.html>`_. This base ``Simulation`` object contains information about the simulation domain such as structures, boundary conditions, grid specifications, and monitors. Note that sources should not be included in the base simulation, but rather in the ``ports`` field instead.
* The ``ports`` field defines the list of source excitations. These are commonly of type ``LumpedPort`` or ``WavePort``. The number of ports determines the number of batch jobs in the ``TerminalComponentModeler`` and the dimensionality of the S-parameter matrix.
* The ``freqs`` field defines the list of frequency points for the simulation.

More information and explanation for additional fields can be found in the documentation page for the ``TerminalComponentModeler``.

.. seealso::

   Please refer to the following example models to see the ``TerminalComponentModeler`` in action:

   + `Differential stripline benchmark <../notebooks/DifferentialStripline.html>`_
   + `Edge feed patch antenna benchmark <../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../notebooks/HybridMicrostripCPWBandpassFilter.html>`_

~~~~
