.. currentmodule:: tidy3d

Local Propagation
--------------------

:class:`.EMESimulation` exposes a pipeline for running EME end-to-end on a
local machine, combining the local mode solver with a set of per-element
stage methods for propagation.  The convenience entry point is
:meth:`.EMESimulation.propagate`; the staged methods give finer control for
iterative design, caching intermediates to HDF5, or cloud-parallelizing mode
solves.  The propagation methods require the optional ``tidy3d-extras``
package with the ``local_eme`` feature; see the :doc:`EME FAQ </faq/docs/eme>`
for recipes and prerequisites.

.. note::

   The local pipeline returns only the device S-matrix.  Any
   :class:`.EMESimulation` ``monitors`` — :class:`.EMEFieldMonitor`,
   :class:`.EMEModeSolverMonitor`, :class:`.EMECoefficientMonitor`, etc. — are
   dropped with a warning; run the simulation through the remote backend if
   you need monitor data.  :class:`.EMEFreqSweep` and anisotropic media in
   bent cells with ``bend_medium_frame="global"`` are also unsupported on this
   path.

.. autosummary::
   :toctree: ../_autosummary/

   EMESimulation.mode_simulations
   EMESimulation.propagate
   EMESimulation.compute_overlaps
   EMESimulation.propagate_from_overlaps
   EMESimulation.stage_cell_modes
   EMESimulation.compute_cell_overlap
   EMESimulation.compute_interface_overlap
   EMESimulation.compute_cell_smatrix
   EMESimulation.compute_interface_smatrix
   EMESimulation.compute_smatrix
