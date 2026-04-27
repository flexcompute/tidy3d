EME |:rainbow:|
===============

Eigenmode expansion (EME) computes the scattering matrix of a structure by
solving for local eigenmodes in each cell of an :class:`.EMESimulation` and
propagating them through the device.  It is a frequency-domain, modal
technique well-suited to guided-wave structures that are translation-invariant
or slowly varying along a single propagation axis — tapers, mode converters,
adiabatic bends, directional couplers, DBRs, and long periodic stacks.  For
strongly radiating, broadband-transient, or nonlinear problems, prefer FDTD.

The primary entry point is :func:`.web.run`:

.. code-block:: python

   import tidy3d as td
   import tidy3d.web as web

   sim = td.EMESimulation(
       size=(2.0, 2.0, 10.0),
       structures=[my_waveguide],
       freqs=[td.C_0 / 1.55],
       axis=2,
       eme_grid_spec=td.EMEUniformGrid(
           num_cells=5,
           mode_spec=td.EMEModeSpec(num_modes=4),
       ),
   )
   sim_data = web.run(sim, task_name="my_eme")
   smatrix = sim_data.smatrix

The reference pages below cover the simulation object and its monitors, grid
specification, propagation sweeps, and output data.  The
:doc:`local_propagation` page documents an advanced workflow for running EME
end-to-end on a local machine.  For usage guidance (setup, convergence
testing, bent waveguides, periodic structures, port / S-matrix basis), see
the :doc:`EME FAQ </faq/docs/eme>`.

.. toctree::
    :hidden:

    simulation
    monitor
    grid
    sweep
    output_data
    local_propagation


.. include:: /api/eme/simulation.rst
.. include:: /api/eme/monitor.rst
.. include:: /api/eme/grid.rst
.. include:: /api/eme/sweep.rst
.. include:: /api/eme/output_data.rst
.. include:: /api/eme/local_propagation.rst
