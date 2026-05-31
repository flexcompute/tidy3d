Charge |:zap:|
==============

The charge solver computes the steady-state (DC) and optional small-signal
AC response of semiconductor devices from the drift-diffusion equations. A
charge simulation is composed of a :class:`HeatChargeSimulation` that bundles
the device structures and their charge mediums, the boundary conditions
(voltage/current contacts), the monitors that record the fields and device
characteristics, and an analysis specification that drives the solve.

Charge simulations run on the **accelerated solver** by default. The analysis
specifications are listed under Analysis; tolerance and convergence guidance and
the accelerated-only features are covered in the
:ref:`Accelerated solver and convergence <charge-accelerated-solver>` section.

.. toctree::
    :hidden:

    simulation
    mediums
    boundary_conditions
    source
    monitor
    analysis
    convergence
    output_data


.. include:: /api/charge/simulation.rst
.. include:: /api/charge/mediums.rst
.. include:: /api/charge/boundary_conditions.rst
.. include:: /api/charge/source.rst
.. include:: /api/charge/monitor.rst
.. include:: /api/charge/analysis.rst

.. _charge-accelerated-solver:

.. include:: /api/charge/convergence.rst
.. include:: /api/charge/output_data.rst

.. seealso::

   Grid specifications for unstructured charge simulations are documented under :doc:`/api/mesh/discretization`.