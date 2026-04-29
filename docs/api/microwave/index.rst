Microwave & RF |:satellite:|
==============================

.. toctree::
    :hidden:

    component_modeler
    smatrix_definitions
    material
    rf_material_library
    path_integrals
    impedance_calculator
    mode_solver
    ports/lumped
    ports/wave
    radiation_scattering
    output_data

Overview
--------

.. warning::

   RF simulations and functionality will require new license requirements in an upcoming release. All RF-specific classes are now available within the sub-package 'tidy3d.rf'.

.. warning::

   Breaking changes were introduced in ``v2.10.0``, please see the migration guide for help migrating your code.

   + `Migration Guide <microwave_migration.html>`_

This page consolidates Tidy3D features related to microwave and RF simulation. While microwave/RF and optical simulations have many properties in common, there are some differences in the typical RF user workflow that deserve special consideration.

The following sections discuss:

* :doc:`component_modeler` — The core simulation object in microwave/RF models
* :doc:`smatrix_definitions` — Explanation of pseudo-wave, power-wave, and symmetric pseudo-wave formulations
* :doc:`material` — Typical material types in microwave/RF simulation
* :doc:`rf_material_library` — Dispersive models for real-world RF materials
* :ref:`layer_grid_refinement` — Automated grid refinement for planar structures (e.g. PCBs)
* :doc:`path_integrals` — Tools for computing voltage and current from electromagnetic fields
* :doc:`impedance_calculator` — Post-processing tool for impedance calculation
* :doc:`mode_solver` — RF-specific mode analysis and characteristic impedance
* :doc:`ports/lumped` — Lumped excitations and circuit elements
* :doc:`ports/wave` — Port excitation based on modal or terminal modal fields
* :doc:`radiation_scattering` — Antenna and scattering analysis tools
* :doc:`output_data` — Data containers for microwave simulation results

.. seealso::

   If you are completely new to Tidy3D, we recommend first checking out the following beginner resources:

   + `Quickstart <../../notebooks/StartHere.html>`__
   + `Tidy3D first walkthrough <../../notebooks/Simulation.html>`__
   + `Introduction to Tidy3D working principles <../../notebooks/Primer.html>`__
