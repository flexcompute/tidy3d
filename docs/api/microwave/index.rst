Microwave & RF |:satellite:|
==============================

.. toctree::
    :hidden:

    component_modeler
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

* `TerminalComponentModeler`_: The core simulation object in microwave/RF models
* `RF Materials Models`_: Typical material types in microwave/RF simulation
* `RF Materials Library`_: The RF material library contains various dispersive models for real-world RF materials.
* `Layer-based Grid Refinement`_: Automated grid refinement strategy for planar structures (e.g. printed circuit boards)
* `Path Integrals`_: Tools for computing voltage and current from electromagnetic fields
* `Impedance Calculator`_: Post-processing tool for impedance calculation from electromagnetic fields
* `RF Mode Analysis`_: Performing RF-specific mode analysis, like computing the characteristic impedance of transmission line modes
* `Lumped Port & Elements`_: Lumped excitations and circuit elements
* `Wave Port`_: Port excitation based on modal fields or terminal modal fields
* `Radiation & Scattering`_: Useful features for antenna and scattering problems
* `RF Output Data`_: Data containers for microwave simulation results

.. seealso::

   If you are completely new to Tidy3D, we recommend first checking out the following beginner resources:

   + `Quickstart <../../notebooks/StartHere.html>`_
   + `Tidy3D first walkthrough <../../notebooks/Simulation.html>`_
   + `Introduction to Tidy3D working principles <../../notebooks/Primer.html>`_

.. include:: /api/microwave/component_modeler.rst
.. include:: /api/microwave/material.rst
.. include:: /api/microwave/rf_material_library.rst
.. include:: /api/discretization/layer.rst
.. include:: /api/microwave/path_integrals.rst
.. include:: /api/microwave/impedance_calculator.rst
.. include:: /api/microwave/mode_solver.rst
.. include:: /api/microwave/ports/lumped.rst
.. include:: /api/microwave/ports/wave.rst
.. include:: /api/microwave/radiation_scattering.rst
.. include:: /api/microwave/output_data.rst
