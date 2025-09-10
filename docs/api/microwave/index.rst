Microwave & RF |:satellite:|
==============================

Overview
--------

.. warning::

   RF simulations will be subject to new license requirements in the future.

This page consolidates Tidy3D features related to microwave and RF simulation. While microwave/RF and optical simulations have many properties in common, there are some differences in the typical RF user workflow that deserve special consideration.

The following sections discuss:

* `TerminalComponentModeler`_: The core simulation object in microwave/RF models
* `RF Materials`_: Typical material types in microwave/RF simulation
* `Layer-based Grid Refinement`_: Automated grid refinement strategy for planar structures (e.g. printed circuit boards)
* `Lumped Port and Elements`_: Lumped excitations and circuit elements
* `Wave Port`_: Port excitation based on modal fields
* `Radiation and Scattering`_: Useful features for antenna and scattering problems

.. seealso::

   If you are completely new to Tidy3D, we recommend first checking out the following beginner resources:

   + `Quickstart <../../notebooks/StartHere.html>`_
   + `Tidy3D first walkthrough <../../notebooks/Simulation.html>`_
   + `Introduction to Tidy3D working principles <../../notebooks/Primer.html>`_

.. include:: /api/microwave/component_modeler.rst
.. include:: /api/microwave/material.rst
.. include:: /api/discretization/layer.rst
.. include:: /api/microwave/ports/lumped.rst
.. include:: /api/microwave/ports/wave.rst
.. include:: /api/microwave/radiation_scattering.rst
