.. currentmodule:: tidy3d

S-Matrix Component Modelers Plugin
==================================

This plugin computes S-parameters (scattering parameters) for photonic devices --- waveguides, splitters, filters, and the like --- with the **ModalComponentModeler**, which builds the S-matrix from mode overlap integrals.

.. warning::

   Breaking changes were introduced in ``v2.10.0``, please see the :ref:`smatrix_migration` guide for help migrating your code.

.. seealso::

   For RF and microwave S-parameters, use the terminal component modeler in Flexcompute RF (``flexcompute.rf.tidy3d``). See :doc:`../microwave`.

Photonics Component Modelers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   plugins.smatrix.ModalComponentModeler
   plugins.smatrix.ModalComponentModelerData
   plugins.smatrix.Port
   plugins.smatrix.ModalPortDataArray

.. _smatrix_migration:

.. include:: /api/plugins/smatrix_migration.rst

Further Details
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   plugins.smatrix.AbstractComponentModeler
   plugins.smatrix.data.base.AbstractComponentModelerData
   SimulationMap
   SimulationDataMap
