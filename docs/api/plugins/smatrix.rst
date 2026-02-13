.. currentmodule:: tidy3d

S-Matrix Component Modelers Plugin
==================================

This plugin provides component modelers for computing S-parameters (scattering parameters) for both **photonics** and **RF/microwave** applications. The plugin supports:

* **Photonics**: Modal component modelers for photonic devices (waveguides, splitters, filters, etc.)
* **RF/Microwave**: Terminal component modelers for microwave circuits and antennas (available in the :mod:`~tidy3d.rf` subpackage as well)

.. warning::

   Breaking changes were introduced in ``v2.10.0``, please see the :ref:`smatrix_migration` guide for help migrating your code.

Photonics Component Modelers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For photonics applications, use the **ModalComponentModeler** which computes modal S-parameters based on mode overlap integrals.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.ModalComponentModeler
   tidy3d.plugins.smatrix.ModalComponentModelerData
   tidy3d.plugins.smatrix.Port
   tidy3d.plugins.smatrix.ModalPortDataArray

RF/Microwave Component Modelers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. seealso::
   
   For classes related to microwave/RF modeling, please refer to the main `Microwave and RF <../microwave/index.html>`_ page and `tidy3d.rf` sub-package. 


For RF and microwave applications, use the **TerminalComponentModeler** (available in ``tidy3d.rf`` as well) which computes terminal-based S-parameters.

.. warning::

   RF simulations will require new license requirements in an upcoming release. All RF-specific classes are available in the ``tidy3d.rf`` subpackage.


.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.plugins.smatrix.LumpedPort
   tidy3d.plugins.smatrix.CoaxialLumpedPort
   tidy3d.rf.WavePort
   tidy3d.rf.MicrowaveSMatrixData
   tidy3d.rf.TerminalPortDataArray
   tidy3d.rf.PortDataArray

.. seealso::
   For complete RF/microwave documentation, see:

      * `RF Simulation Workflow <../microwave/component_modeler.html>`_ - Main RF simulation workflow
      * `Microwave & RF Documentation <../microwave/index.html>`_ - Comprehensive RF features

.. include:: /api/plugins/smatrix_migration.rst

Further Details
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.AbstractComponentModeler
   tidy3d.plugins.smatrix.data.base.AbstractComponentModelerData
   tidy3d.SimulationMap
   tidy3d.SimulationDataMap
