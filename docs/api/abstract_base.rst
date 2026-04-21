.. currentmodule:: tidy3d

Abstract Base Classes
=====================

Base classes that represent abstractions of common components. Provide inherited functionality.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   components.base_sim.data.sim_data.AbstractSimulationData
   components.base_sim.data.monitor_data.AbstractMonitorData
   components.tcad.data.monitor_data.abstract.HeatChargeMonitorData
   components.base_sim.monitor.AbstractMonitor
   components.base_sim.simulation.AbstractSimulation
   components.base_sim.source.AbstractSource
   components.data.dataset.AbstractFieldDataset
   components.data.dataset.AuxFieldTimeDataset
   components.data.dataset.ElectromagneticFieldDataset
   components.data.dataset.ElectromagneticSurfaceFieldDataset
   components.data.monitor_data.ElectromagneticSurfaceFieldData
   components.data.data_array.AbstractSpatialDataArray
   components.data.monitor_data.AbstractFieldData
   components.data.monitor_data.AbstractFieldProjectionData
   components.parameter_perturbation.AbstractPerturbation
   components.parameter_perturbation.AbstractPerturbation
   components.medium.AbstractCustomMedium
   components.medium.AbstractMedium
   components.microwave.base.MicrowaveBaseModel
   components.simulation.AbstractYeeGridSimulation
   components.structure.AbstractStructure
   components.time.AbstractTimeDependence
