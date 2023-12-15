.. currentmodule:: tidy3d

Abstract Models
===============

These are some classes that are used to organize the tidy3d components, but aren't to be used directly in the code.  Documented here mainly for reference.


.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.components.base.Tidy3dBaseModel
   tidy3d.components.base_sim.simulation.AbstractSimulation
   tidy3d.components.boundary.AbsorberSpec
   tidy3d.Geometry
   tidy3d.components.geometry.base.Centered
   tidy3d.components.geometry.base.Planar
   tidy3d.components.geometry.base.Circular
   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.AbstractCustomMedium
   tidy3d.components.medium.DispersiveMedium
   tidy3d.components.medium.CustomDispersiveMedium
   tidy3d.components.structure.AbstractStructure
   tidy3d.components.source.SourceTime
   tidy3d.components.source.Source
   tidy3d.components.source.FieldSource
   tidy3d.components.source.CurrentSource
   tidy3d.components.source.ReverseInterpolatedSource
   tidy3d.components.source.AngledFieldSource
   tidy3d.components.source.PlanarSource
   tidy3d.components.source.DirectionalSource
   tidy3d.components.source.BroadbandSource
   tidy3d.components.source.VolumeSource
   tidy3d.components.source.Pulse
   tidy3d.components.monitor.Monitor
   tidy3d.components.monitor.FreqMonitor
   tidy3d.components.monitor.TimeMonitor
   tidy3d.components.monitor.AbstractFieldMonitor
   tidy3d.components.monitor.AbstractFluxMonitor
   tidy3d.components.monitor.PlanarMonitor
   tidy3d.components.monitor.AbstractFieldProjectionMonitor
   tidy3d.components.grid.grid_spec.GridSpec1d
   tidy3d.components.data.data_array.DataArray
   tidy3d.components.data.monitor_data.MonitorData
   tidy3d.components.data.monitor_data.AbstractFieldProjectionData
   tidy3d.components.data.monitor_data.ElectromagneticFieldData
   tidy3d.components.data.monitor_data.AbstractMonitorData
   tidy3d.components.data.dataset.AbstractFieldDataset
   tidy3d.components.data.dataset.FieldDataset
   tidy3d.components.data.dataset.FieldTimeDataset
   tidy3d.components.data.dataset.ModeSolverDataset