.. currentmodule:: tidy3d

Base Classes
============

These are some of the classes that are used to organize Tidy3D components, but aren't to be used directly by the user. They are documented here mainly for user reference of inherited components.


.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   Geometry
   Tidy3dBaseModel
   components.geometry.base.Centered
   components.geometry.base.Circular
   components.geometry.base.Planar
   components.geometry.base.SimplePlaneIntersection
   components.geometry.polyslab.ComplexPolySlabBase
   components.structure.AbstractStructure
   components.medium.AbstractMedium
   components.medium.AbstractCustomMedium
   components.medium.DispersiveMedium
   components.medium.CustomDispersiveMedium
   components.medium.AnisotropicMediumFromMedium2D
   components.source.time.SourceTime
   components.source.time.Pulse
   components.source.base.Source
   components.source.field.FieldSource
   components.source.field.PlanarSource
   components.source.field.VolumeSource
   components.source.field.DirectionalSource
   components.source.field.BroadbandSource
   components.source.field.AngledFieldSource
   components.source.current.CurrentSource
   components.source.current.ReverseInterpolatedSource
   Monitor
   components.tcad.monitors.abstract.HeatChargeMonitor
   components.monitor.FreqMonitor
   components.monitor.TimeMonitor
   components.monitor.AbstractFieldMonitor
   components.monitor.AbstractFluxMonitor
   components.monitor.PlanarMonitor
   components.monitor.SurfaceIntegrationMonitor
   components.monitor.AbstractFieldProjectionMonitor
   components.lumped_element.LumpedElement
   components.lumped_element.RectangularLumpedElement
   components.grid.grid_spec.GridSpec1d
   components.data.sim_data.AbstractYeeGridSimulationData
   components.data.sim_data.SimulationData
   components.boundary.AbsorberSpec
   components.tcad.boundary.abstract.HeatChargeBC
   components.subpixel_spec.AbstractSubpixelAveragingMethod
   components.data.data_array.DataArray
   components.data.dataset.FieldDataset
   components.data.dataset.FieldTimeDataset
   components.data.dataset.ModeSolverDataset
   components.data.monitor_data.ElectromagneticFieldData
   components.data.monitor_data.MonitorData
   components.spice.analysis.ac.AbstractSSACAnalysis
   components.bc_placement.AbstractBCPlacement
   components.boundary.AbstractABCBoundary
   components.grid.grid_spec.AbstractAutoGrid
   components.material.tcad.charge.AbstractChargeMedium
   components.material.tcad.heat.AbstractHeatMedium
   components.medium.AbstractSurfaceRoughness
   components.mode_spec.AbstractModeSpec
   components.base_sim.monitor.AbstractMonitor
   components.monitor.AbstractGaussianOverlapMonitor
   components.monitor.AbstractMediumPropertyMonitor
   components.monitor.AbstractModeMonitor
   components.microwave.monitor.MicrowaveModeMonitorBase
   components.microwave.path_integrals.specs.base.AbstractAxesRH
   components.source.field.AbstractAngularSpec
   components.tcad.data.sim_data.AbstractHeatChargeSimulationData
   components.tcad.doping.AbstractDopingBox
   plugins.autograd.invdes.filters.AbstractFilter
   plugins.invdes.base.InvdesBaseModel
   plugins.invdes.design.AbstractInverseDesign
   plugins.microwave.array_factor.AbstractAntennaArrayCalculator
   plugins.smatrix.ports.base_lumped.AbstractLumpedPort
   plugins.smatrix.ports.base_terminal.AbstractTerminalPort
