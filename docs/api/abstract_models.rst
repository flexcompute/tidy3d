.. currentmodule:: tidy3d

Base Classes
============

These are some of the classes that are used to organize Tidy3D components, but aren't to be used directly by the user. They are documented here mainly for user reference of inherited components.


.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Geometry
   tidy3d.Tidy3dBaseModel
   tidy3d.components.geometry.base.Centered
   tidy3d.components.geometry.base.Circular
   tidy3d.components.geometry.base.Planar
   tidy3d.components.geometry.base.SimplePlaneIntersection
   tidy3d.components.geometry.polyslab.ComplexPolySlabBase
   tidy3d.components.structure.AbstractStructure
   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.AbstractCustomMedium
   tidy3d.components.medium.DispersiveMedium
   tidy3d.components.medium.CustomDispersiveMedium
   tidy3d.components.medium.AnisotropicMediumFromMedium2D
   tidy3d.components.source.time.SourceTime
   tidy3d.components.source.time.Pulse
   tidy3d.components.source.base.Source
   tidy3d.components.source.field.FieldSource
   tidy3d.components.source.field.PlanarSource
   tidy3d.components.source.field.VolumeSource
   tidy3d.components.source.field.DirectionalSource
   tidy3d.components.source.field.BroadbandSource
   tidy3d.components.source.field.AngledFieldSource
   tidy3d.components.source.current.CurrentSource
   tidy3d.components.source.current.ReverseInterpolatedSource
   tidy3d.Monitor
   tidy3d.components.tcad.monitors.abstract.HeatChargeMonitor
   tidy3d.components.monitor.FreqMonitor
   tidy3d.components.monitor.TimeMonitor
   tidy3d.components.monitor.AbstractFieldMonitor
   tidy3d.components.monitor.AbstractFluxMonitor
   tidy3d.components.monitor.PlanarMonitor
   tidy3d.components.monitor.SurfaceIntegrationMonitor
   tidy3d.components.monitor.AbstractFieldProjectionMonitor
   tidy3d.components.lumped_element.LumpedElement
   tidy3d.components.lumped_element.RectangularLumpedElement
   tidy3d.components.grid.grid_spec.GridSpec1d
   tidy3d.components.data.sim_data.AbstractYeeGridSimulationData
   tidy3d.components.data.sim_data.SimulationData
   tidy3d.components.boundary.AbsorberSpec
   tidy3d.components.tcad.boundary.abstract.HeatChargeBC
   tidy3d.components.subpixel_spec.AbstractSubpixelAveragingMethod
   tidy3d.components.data.data_array.DataArray
   tidy3d.components.data.dataset.FieldDataset
   tidy3d.components.data.dataset.FieldTimeDataset
   tidy3d.components.data.dataset.ModeSolverDataset
   tidy3d.components.data.monitor_data.ElectromagneticFieldData
   tidy3d.components.data.monitor_data.MonitorData
   tidy3d.components.spice.analysis.ac.AbstractSSACAnalysis
   tidy3d.components.bc_placement.AbstractBCPlacement
   tidy3d.components.boundary.AbstractABCBoundary
   tidy3d.components.grid.grid_spec.AbstractAutoGrid
   tidy3d.components.material.tcad.charge.AbstractChargeMedium
   tidy3d.components.material.tcad.heat.AbstractHeatMedium
   tidy3d.components.medium.AbstractSurfaceRoughness
   tidy3d.components.mode_spec.AbstractModeSpec
   tidy3d.components.base_sim.monitor.AbstractMonitor
   tidy3d.components.monitor.AbstractGaussianOverlapMonitor
   tidy3d.components.monitor.AbstractMediumPropertyMonitor
   tidy3d.components.monitor.AbstractModeMonitor
   tidy3d.components.microwave.monitor.MicrowaveModeMonitorBase
   tidy3d.components.microwave.path_integrals.specs.base.AbstractAxesRH
   tidy3d.components.source.field.AbstractAngularSpec
   tidy3d.components.tcad.data.sim_data.AbstractHeatChargeSimulationData
   tidy3d.components.tcad.doping.AbstractDopingBox
   tidy3d.plugins.autograd.invdes.filters.AbstractFilter
   tidy3d.plugins.invdes.base.InvdesBaseModel
   tidy3d.plugins.invdes.design.AbstractInverseDesign
   tidy3d.plugins.microwave.array_factor.AbstractAntennaArrayCalculator
   tidy3d.plugins.smatrix.ports.base_lumped.AbstractLumpedPort
   tidy3d.plugins.smatrix.ports.base_terminal.AbstractTerminalPort
