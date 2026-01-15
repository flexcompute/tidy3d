# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed
- Fixed docstrings with missing Notes sections causing spurious parameters in Sphinx documentation.

## [2.10.1] - 2026-01-14

### Added
- Added priority attribute to `TopologyDesignRegion` to enable manual control of overlapping structures.
- `to_mat_file()` method is now available on `ModeSimulationData` and `HeatChargeSimulationData` for exporting results to MATLAB `.mat` files.
- Added autograd support for diagonal `AnisotropicMedium` and `CustomAnisotropicMedium` with diagonal permittivity tensor.
- Added support of numpy 2.4
- Added validation to `DCVoltageSource` that warns when duplicate voltage values are detected in the `voltage` array, including treating `0` and `-0` as the same value.

### Changed
- For `HeatChargeSimulation` objects, the `plot` function now adds the simulation boundary conditions.

### Fixed
- Fixed `AutoImpedanceSpec` validation to check path intersections against all conductors, not just filtered ones, as well as the mode plane bounds.
- Fixed `WavePort` validation so invalid `mode_spec` errors are no longer masked by a `KeyError`.
- Fixed adjoint gradients being treated as zero due to scale-dependent `np.allclose(..., atol=1e-8)` checks, which could skip adjoint simulations and return zero gradients.
- Fixed adjoint setup crashing when traced monitor outputs produce no adjoint sources, returning no adjoint simulations instead.
- Fixed autograd `interpn` compatibility with newer SciPy versions by avoiding dtype coercion during interpolation setup.
- Fixed interpolation handling for permittivity and conductivity gradients in CustomMedium.
- Restored original batch-load logging by suppressing per-task “Loading simulation…” messages. 
- Fixed output range of `tidy3d.plugins.invdes.FilterAndProject` to be between 0 and 1.
- Cropped adjoint monitor sizes in 2D simulations to planar geometry intersection.
- Fixed `Batch.download()` silently succeeding when background downloads fail (e.g., gzip extraction errors).
- Handling of zero values when using `sim_data.plot_field` with `scale=dB`.
- Fixed `intersections_plane` method in `PolySlab`, which sometimes missed vertices for planes coincident with `PolySlab` side faces.
- Fixed `http_interceptor` crash on non-dict JSON responses

## [2.10.0] - 2025-12-18

### Added
- Added rectangular and radial taper support to `RectangularAntennaArrayCalculator` for phased array amplitude weighting; refactored array factor calculation for improved clarity and performance. 
- Selective simulation capabilities to `TerminalComponentModeler` via `run_only` and `element_mappings` fields, allowing users to run fewer simulations and extract only needed scattering matrix elements.
- Added KLayout plugin, with DRC functionality for running design rule checks in `plugins.klayout.drc`. Supports running DRC on GDS files as well as `Geometry`, `Structure`, and `Simulation` objects.
- Added "mil" and "in" (inch) units to `plot_length_units`.
- Objective functions that involve running `tidy3d.plugins.smatrix.ComponentModeler` can be differentiated with autograd.
- Access field decay values in `SimulationData` via `sim_data.field_decay` as `TimeDataArray`.
- Added ability to set first-order absorbing boundary conditions on simulation domain boundaries using either `ABCBoundary` or `ModeABCBoundary` classes.
- Added `frame` field to `ModeSource` (default: `None`) and `WavePort` (default: `PECFrame()`). Setting this to `PECFrame(length=...)` automatically places a thin PEC frame of specified length around the source/port. The automatically created frames can be inspected using `Simulation._finalized` property.
- Added `InternalAbsorber` class for placing first-order absorbing boundary conditions on planes inside the simulation domain. Internal absorbers are automatically wrapped in a PEC frame with a backing PEC plate on the non-absorbing side.
- Added `absorber` field (default: `True`) to `WavePort` for automatically placing an absorber behind the port.
- Added `conjugated_dot_product` field in `ModeMonitor` (default: `True`) and `WavePort` (default: `False`) to allow selecting the conjugated or non-conjugated dot product for mode decomposition.
- Support for gradients with respect to the `conductivity` of a `CustomMedium`.
- Added `VerticalNaturalConvectionCoeffModel`, a model for heat transfer due to natural convection from a vertical plate. It can be used in `ConvectionBC` to compute the heat transfer coefficient from fluid properties, using standard Nusselt number correlations for both laminar and turbulent flow.
- Added `BroadbandModeABCSpec` class for setting broadband absorbing boundary conditions that can absorb waveguide modes over a specified frequency range using a pole-residue pair model.
- `Scene.plot_3d()` method to make 3D rendering of scene.
- Added native web and batch support to run `ModalComponentModeler` and `TerminalComponentModeler` workflows.
- Added `SimulationMap` and `SimulationDataMap` immutable dictionary-like containers for managing collections of simulations and results.
- Added `TerminalComponentModelerData`, `ComponentModelerData`, `MicrowaveSMatrixData`, and introduced multiple DataArrays for modeler workflow data structures.
- Added autograd support for dispersive material models: `Sellmeier`, `Drude`, `Lorentz`, `Debye` and their custom medium variants.
- Added check and exception for NaN data in the adjoint pipeline to raise issue to user before adjoint source creation failure.
- Added autograd support for `TerminalComponentModeler` and `ModalComponentModeler`.
- Added `initialize_params_from_simulation` to `tidy3d.plugins.autograd.invdes` to initialize topology design regions from an underlying simulation geometry.
- Added autograd support for sidewall angles in `td.Cylinder` and `td.PolySlab`.
- New `MediumMonitor` that returns both permittivity and permeability profiles.
- Task names are now optional when using `run(sim)` or `Job`. When running multiple jobs (via `run_async` or `Batch`), you can also provide simulations as a list without specifying task names. The previous dictionary-based format with explicit task names is still supported.
- Enabled lazy loading of data via `web.load(..., lazy=True)`. When used, this returns a lightweight proxy object holding a reference to the data. On first access to any field or method, the proxy transparently loads the full object (same as with the default lazy=False).
- A new type of doping box has been introduced, `CustomDoping` which accepts a `SpatialDataArray` to define doping concentration. Unlike in the case where a `SpatialDataArray`, custom doping defined with `CustomDoping` have additive behavior, i.e., one can add other doping on top. This deprecates the `SpatialDataArray` as direct input for `N_a` and `N_d`.
- Non-isothermal Charge simulations are now available. One can now run this type of simulations by using the `SteadyChargeDCAnalysis` as the `analysis_spec` of a `HeatChargeSimulation`. This type of simulations couple the heat equation with the drift-diffusion equations which allow to account for self heating behavior.
- Because non-isothermal Charge simulations are now supported, new models for the effective density of states and bandgap energy have been introduced. These models are the following: `ConstantEffectiveDOS`, `IsotropicEffectiveDOS`, `MultiValleyEffectiveDOS`, `DualValleyEffectiveDOS`.
- Added the Hurkx model for direct band-to-band tunneling `HurkxDirectBandToBandTunneling`.
- Added Selberherr's model for impact ionization `SelberherrImpactIonization`.
- Added S-parameter de-embedding to `TerminalComponentModelerData`, enabling recalculation with shifted reference planes.
- Added optional automatic extrusion of structures intersecting with a `WavePort` via the new `extrude_structures` field, ensuring mode sources, absorbers, and PEC frames are fully contained.
- Added support for `tidy3d-extras`, an optional plugin that enables more accurate local mode solving via subpixel averaging.
- Added support for `symlog` and `log` scale plotting in `Scene.plot_eps()` and `Scene.plot_structures_property()` methods. The `symlog` scale provides linear behavior near zero and logarithmic behavior elsewhere, while 'log' is a base 10 logarithmic scale.
- Added `LowFrequencySmoothingSpec` and `ModelerLowFrequencySmoothingSpec` for automatic smoothing of mode monitor data at low frequencies where DFT sampling is insufficient.
- Added `MicrowaveModeSpec` for RF-specific mode information with customizable characteristic impedance calculations.
- Added `MicrowaveModeMonitor` and `MicrowaveModeSolverMonitor` for microwave and RF mode analysis with transmission line data.
- Added `MicrowaveModeData` and `MicrowaveModeSolverData` extending mode solver results with characteristic impedance, voltage coefficients, and current coefficients.
- Added `AutoImpedanceSpec` for automatic transmission line impedance calculation based on simulation geometry.
- Added `CustomImpedanceSpec` for user-defined voltage and current path specifications in impedance calculations.
- Added voltage integral specification classes: `AxisAlignedVoltageIntegralSpec` and `Custom2DVoltageIntegralSpec`.
- Added current integral specification classes: `AxisAlignedCurrentIntegralSpec`, `CompositeCurrentIntegralSpec`, and `Custom2DCurrentIntegralSpec`.
- `sort_spec` in `ModeSpec` allows for fine-grained filtering and sorting of modes. This also deprecates `filter_pol`. The equivalent usage for example to `filter_pol="te"` is `sort_spec=ModeSortSpec(filter_key="TE_polarization", filter_reference=0.5)`. `ModeSpec.track_freq` has also been deprecated and moved to `ModeSortSpec.track_freq`.
- Added `custom_source_time` parameter to `ComponentModeler` classes (`ModalComponentModeler` and `TerminalComponentModeler`), allowing specification of custom source time dependence.
- Validation for `run_only` field in component modelers to catch duplicate or invalid matrix indices early with clear error messages.
- Introduced a profile-based configuration manager with TOML persistence and runtime overrides exposed via `tidy3d.config`.
- Added support of `os.PathLike` objects as paths like `pathlib.Path` alongside `str` paths in all path-related functions.
- Added local simulation result caching to avoid rerunning identical simulations. Enable and configure it through `td.config.local_cache` (size limits, cache directory). Use CLI commands `tidy3d cache {info, list, clear}` to inspect or clear the cache.
- Added `DirectivityMonitorSpec` for automated creation and configuration of directivity radiation monitors in `TerminalComponentModeler`.
- Added multimode support to `WavePort` in the smatrix plugin, allowing multiple modes to be analyzed per port.
- Added support for `.lydrc` files for design rule checking in the `klayout` plugin.
- Added a Gaussian inverse design filter option with autograd gradients and complete padding mode coverage.
- Added support for argument passing to DRC file when running checks with `DRCRunner.run(..., drc_args={key: value})` in klayout plugin.
- Added support for `nonlinear_spec` in `CustomMedium` and `CustomDispersiveMedium`.
- `tidy3d.plugins.design.DesignSpace.run(..., fn_post=...)` now accepts a `priority` keyword to propagate vGPU queue priority to all automatically batched simulations.
- Introduced `BroadbandPulse` for exciting simulations across a wide frequency spectrum.
- Added `interp_spec` in `ModeSpec` to allow downsampling and interpolation of waveguide modes in frequency.
- Added warning if port mesh refinement is incompatible with the `GridSpec` in the `TerminalComponentModeler`.
- Various types, e.g. different `Simulation` or `SimulationData` sub-classes, can be loaded from file directly with `Tidy3dBaseModel.from_file()`.
- Added `interp_spec` in `EMEModeSpec` to enable faster multi-frequency EME simulations. Note that the default is now `ModeInterpSpec.cheb(num_points=3, reduce_data=True)`; previously the computation was repeated at all frequencies.
- Added `smoothed_projection` for topology optimization of completely binarized designs.
- Added more RF-specific mode characteristics to `MicrowaveModeData`, including propagation constants (alpha, beta, gamma), phase/group velocities, wave impedance, and automatic mode classification with configurable polarization thresholds in `MicrowaveModeSpec`.
- Introduce `tidy3d.rf` namespace to consolidate all RF classes.
- Added support for custom colormaps in `plot_field`.
- Added `symmetrize_mirror`, `symmetrize_rotation`, `symmetrize_diagonal` functions to the autograd plugin. They can be used for enforcing symmetries in topology optimization.
- Klayout plugin automatically finds klayout installation path at common locations.
- Added autograd support for `TriangleMesh`, allowing gradient computation with respect to mesh vertices for inverse design.
- Added `EMESimulation.store_coeffs` to store coefficients from the EME solver, including mode overlaps, interface S matrices, and effective propagation indices.
- Get cell-related information from violation markers in `DRCResults` and `DRCViolation` to the klayout plugin: Use for example `DRCResults.violations_by_cell` to group them.
- `pixel_exact` option for `to_gds` export of simulations and structures. If this option is set, any custom medium will be exported as rectangular pixels according to the specification of the medium coordinates.

### Changed
- Adaptive minimum spacing for `PolySlab` integration is now wavelength relative and a minimum discretization is set for computing gradients for cylinders.
- The `TerminalComponentModeler` defaults to the pseudo wave definition of scattering parameters. The new field `s_param_def` can be used to switch between either pseudo or power wave definitions.
- Restructured the smatrix plugin with backwards-incompatible changes for a more robust architecture. Notably, `ComponentModeler` has been renamed to `ModalComponentModeler` and internal web API methods have been removed. Please see our migration guide for details on updating your workflows.
- Prevent small bandwidth sources from being created in `TerminalComponentModeler` when modeler frequencies are close together.
- `Simulation.epsilon` now samples off-diagonal elements of fully tensorial permittivity at grid *boundaries*, to be consistent with the how these enter the FDTD simulation. This means that all off-diagonal components are now sampled at the same locations.
- Propagate `verbose` to `start` function in web API.
- `LayerRefinementSpec` defaults to assuming structures made of different materials are interior-disjoint for more efficient mesh generation.
- Improved performance of antenna metrics calculation by utilizing cached wave amplitude calculations instead of recomputing wave amplitudes for each port excitation in the `TerminalComponentModelerData`.
- Changed hashing method in `Tidy3dBaseModel` from sha256 to md5.
- Removed validators on limiting the number of geometries in a ClipOperation geometry.
- Improved the speed of computing `Box` shape derivatives when used inside a `GeometryGroup`.
- All RF and microwave specific components now inherit from `MicrowaveBaseModel`.
- `DirectivityMonitor` now forces `far_field_approx` to `True`, which was previously configurable.
- Unified run submission API: `web.run(...)` is now a container-aware wrapper that accepts a single simulation or arbitrarily nested containers (`list`, `tuple`, `dict` values) and returns results in the same shape.
- `web.Batch(ComponentModeler)` and `web.Job(ComponentModeler)` native support
- Simulation data of batch jobs are now automatically downloaded upon their individual completion in `Batch.run()`, avoiding waiting for the entire batch to reach completion.
- Port names in `ModalComponentModeler` and `TerminalComponentModeler` can no longer include the `@` symbol.
- Improved speed of convolutions for large inputs.
- Default value of `EMEModeSpec.interp_spec` is `ModeInterpSpec.cheb(num_points=3, reduce_data=True)` for faster multi-frequency EME simulations.
- Default value of `num_sweep` in `EMECoefficientMonitor` is now `None`, recording all sweep indices.
- Maximum number of frequencies in an `EMESimulation` is now larger if mode interpolation via `ModeInterpSpec` is used.
- Removed validator that would warn if `PerturbationMedium` values could become numerically unstable, since an error will anyway be raised if this actually happens when the medium is converted using actual perturbation data.
- Improved performance of adjoint gradient computation for `PolySlab` and `Cylinder` geometries through vectorized sidewall patch collection (~4x speedup).
- Added optional `max_results` handling to the kLayout `DRCLoader` and `DRCRunner` and speeded up parsing of big result sets.

### Fixed
- Bug in `TerminalComponentModeler.get_antenna_metrics_data` when port amplitudes are set to zero.
- Added missing `solver_version` keyword argument to `run_async`.
- Fixed `interpn` data array method to be compatible with extrapolation outside of data array coordinates.
- Fixed `overlap_sort` to use the same value of the `conjugated_dot_product` field in `ModeMonitor`, and added the `conjugated_dot_product` field to `ModeSolver` and `ModeSimulation`.
- Stricter validation for `bend_radius` in mode simulations, preventing the bend center from coinciding with the simulation boundary.
- Prevent autograd adjoint simulations from reusing out-of-range `normalize_index` values by defaulting their normalization to the first adjoint source when needed.
- Subtasks validation errors from `web.upload(ComponentModeler)` previously were not being propagated to users, and hung without response.
- Ensured the legacy `Env` proxy mirrors `config.web` profile switches and preserves API URL.
- More robust `Sellmeier` and `Debye` material model, and prevent very large pole parameters in `PoleResidue` material model.
- Bug in `WavePort` when more than one mode is requested in the `ModeSpec`.
- Solver error for named 2D materials with inhomogeneous substrates.
- In `Tidy3dBaseModel` the hash (and cached `.json_string`) are now sensitive to changes in `.attrs`.
- More accurate frequency range for ``GaussianPulse`` when DC is removed.
- Bug in `TerminalComponentModelerData.get_antenna_metrics_data()` where `WavePort` mode indices were not properly handled. Improved docstrings and type hints to make the usage clearer.
- Improved type hints for `Tidy3dBaseModel`, so that all derived classes will have more accurate return types.
- More robust method for suppressing RF license warnings during tests.
- Fixed frequency sampling of `TransmissionLineDataset` within `MicrowaveModeData` when using group index calculation.
- Fixed DRC parsing for quoted categories in klayout plugin.
- Removed mode solver warnings about evaluating permittivity of a `Medium2D`.
- Maximum number of grid points in an EME simulation is now based solely on transverse grid points. Maximum number of EME cells is unchanged.
- Fixed handling of polygons with holes (interiors) in `subdivide()` function. The function now properly converts polygons with interiors into `PolySlab` geometries using subtraction operations with `GeometryGroup`.
- Fix to `outer_dot` when frequencies stored in the data were not in increasing order. Previously, the result would be provided with re-sorted frequencies, which would not match the order of the original data.
- Fixed bug where an extra spatial coordinate could appear in `complex_flux` and `ImpedanceCalculator` results.
- Fixed normal for `Box` shape gradient computation to always point outward from boundary which is needed for correct PEC handling.
- Fixed `Box` gradients within `GeometryGroup` where the group intersection boundaries were forwarded.
- Fixed `Box` gradients to use automatic permittivity detection for inside/outside permittivity.
- Improved degenerate mode handling in the mode solver to ensure modes respect the bi-orthogonality condition.

### Removed
- Removed deprecated `use_complex_fields` parameter from `TwoPhotonAbsorption` and `KerrNonlinearity`. Parameters `beta` and `n2` are now real-valued only, as is `n0` if specified.
- Removed the Adjoint plugin.
- Deprecation of Python 3.9 support.

### Breaking Changes
- Native web and batch support of `ModalComponentModeler` and `TerminalComponentModeler` required changing these classes schemas considerably. 
  - Equivalent behaviour is available using a different workflow.
  - `folder_name`, `verbose`, `path_dir`, `batch_cached`, `verbose`, `callback_url` fields have been removed. They are accessible through the `tidy3d.web` interface.
- Edge singularity correction at PEC and lossy metal edges defaults to `True`.
- `angle_threshold` in `CornerFinderSpec` now defaults to `pi/4`.
- `WavePort` has been refactored to use `MicrowaveModeSpec`. The fields `voltage_integral`, and `current_integral` have been removed. 
- Impedance specifications are now defined in `MicrowaveModeSpec.impedance_specs`. Please see our [migration guide](https://docs.flexcompute.com/projects/tidy3d/en/v2.10.0/api/microwave/microwave_migration.html) for details on updating your code.

### Planned Deprecation
**Note: These changes only affect the microwave and smatrix plugins.**
- Renamed path integral classes for improved consistency. Please see our [migration guide](https://docs.flexcompute.com/projects/tidy3d/en/v2.10.0/api/microwave/microwave_migration.html) for details on updating your code. Old class naming is aliased to the new classes for 2.10.
  - `VoltageIntegralAxisAligned` → `AxisAlignedVoltageIntegral`
  - `CurrentIntegralAxisAligned` → `AxisAlignedCurrentIntegral`
  - `CustomPathIntegral2D` → `Custom2DPathIntegral`
  - `CustomVoltageIntegral2D` → `Custom2DVoltageIntegral`
  - `CustomCurrentIntegral2D` → `Custom2DCurrentIntegral`
  - Path integral and impedance calculator classes have been refactored and moved from the microwave plugin into Tidy3D components. They are now publicly exported via the top-level package `__init__.py`.


## [2.9.3]

### Added

### Changed

### Fixed
- Fixed `pyproject.toml` and `tidy3d.__version__` mismatch.


## [2.9.2]

### Added

### Changed
- Improved Nexus platform configuration support and instantiation. Includes improvements to use the new `tidy3d configure ... --nexus-url <url>` CLI functionality.

### Fixed


## [2.9.1] - 2025-08-13

### Changed
- Validate mode solver object for large number of grid points on the modal plane.

### Fixed
- Fixed missing amplitude factor and handling of negative normal direction case when making adjoint sources from `DiffractionMonitor`.
- Improved the robustness of batch jobs. The batch state, including all `task_ids`, is now saved to `batch.hdf5` immediately after upload. This fixes an issue where an interrupted batch (e.g., due to a kernel crash or network loss) would be unrecoverable.
- Fixed warning for running symmetric adjoint simulations by port to not trigger when there is a single port.
- Bug in `CoaxialLumpedPort` where source injection is off when the `normal_axis` is not `z`.
- Validation of `freqs` in the `ComponentModeler` and `TerminalComponentModeler`.
- Calculation of voltage and current in the `WavePort`, when one type of path integral is supplied and the transmission line mode is lossy.
- Polygon vertices cleanup in `ClipOperation.intersections_plane`.
- Removed sources from `sim_inf_structure` simulation object in `postprocess_adj` to avoid source and background medium validation errors.
- Revert overly restrictive validation of `freqs` in the `ComponentModeler` and `TerminalComponentModeler`.
- Fixed `ElectromagneticFieldData.to_zbf()` to support single frequency monitors and apply the correct flattening order.


## [2.9.0] - 2025-08-04

### Added
- Fields `convex_resolution`, `concave_resolution`, and `mixed_resolution` in `CornerFinderSpec` can be used to take into account the dimensions of autodetected convex, concave, or mixed geometric features when `dl_min` is automatically inferred during automatic grid generation.
- `LayerRefinementSpec` now supports automatic thin gap meshing through fields `gap_meshing_iters` and `dl_min_from_gap_width`.
- Added `eps_lim` keyword argument to `Simulation.plot_eps()` for manual control over the permittivity color limits.
- Added `thickness` parameter to `LossyMetalMedium` for computing surface impedance of a thin conductor.
- `priority` field in `Structure` and `MeshOverrideStructure` for setting the behavior in structure overlapping region. When its value is `None`, the priority is automatically determined based on the material property and simulation's `structure_priority_mode`.
- Automatically apply `matplotlib` styles when importing `tidy3d` which can be reverted via the `td.restore_matplotlib_rcparams()` function.
- Added `TriangleMesh.from_height_expression` class method to create a mesh from an analytical height function defined on a 2D grid and `TriangleMesh.from_height_grid` class method to create a mesh from height values sampled on a 2D grid.
- Added heat sources with custom spatial dependence. It is now possible to add a `SpatialDataArray` as the `rate` in a `HeatSource`.
- Added Transient Heat simulations. It is now possible to run transient Heat simulations. This can be done by specifying `analysis_spec` of `HeatChargeSimulation` object as `UnsteadyHeatAnalysis`. 
- A `num_grid_cells` field to `WavePort`, which ensures that there are 5 grid cells across the port. Grid refinement can be disabled by passing `None` to `num_grid_cells`.
- Implemented `FreqRange` utility class for frequency/wavelength handling with constructor methods `from_freq_interval()`, `from_wavelength()`, and `from_wvl_interval()`. 
- Add support for `np.unwrap` in `tidy3d.plugins.autograd`.
- Add Nunley variant to germanium material library based on Nunley et al. 2016 data.
- Add `PointDipole.sources_from_angles()` that constructs a list of `PointDipole` objects needed to emulate a dipole oriented at a user-provided set of polar and azimuthal angles.
- Added `priority` parameter to `web.run()` and related functions to allow vGPU users to set task priority (1-10) in the queue.
- `EMEFieldMonitor` now supports `interval_space`.
- `Simulation.precision` option allows to select `"double"` precision for very high-accuracy results. Note that this is very rarely needed, and doubles the simulation computational weight and correspondingly FlexCredit cost.
- Added material type `PMCMedium` for perfect magnetic conductor.
- `ModeSimulation.plot()` method that plots the mode simulation plane by default, or the containing FDTD simulation if any of ``x``, ``y``, or ``z`` is passed. 
- Enable singularity correction at PEC and lossy metal edges.
- New `VolumeMesher` simulation type and associated `VolumeMeshMonitor` and `VolumeMesherData`, which can be used to run the unstructured meshing for a `HeatChargeSimulation` separately before running the solver.
- The current validator for Conduction simulations has been modified so that it checks that in a Conduction simulation there is at least one structure defined with `ChargeConductorMedium` in the `charge` field of a `MultiPhysicsMedium`. This is necessary to ensure simulations are properly set up in the back-end since it relies on the `conductivity` field of `ChargeConductorMedium` and not that of `Medium`.
- Access field decay values in `SimulationData` via `sim_data.field_decay` as `TimeDataArray`.

### Changed
- Relaxed bounds checking of path integrals during `WavePort` validation.
- Internal adjoint helper methods are now prefixed with an underscore to separate them from the public API.
- Drop the dependency on `gdspy`, which has been unmaintained for over two years. Interfaces previously relying on `gdspy` now use its maintained successor, `gdstk`, with equivalent functionality.
- Small (around 1e-4) numerical precision improvements in EME solver.
- Adjoint source frequency width is adjusted to decay sufficiently before zero frequency when possible to improve accuracy of simulation normalization when using custom current sources.
- Change `VisualizationSpec` validator for checking validity of user specified colors to only issue a warning if matplotlib is not installed instead of an error.
- Improved performance of `tidy3d.web.delete_old()` for large folders.
- `tidy3d.plugins.autograd.interpolate_spline()` and `tidy3d.plugins.autograd.add_at()` can now be called with keyword arguments during tracing.
- Zero-size dimensions automatically receive periodic boundary conditions instead of raising an error.
- Set `ModeSpec` precision to `double` by default for more accurate mode solver results. Does not apply to `EMEModeSpec`, where the `auto` precision is still default for speed and cost.
- Switched to an analytical gradient calculation for spatially-varying pole-residue models (`CustomPoleResidue`).
- Significantly improved performance of the `tidy3d.plugins.autograd.grey_dilation` morphological operation and its gradient calculation. The new implementation is orders of magnitude faster, especially for large arrays and kernel sizes.
- Warnings are now generated (instead of errors) when instantiating `PML`, `StablePML`,
or `Absorber` classes (or when invoking `pml()`, `stable_pml()`, or `absorber()` functions)
with fewer layers than recommended.
- Warnings and error messages originating from `Structure`, `Source`, or `Monitor` classes now refer to problematic objects by their user-supplied `name` attribute, alongside their index.
- File downloads are atomic. Interruptions or failures during download will no longer result in incomplete files.
- Warnings are now generated (instead of errors) when instantiating `PML`, `StablePML`, or `Absorber` classes (or when invoking `pml()`, `stable_pml()`, or `absorber()` functions) with fewer layers than recommended.
- `Simulation.subsection` can no longer take `symmetry` as an argument - the symmetry is always taken from the original simulation.
- If a mode simulation is crossing a symmetry plane of the larger simulation domain, but the mode plane is not symmetric, a warning is issued that it will be expanded symmetrically. Previously this warning only happened during the solver run.
- Enhanced `PolySlab` and `Cylinder` gradient computation via adaptive field sampling along geometry boundaries instead of fixed-grid center sampling.
- Shape derivatives have been sped up significantly, especially for simulations containing many structures in a `GeometryGroup`.
- By default, batch downloads will skip files that already exist locally. To force re-downloading and replace existing files, pass the `replace_existing=True` argument to `Batch.load()`, `Batch.download()`, or `BatchData.load()`.
- The `BatchData.load_sim_data()` function now overwrites any previously downloaded simulation files (instead of skipping them).
- Tighter `TOL_EIGS` used for mode solver, since `scipy` sometimes failed to find modes.

### Fixed

- Fixed bug in broadband adjoint source creation when forward simulation had a pulse amplitude greater than 1 or a nonzero pulse phase.
- Fixed shaping of `CustomMedium` gradients when permittivity data includes a frequency dimension with multiple entries.
- Bug in contains check for `LumpedElement`, which should allow the case of a `LumpedElement` touching the simulation boundaries.
- Bug when generating a grid with snapping points near the simulation boundaries.
- Fixed field colocation in `EMEModeSolverMonitor`.
- Internal interpolation errors with some versions of `xarray` and `numpy`.
- If `ModeSpec.angle_rotation=True` for a mode object, validate that the structure rotation can be successfully done. Also, error if the medium cannot be rotated (e.g. anisotropic or custom medium), which would previously have just produced wrong results.
- Characteristic impedance calculations in the `ImpedanceCalculator` using definitions that rely on flux, which were giving incorrect results for lossy transmission lines.
- Validation for `CustomGridBoundaries`, which was previously allowing unsorted arrays and arrays with less than two entries.
- `DiffractionMonitor` results to apply finite grid field corrections for higher precision when comparing e.g. to `FluxMonitor` computations of total power.
- Bug when validating the grid resolution near `CoaxialLumpedPort`.
- Arrow lengths are now scaled consistently in the X and Y directions, and their lengths no longer exceed the height of the plot window.
- Plots of objects defined by shape intersection logic will no longer display thin line artifacts.
- Fixed incorrect gradient computation in PyTorch plugin (`to_torch`) for functions returning multi-element arrays.
- `MonitorData.get_amplitude()` no longers multiplies by a factor of `1j` and now directly returns the complex value of the data.
- `EMESimulationData.port_modes_tuple` is now symmetry-expanded.
- Fixed `Medium2D` validation error message when invalid data is passed to `ss`.
- The phase of the amplitudes of a `DiffractionMonitor` was correctly centered such that the origin is at the monitor center.
- Giving opposite boundaries different names no longer causes a symmetry validator failure.
- Fixed issue with parameters in `InverseDesignResult` sometimes being outside of the valid parameter range.
- Fixed performance regression for multi-frequency adjoint calculations.
- Disallow `EMEFieldMonitor` in EME simulations with `EMELengthSweep`.
- Fixed bug in adjoint postprocessing frequency batching that was causing gradients to be zero or incorrect. The error was surfacing when selecting a subset of the monitor frequencies in the objective function.



## [2.8.5] - 2025-07-07

### Fixed
- Bug in `PlaneWave` defined with a negative `angle_theta` which would lead to wrong injection.
- `GaussianBeam` and `AstigmaticGaussianBeam` default `num_freqs` reset to 1 (it was set to 3 in v2.8.0) and a warning is issued for a broadband, angled beam for which `num_freqs` may not be sufficiently large.
- Set the maximum `num_freqs` to 20 for all broadband sources (we have been warning about the introduction of this hard limit for a while).
- Solver error for EME simulations with bends, introduced when support for 2D EME simulations was added.
- Fixed handling of symmetry when creating adjoint field sources and added warning when broken up adjoint simulations do not have the same symmetry as the forward simulation.

## [2.8.4] - 2025-05-15

### Added
- The method `Geometry.reflected` can be used to create a reflected copy of any geometry off a plane. As for other transformations, for efficiency, `reflected` `PolySlab` directly returns an updated `PolySlab` object rather than a `Transformed` object, except when the normal of the plane of reflection has a non-zero component along the slab axis, in which case `Transformed` is still returned.
- Validation check for unit error in grid spacing.
- Validation that when symmetry is imposed along a given axis, the boundary conditions on each side of the axis are identical.

### Changed
- Supplying autograd-traced values to geometric fields (`center`, `size`) of simulations, monitors, and sources now logs a warning and falls back to the static value instead of erroring.
- Attempting to differentiate server-side field projections now raises a clear error instead of silently failing.
- Improved error message and handling when attempting to load a non-existent task ID.
- `ClipOperation` now fails validation if traced fields are detected.
- Warn if more than 20 frequencies are used in EME, as this may lead to slower or more expensive simulations.
- EME now supports 2D simulations.
- 'EMESimulation' now supports 'PermittivityMonitor'.

### Fixed
- Fixed issue with `CustomMedium` gradients where other frequencies would wrongly contribute to the gradient.
- Fixed bug when computing `PolySlab` bounds in plotting functions.

## [2.8.3] - 2025-04-24

### Added
- Ability to select payment option when submitting jobs from the Python client.
- Periodic repetition of EME subgrids via `num_reps` or `EMEPeriodicitySweep`.
- Methods `EMEExplicitGrid.from_structures` and `EMECompositeGrid.from_structure_groups` to place EME cell boundaries at structure bounds.
- 'ModeSimulation' now supports 'PermittivityMonitor'.
- Classmethod `from_frequency_range` in `GaussianPulse` for generating a pulse whose amplitude in the frequency_range [fmin, fmax] is maximized, which is particularly useful for running broadband simulations.
- Differentiable function `td.plugins.autograd.interpolate_spline` for 1D linear, quadratic, and cubic spline interpolation, supporting differentiation with respect to the interpolated values (`y_points`) and optional endpoint derivative constraints.
- `SteadyEnergyBandMonitor` in the Charge solver.
- Pretty printing enabled with `rich.print` for the material library, materials, and their variants. In notebooks, this can be accessed using `rich.print` or `display`, or by evaluating the material library, a material, or a variant in a cell.
- `FieldData` and `ModeData` support exporting E fields to a Zemax Beam File (ZBF) with `.to_zbf()` (warning: experimental feature).
- `FieldDataset` supports reading E fields from a Zemax Beam File (ZBF) with `.from_zbf()` (warning: experimental feature).
- Unstructured grid now supports 2D/3D box-shaped refinement regions and 1D refinement lines of arbitrary direction.

### Changed
- Performance enhancement for adjoint gradient calculations by optimizing field interpolation.
- Auto grid in EME simulations with multiple `freqs` provided uses the largest instead of raising an error.
- Increased maximum number of frequencies in an EME simulation from 20 to 500
- Named mediums now display by name for brevity; materials/variants print concise summaries including references.

### Fixed
- Fixed `reverse` property of `td.Scene.plot_structures_property()` to also reverse the colorbar.
- Fixed bug in surface gradient computation where fields, instead of gradients, were being summed in frequency.

## [2.8.2] - 2025-04-09

### Added
- `fill` and `fill_structures` argument in `td.Simulation.plot_structures()` and `td.Simulation.plot()` respectively to disable fill and plot outlines of structures only.
- New subpixel averaging option `ContourPathAveraging` applied to dielectric material boundaries.
- A property `interior_angle` in `PolySlab` that stores angles formed inside polygon by two adjacent edges.
- `eps_component` argument in `td.Simulation.plot_eps()` to optionally select a specific permittivity component to plot (eg. `"xx"`).
- Monitor `AuxFieldTimeMonitor` for aux fields like the free carrier density in `TwoPhotonAbsorption`.
- Broadband handling (`num_freqs` argument) to the TFSF source.
- Ability to define a `WavePort` using only a voltage or current path integral, with the missing quantity inferred via power conservation.

### Fixed
- Compatibility with `xarray>=2025.03`.
- Inaccurate gradient when auto-grabbing permittivities for structures using `td.PolySlab` when using dispersive material models.
- Fixed scaling for adjoint sources when differentiating with respect to `FieldData` to account for the mesh size of the monitor and thus the created source. This aligns adjoint gradient magnitudes with numerical finite difference gradients for field data.
- Warn when mode solver pml covers a significant portion of the mode plane.
- TFSF server errors related to the auxiliary plane wave source that would previously happen on the server are now caught upon simulation creation.
- Opposite arrow curvature for mode sources and monitors with non-zero bendind radius when plotted in the figure's Y axis.

### Changed
- `num_freqs` in Gaussian beam type sources limited to 20, which should besufficient for all cases.
- The `angle_phi` parameter of `ModeSpec` is only limited to multiples of `np.pi / 2` if `angle_rotation` is set to `True`, as other values would currently not work correctly.
- The `ramp_up_iters` parameter of the `ChargeSolver` was changed back to 1 for efficiency. It can be increased in cases with e.g. high doping when convergence is more difficult.

## [2.8.1] - 2025-03-20

### Added
- New `LobeMeasurer` tool in the `microwave` plugin that locates lobes in antenna patterns and calculates lobe measures like half-power beamwidth and sidelobe level.
- Validation step that raises a `ValueError` when no frequency-domain monitors are present, preventing invalid adjoint runs.
- Metal surface roughness models: modified Hammerstad, Huray Snowball, and Cannonball-Huray.
- Support for Fermi-Dirac statistics in Charge solver. This option can be activated when defining the analysis type `IsothermalSteadyChargeDCAnalysis` with `fermi_dirac=True`. This option will provide more accurate results in simulations where very high doping may lead the pseudo-Fermi energy levels to approach either the conduction or the valence energy levels.

### Changed

- Dependencies for the `design` plugin can now be installed via `pip install tidy3d[design]`.

### Fixed
- Bug in `LayerRefinementSpec` that refines grids outside the layer region when one in-plane dimension is of size infinity.
- Querying tasks was sometimes erroring unexpectedly.
- Fixed automatic creation of missing output directories.
- Bug in handling of tuple-type gradients that could lead to empty tuples or failing gradient calculations when differentiating w.r.t. (for instance) `td.Box.center`.
- Bug causing incorrect field projection results when multiple projection monitors with numerous sampling points were used.
- Improved accuracy for normal E-field components in mode solver at microwave frequencies.
- Deleting tasks using `web.delete(task_id)` would error when deleting tasks in the tidy3d root folder and others would not get completely removed in the Web GUI.

## [2.8.0] - 2025-03-04

### Added

- Autograd support for polyslab transformations.
- Support for differentiation with respect to `ComplexPolySlab.vertices`.
- Introduce RF material library. Users can now import `rf_material_library` from `tidy3d.plugins.microwave`.
- Users can specify the background medium for a structure in automatic differentiation by supplying `Structure.background_permittivity`.
- `DirectivityMonitor` to compute antenna directivity and associated `DirectivityData` data container.
- Ability to directly create `DirectivityData` from an `xarrray.Dataset` containing electromagnetic fields, where the flux is calculated by integrating the fields over the surface of a sphere.
- `DirectivityData` includes parameters of interest like radiation intensity and gain. In addition, antenna parameters can be decomposed into contributions from individual polarization components according to a specified polarization basis, either `linear` or `circular`. The linear polarization basis can be optionally rotated by `tilt_angle` from the theta-axis.
- Added `plot_length_units` to `Simulation` and `Scene` to allow for specifying units, which improves axis labels and scaling when plotting.
- Added the helper function `compute_power_delivered_by_port`  to `TerminalComponentModeler` which computes power delivered to a microwave network from a port.
- Added the account function `account` to check credit balance and daily free simulation balance.
- Added `WavePort` to the `TerminalComponentModeler` which is the suggested port type for exciting transmission lines.
- Added plotting functionality to voltage and current path integrals in the `microwave` plugin.
- Added convenience functions `from_terminal_positions` and `from_circular_path` to simplify setup of `VoltageIntegralAxisAligned` and `CustomCurrentIntegral2D`, respectively.
`ComponentModeler.batch_data` convenience property to access the `BatchData` corresponding to the component modeler run.
- Added optimization methods to the Design plugin. The plugin has been expanded to include Bayesian optimization, genetic algorithms and particle swarm optimization. Explanations of these methods are available in new and updated notebooks.
- Added new support functions for the Design plugin: automated batching of `Simulation` objects, and summary functions with `DesignSpace.estimate_cost` and `DesignSpace.summarize`.
- Added validation and repair methods for `TriangleMesh` with inward-facing normals.
- Added `from_admittance_coeffs` to `PoleResidue`, which allows for directly constructing a `PoleResidue` medium from an admittance function in the Laplace domain.
- Added material type `LossyMetalMedium` that has high DC-conductivity. Its boundaries can be modeled by a surface impedance boundary condition (SIBC). This is treated as a subpixel method, which can be switched by setting `SubpixelSpec(lossy_metal=method)` where `method` can be `Staircasing()`, `VolumetricAveraging()`, or `SurfaceImpedance()`.
- Added mode solver option `precision='auto'` to automatically select single or double precision for optimizing performance and accuracy.
- Added `LinearLumpedElement` as a new supported `LumpedElement` type. This enhancement allows for the modeling of two-terminal passive networks, which can include any configuration of resistors, inductors, and capacitors, within the `Simulation`. Simple RLC networks are described using `RLCNetwork`, while more complex networks can be represented by their admittance transfer function through `AdmittanceNetwork`.
- Option for the `ModeSolver` and `ModeSolverMonitor` to store only a subset of field components.
- Added an option to simulate plane wave propagation at fixed angles by setting parameter `angular_spec=FixedAngleSpec()` when defining a `PlaneWave` source.
- The universal `tidy3d.web` can now also be used to handle `ModeSolver` simulations.
- Support for differentiation with respect to `PolySlab.slab_bounds`.
- `VisualizationSpec` that allows `Medium` instances to specify color and transparency plotting attributes that override default ones.
- `reduce_simulation` argument added to all web functions to allow automatically reducing structures only to the simulation domain, including truncating data in custom media, thus reducing the simulation upload size. Currently only implemented for mode solver simulation types.
- Support for quasi-uniform grid specifications via `QuasiUniformGrid` that subclasses from `GridSpec1d`. The grids are almost uniform, but can adjust locally to the edge of structure bounding boxes, and snapping points.
- New field `min_steps_per_sim_size` in `AutoGrid` that sets minimal number of grid steps per longest edge length of simulation domain.
- New field `shadow` in `MeshOverrideStructure` that sets grid size in overlapping region according to structure list or minimal grid size.
- New field `drop_outside_sim` in `MeshOverrideStructure` to specify whether to drop an override structure if it is outside the simulation domain, but it overlaps with the simulation domain when projected to an axis.
- Scientific notation for frequency range warnings.
- `MultiphysicsMedium` allows for the modularization definition of physical properties. So in addition to the usual optical properties, one can now add heat properties or electric properties.
- Added :zap: Charge Solver API. It is now possible to solve the Drift-Diffusion (DD) equations for semiconductors. These simulations can be set up with the new class `HeatChargeSimulation` (which supersedes `HeatSimulation`) in a way similar to that of Heat simulations. The solver has currently limited capabilities (which will be expanded in future releases): steady only, isothermal, Boltzmann statistics (non-degenerate semiconductors only).
- `SemiconductorMedium` has been added. This is used in combination with `MultiPhysicsMedium` to define the mediums used in the Charge solver. It allows to specify mobility, bandgap narrowing, and generation/recombination models as well as doping. Doping can be defined as constant or through "doping boxes" or, more generally, via `SpatialDataArray` objects for full flexibility.
- Mobility models for `SemiconductorMedium`: `ConstantMobilityModel`, `CaugheyThomasMobility`
- Bandgap narrowing models for `SemiconductorMedium`: `SlotboomBandGapNarrowing`
- Generation-recombination models for `SemiconductorMedium`: `ShockleyReedHallRecombination`, `RadiativeRecombination`, `AugerRecombination`
- Accessors and length functions implemented for `Result` class in design plugin.
- New interface for mode solver simulations via `ModeSimulation` class.
- Advanced option `dist_type` that allows `LinearLumpedElement` to be distributed across grid cells in different ways. For example, restricting the network portion to a single cell and using PEC wires to connect to the desired location of the terminals. 
- New `layer_refinement_specs` field in `GridSpec` that takes a list of `LayerRefinementSpec` for automatic mesh refinement and snapping in layered structures. Structure corners on the cross section perpendicular to layer thickness direction can be automatically identified. Mesh is automatically snapped and refined around those corners.
- Function `translated_copy` in `ElectromagneticFieldData` to facilitate computing overlaps between field data at different locations.
- `PlaneWaveBeamProfile`, `GaussianBeamProfile` and `AstigmaticGaussianBeamProfile` components to compute field data associated to such beams based on their corresponding analytical expressions, and compute field overlaps with other data.
- `num_freqs` argument to the `PlaneWave` source.
- Support for running multiple adjoint simulations from a single forward simulation in adjoint pipeline depending on monitor configuration.
- Ability to the `TerminalComponentModeler` that enables the computation of antenna parameters and figures of merit, such as gain, radiation efficiency, and reflection efficiency. When there are multiple ports in the `TerminalComponentModeler`, these antenna parameters may be calculated with user-specified port excitation magnitudes and phases.
- `RectangularAntennaArrayCalculator` class to compute the array factor and far-field radiation patterns for rectangular phased antenna arrays.
- Added validation for the monitors frequency range to be consistent with the frequency range defined for the sources.
- `Simulation.grid_info` property to collect various properties of the grids in the simulation.
- `angle_rotation` in `ModeSpec` to improve accuracy in some cases when `angle_theta` is defined. This option is disabled by default but it can be tried when very high accuracy is required in `ModeSource` and `ModeMonitor` objects that are placed in slanted waveguides.

### Changed

- Priority is given to `snapping_points` in `GridSpec` when close to structure boundaries, which reduces the chance of them being skipped.
- Gradients for autograd are computed server-side by default. They can be computed locally (requiring more data download) by passing `local_gradient=True` to the `web.run()` and related functions.
- Passing `path_dir` to `ComponentModeler` methods is deprecated in favor of setting `ComponentModeler.path_dir` and will result in an error if the two don't match.
- `BatchData` is now a mapping and can be accessed and iterated over like a Python dictionary (`.keys()`, `.values()`, `.items()`).
- `MethodRandom` and `MethodRandomCustom` have been removed from the Design plugin, and `DesignSpace.run_batch` has been superseded by `.run`.
- Design plugin has been significantly reworked to improve ease of use and allow for new optimization methods.
- Behavior of `FieldProjector` now matches the server-side computation, which does not truncate the integration surface when it extends into PML regions.
- Enabled the user to set the `ModeMonitor.colocate` field and changed to `True` by default (fields were actually already returned colocated even though this field was `False` previously).
- More robust mode solver at radio frequencies.
- Disallow very small polygons when subdividing 2D structures.
- For efficiency, `translated`, `scaled`, and `rotated` `PolySlab`-s now return updated `PolySlab` objects rather than a `Transformed` object, except when the rotation axis is not the same as the slab axis, in which case `Transformed` is still returned.
- Default choice of frequency in `Simulation.plot_eps` and `Simulation.plot_structures_eps` is now the central frequency of all sources in the simulation. If the central frequencies differ, the permittivity is evaluated at infinite frequency, and a warning is emitted. 
- Mode solver fields are more consistently normalized with respect to grid-dependent sign inversions in high order modes.
- `MeshOverrideStructure` accepts `Box` as geometry. Other geometry types will raise a warning and convert to bounding box.
- Double precision mode solver is now supported in EME.
- `estimate_cost` is now called at the end of every `web.upload` call.
- Internal refactor of adjoint shape gradients using `GradientSurfaceMesh`.
- Enhanced progress bar display in batch operations with better formatting, colors, and status tracking.
- `ModeMonitor` and `ModeSolverMonitor` now use the default `td.ModeSpec()` with `num_modes=1` when `mode_spec` is not provided.
- Update sidewall angle validator to clarify angle should be specified in radians.
- Reduced the complex tolerance in the mode solver below which permittivity is considered lossless, in order to correctly compute very low-loss modes.
-`HeatChargeSimulation` supersedes `HeatSimulation`. Though both of those can be used for Heat simulation interchangeably, the latter has been deprecated and will disappear in the future. 
- `FluidSpec` and `SolidSpec` are now deprecated in favor of `FluidMedium` and `SolidMedium`. Both can still be used interchangeably.
- Exclude `FluxMonitor` frequencies from adjoint design region gradient monitors since we do not differentiate through `FluxData`
- Check for empty source list in `wvl_mat_min` in `Simulation` and raise user friendly error if no sources exist in the simulation.
- The coordinate of snapping points in `GridSpec` can take value `None`, so that mesh can be selectively snapped only along certain dimensions.
- Grid snapping for `MeshOverrideStructure` with `shadow=False` is only on if the structure refines the mesh.
- Renamed `SkinDepthFitterParam` --> `SurfaceImpedanceFitterParam` used in `LossyMetalMedium`. `plot` in `LossyMetalMedium` now plots complex-valued surface impedance.
- Changed plot_3d iframe url to tidy3d production environment.
- `num_freqs` is now set to 3 by default for the `PlaneWave`, `GaussianBeam`, and `AnalyticGaussianBeam` sources, which makes the injection more accurate in broadband cases.
- Nonlinear models `KerrNonlinearity` and `TwoPhotonAbsorption` now default to using the physical real fields instead of complex fields.
- Added warning when a lumped element is not completely within the simulation bounds, since now lumped elements will only have an effect on the `Simulation` when they are completely within the simulation bounds.
- Allow `ModeData` to be passed to path integral computations in the `microwave` plugin.
- Default number of grid cells refining lumped elements is changed to 1, and some of the generated `MeshOverrideStructure` are replaced by grid snapping points.
- Simulation tasks now print a link to the task's folder.

### Fixed

- Significant speedup for field projection computations.
- Fix numerical precision issue in `FieldProjectionCartesianMonitor`.
- Bug where lumped elements in the `Simulation` were being overwritten by the `TerminalComponentModeler`.
- Bug in `Simulation.subsection` where lumped elements were not being correctly removed.
- Bug when adding 2D structures to the `Simulation` that are composed of multiple overlapping polygons.
- Fields stored in `ModeMonitor` objects were computed colocated to the grid boundaries, but the built-in `ModeMonitor.colocate=False` was resulting in wrong results in some cases, most notably if symmetries are also involved.
- Small inaccuracy when applying a mode solver `bend_radius` when the simulation grid is not symmetric w.r.t. the mode plane center. Previously, the radius was defined w.r.t. the middle grid coordinate, while now it is correctly applied w.r.t. the plane center.
- Silence warning in graphene from checking fit quality at large frequencies.
- `CustomCurrentSource` now correctly validates its current dataset.
- Better plotting of grids, respecting 2D and 1D simulation edge cases.
- Bug when `td.inf` are in `attrs` and files saved and loaded to .json twice.
- Cached property cleared in copying an object with `validate=False`.
- Error when `JaxSimulation` passed to `ModeSolver.simulation`: now it will be converted to a regular `Simulation`, with a warning.
- Thread-safe creation of `TIDY3D_DIR`.
- NumPy 2.1 compatibility issue where `numpy.float64` values passed to xarray interpolation would raise TypeError.
- Validation error in inverse design plugin when simulation has no sources by adding a source existence check before validating pixel size.
- System-dependent floating-point precision issue in EMEGrid validation.
- Fixed magnitude of gradient computation in `CustomMedium` by accounting properly for full volume element when permittivity data is defined over less dimensions than the medium.
- Fixed key ordering in design plugin when returning `Result` from an optimization method run.
- Bug in `Batch`, where duplicate folders might be created in the Web UI if the folder did not already exist.
- Make gauge selection for non-converged modes more robust.
- Fixed extremely long runtime generated by `RunTimeSpec` in the presence of `LossyMetalMedium`.
- Fixed `ComponentModeler.to_file` when its batch is empty.
- In web api, mode solver is patched with remote data so that certain methods like `plot_field` show remote data.
- Fixed common cross-referencing typo in docstrings.
- Added `viz_spec` property to `AbstractStructure` to fix error when plotting structures that have no `medium`.
- Fixed invalid adjoint source creation by disallowing sources made from non-traced fields.

## [2.7.9] - 2025-01-22

### Fixed
- Potential task name mismatches between forward and adjoint simulations in batch tasks.
- Magnitude of gradient computation in `CustomMedium` by accounting properly for full volume element when permittivity data is defined over fewer dimensions than the medium.

## [2.7.8] - 2024-11-27

### Changed
- `BatchData` is now a mapping and can be accessed and iterated over like a Python dictionary (`.keys()`, `.values()`, `.items()`).

### Fixed
- Gradient inaccuracy when a multi-frequency monitor is used but a single frequency is selected.
- Revert single cell center approximation for custom medium gradient.

## [2.7.7] - 2024-11-15

### Added
- Autograd support for local field projections using `FieldProjectionKSpaceMonitor`.
- Function `components.geometry.utils.flatten_groups` now also flattens transformed groups when requested.
- Differentiable `smooth_min`, `smooth_max`, and `least_squares` functions in `tidy3d.plugins.autograd`.
- Differential operators `grad` and `value_and_grad` in `tidy3d.plugins.autograd` that behave similarly to the autograd operators but support auxiliary data via `aux_data=True` as well as differentiation w.r.t. `DataArray`.
- `@scalar_objective` decorator in `tidy3d.plugins.autograd` that wraps objective functions to ensure they return a scalar value and performs additional checks to ensure compatibility of objective functions with autograd. Used by default in `tidy3d.plugins.autograd.value_and_grad` as well as `tidy3d.plugins.autograd.grad`.
- Autograd support for simulations without adjoint sources in `run` as well as `run_async`, which will not attempt to run the simulation but instead return zero gradients. This can sometimes occur if the objective function gradient does not depend on some simulations, for example when using `min` or `max` in the objective.

### Changed
- `CustomMedium` design regions require far less data when performing inverse design by reducing adjoint field monitor size for dims with one pixel.
- Calling `.values` on `DataArray` no longer raises a `DeprecationWarning` during automatic differentiation.
- Minimum number of PML layers set to 6.
- `Structure.background_permittivity : float` for specifying background medium for shape differentiation deprecated in favor of `Structure.background_medium : Medium` for more generality.
- Validate mode objects so that if a bend radius is defined, it is not smaller than half the size of the modal plane along the radial direction (i.e. the bend center is not inside the mode plane).

### Fixed
- Regression in local field projection leading to incorrect projection results.
- Bug when differentiating with respect to `Cylinder.center`.
- `xarray` 2024.10.0 compatibility for autograd.
- Some failing examples in the expressions plugin documentation.
- Inaccuracy in transforming gradients from edge to `PolySlab.vertices`.
- Bug in `run_async` where an adjoint simulation would sometimes be assigned to the wrong forward simulation.
- Validate against nonlinearity or modulation in `FullyAnisotropicMedium.from_diagonal`.
- Add warning to complex-field nonlinearities, which may require more careful physical interpretation.


## [2.7.6] - 2024-10-30

### Added
- Users can pass the `shading` argument in the `SimulationData.plot_field` to `Xarray.plot` method. When `shading='gouraud'`, the image is interpolated, producing a smoother visualization.
- Users can manually specify the background medium for a structure to be used for geometry gradient calculations by supplying `Structure.background_permittivity`. This is useful when there are overlapping structures or structures embedded in other mediums.
- Autograd functions can now be called directly on `DataArray` (e.g., `np.sum(data_array)`) in objective functions.
- Automatic differentiation support for local field projections with `FieldProjectionAngleMonitor` and `FieldProjectionCartesianMonitor` using `FieldProjector.project_fields(far_field_monitor)`.

### Changed
- Improved autograd tracer handling in `DataArray`, resulting in significant speedups for differentiation involving large monitors.
- Triangulation of `PolySlab` polygons now supports polygons with collinear vertices.
- Frequency and wavelength utilities under `tidy3d.frequencies` and `tidy3d.wavelengths`.

### Fixed
- Minor gradient direction and normalization fixes for polyslab, field monitors, and diffraction monitors in autograd.
- Resolved an issue where temporary files for adjoint simulations were not being deleted properly.
- Resolve several edge cases where autograd boxes were incorrectly converted to numpy arrays.
- Resolve issue where scalar frequencies in metric definitions (`ModeAmp(f=freq)` instead of `ModeAmp(f=[freq])`) would erroneously fail validation.

## [2.7.5] - 2024-10-16

### Added
- `TopologyDesignRegion` is now invariant in `z` by default and supports assigning dimensions along which a design should be uniform via `TopologyDesignRegion(uniform=(bool, bool, bool))`.
- Support for arbitrary padding sizes for all padding modes in `tidy3d.plugins.autograd.functions.pad`.
- `Expression.filter(target_type, target_field)` method for extracting object instances and fields from nested expressions.
- Additional constraints and validation logic to ensure correct setup of optimization problems in `invdes` plugin.
- `tidy3d.plugins.pytorch` to wrap autograd functions for interoperability with PyTorch via the `to_torch` wrapper.

### Changed
- Renamed `Metric.freqs` --> `Metric.f` and made frequency argument optional, in which case all frequencies from the relevant monitor will be extracted. Metrics can still be initialized with both `f` or `freqs`.

### Fixed
- Some validation fixes for design region.
- Bug in adjoint source creation that included empty sources for extraneous `FieldMonitor` objects, triggering unnecessary errors.
- Correct sign in objective function history depending on `Optimizer.maximize`.
- Fix to batch mode solver run that could create multiple copies of the same folder.
- Fixed ``ModeSolver.plot`` method when the simulation is not at the origin.
- Gradient calculation is orders of magnitude faster for large datasets and many structures by applying more efficient handling of field interpolation and passing to structures.
- Bug with infinite coordinates in `ClipOperation` not working with shapely.

## [2.7.4] - 2024-09-25

### Added
- New `tidy3d.plugins.expressions` module for constructing and serializing mathematical expressions and simulation metrics like `ModeAmp` and `ModePower`.
- Support for serializable expressions in the `invdes` plugin (`InverseDesign(metric=ModePower(...))`).
- Added `InitializationSpec` as the default way to initialize design region parameters in the `invdes` plugin (`DesignRegion(initialization_spec=RandomInitializationSpec(...))`).
- Callback support in `invdes.Optimizer` and support for running the optimizer for a fixed number of steps via the `num_steps` argument in `Optimizer.continue_run()`.
- Convenience method `Structure.from_permittivity_array(geometry, eps_data)`, which creates structure containing `CustomMedium` with `eps_data` array sampled within `geometry.bounds`.

### Changed
- All filter functions in `plugins/autograd` now accept either an absolute size in pixels or a `radius` and `dl` argument.
- Reverted fix for TFSF close to simulation boundaries that was introduced in 2.7.3 as it could cause different results in some cases with nonuniform mesh along the propagation direction.

### Fixed
- Ensure `path` argument in `run()` function is respected when running under autograd or the adjoint plugin.
- Bug in `Simulation.subsection` (used in the mode solver) when nonlinear materials rely on information about sources outside of the region.


## [2.7.3] - 2024-09-12

### Added
- Added value_and_grad function to the autograd plugin, importable via `from tidy3d.plugins.autograd import value_and_grad`. Supports differentiating functions with auxiliary data (`value_and_grad(f, has_aux=True)`).
- `Simulation.num_computational_grid_points` property to examine the number of grid cells that compose the computational domain corresponding to the simulation. This can differ from `Simulation.num_cells` based on boundary conditions and symmetries.
- Support for `dilation` argument in `JaxPolySlab`.
- Support for autograd differentiation with respect to `Cylinder.radius` and `Cylinder.center` (for elements not along axis dimension).
- `Cylinder.to_polyslab(num_pts_circumference, **kwargs)` to convert a cylinder into a discretized version represented by a `PolySlab`.
- `tidy3d.plugins.invdes.Optimizer` now accepts an optional optimization direction via the `maximize` argument (defaults to `maximize=True`).

### Changed
- `PolySlab` now raises error when differentiating and dilation causes damage to the polygon.
- Validator `boundaries_for_zero_dims` to raise error when Bloch boundaries are used along 0-sized dims.
- `FieldProjectionKSpaceMonitor` support for 2D simulations with `far_field_approx = True`. 

### Fixed
- `DataArray` interpolation failure due to incorrect ordering of coordinates when interpolating with autograd tracers.
- Error in `CustomSourceTime` when evaluating at a list of times entirely outside of the range of the envelope definition times.
- Improved passivity enforcement near high-Q poles in `FastDispersionFitter`. Failed passivity enforcement could lead to simulation divergences.
- More helpful error and suggestion if users try to differentiate w.r.t. unsupported `FluxMonitor` output.
- Removed positive warnings in Simulation validators for Bloch boundary conditions.
- Improve accuracy in `Box` shifting boundary gradients.
- Improve accuracy in `FieldData` operations involving H fields (like `.flux`).
- Better error and warning handling in autograd pipeline.
- Added the option to specify the `num_freqs` argument and `kwargs` to the `.to_source` method for both `ModeSolver` and `ComponentModeler`.
- Fixes to TFSF source in some 2D simulations, and in some cases when the injection plane is close to the simulation domain boundaries.

## [2.7.2] - 2024-08-07

### Added
- Mode solver plugin now supports 'EMESimulation'.
- `TriangleMesh` class: automatic removal of zero-area faces, and functions `fill_holes` and `fix_winding` to attempt mesh repair.
- Added progress bar for EME simulations.
- Support for grid specifications from grid cell boundary coordinates via `CustomGridBoundaries` that subclasses from `GridSpec1d`.
- More convenient mesh importing from another simulation through `grid_spec = GridSpec.from_grid(sim.grid)`.
- `autograd` gradient calculations can be performed on the server by passing `local_gradient = False` into `web.run()` or `web.run_async()`.
- Automatic differentiation with `autograd` supports multiple frequencies through single, broadband adjoint simulation when the objective function can be formulated as depending on a single dataset in the output `SimulationData` with frequency dependence only.
- Convenience method `EMESimulation.subsection` to create a new EME simulation based on a subregion of an existing one.
- Added `flux` and `poynting` properties to `FieldProjectionCartesianData`.

### Changed
- Mode solver now always operates on a reduced simulation copy.
- Moved `EMESimulation` size limit validators to preupload.
- Error if field projection monitors found in 2D simulations, except `FieldProjectionAngleMonitor` or `FieldProjectionCartesianMonitor` with `far_field_approx = True`. Support for other monitors and for exact field projection will be coming in a subsequent Tidy3D version.

### Fixed
- Error when loading a previously run `Batch` or `ComponentModeler` containing custom data.
- Error when plotting mode plane PML and the simulation has symmetry.
- Validators using `TriangleMesh.intersections_plane` will fall back on bounding box in case the method fails for a non-watertight mesh.
- Bug when running the same `ModeSolver` first locally then remotely, or vice versa, in which case the cached data from the first run is always returned.
- Gradient monitors for `PolySlab` only store fields at the center location along axis, reducing data usage.
- Validate the forward simulation on the client side even when using `local_gradient=False` for server-side gradient processing.
- Gradient inaccuracies in `PolySlab.vertices`, `Medium.conductivity`, and `DiffractionData` s-polarization.
- Adjoint field monitors no longer store H fields, which aren't needed for gradient calculation.
- `MeshOverrideStructures` in a `Simulation.GridSpec` are properly handled to remove any derivative tracers.

## [2.7.1] - 2024-07-10

### Added
- Support for differentiation with respect to `GeometryGroup.geometries` elements.
- Users can now export `SimulationData` to MATLAB `.mat` files with the `to_mat_file` method.
- `ModeSolver` methods to plot the mode plane simulation components, including `.plot()`, `.plot_eps()`, `.plot_structures_eps()`, `.plot_grid()`, and `.plot_pml()`.
- Support for differentiation with respect to monitor attributes that require interpolation, such as flux and intensity.
- Support for automatic differentiation with respect to `.eps_inf` and `.poles` contained in dispersive mediums `td.PoleResidue` and `td.CustomPoleResidue`.
- Support for `FieldProjectionAngleMonitor` for 2D simulations with `far_field_approx = True`.

### Fixed
- Bug where boundary layers would be plotted too small in 2D simulations.
- Bug when plotting transformed geometries.
- Bug when snapping `CoaxialLumpedPort` to grid cell boundaries.
- Errors in `PolySlab` when using autograd differentiation with non-zero `sidewall_angle` and `dilation`.
- Error in `EMESimulationData.smatrix_in_basis` when using older versions of xarray.
- Gradients for `Box` objects when simulation size is < 3D.

## [2.7.0] - 2024-06-17

### Added
- `HeatSimulation` is now under the more generic class `HeatChargeSimulation`. This new class sallows for imulations of both HEAT and electric CONDUCTION types. One-way coupling between these is possible in which the CONDUCTION simulation is used to compute Joule heating which is subsequently used as a source in the HEAT simulation.
- EME solver through `EMESimulation` class.
- 2D heat simulations are now fully supported. 
- `tidy3d.plugins.adjoint.web.run_local` used in place of `run` will skip validators that restrict the size or number of `input_structures`.
- Introduces the `microwave` plugin which includes `ImpedanceCalculator` for computing the characteristic impedance of transmission lines.
- `Simulation` now accepts `LumpedElementType`, which currently only supports the `LumpedResistor` type. `LumpedPort` together with `LumpedResistor` make up the new `TerminalComponentModeler` in the `smatrix` plugin.
- Uniaxial medium Lithium Niobate to material library.
- Properties `num_time_steps_adjoint` and `tmesh_adjoint` to `JaxSimulation` to estimate adjoint run time.
- Ability to add `path` to `updated_copy()` method to recursively update sub-components of a tidy3d model. For example `sim2 = sim.updated_copy(size=new_size, path="structures/0/geometry")` creates a recursively updated copy of `sim` where `sim.structures[0].geometry` is updated with `size=new_size`.
- Python 3.12 support. Python 3.8 deprecation. Updated dependencies.
- Tidy3D objects may store arbitrary metadata in an `.attrs` dictionary.
- `JaxSimulation` now supports the following GDS export methods: `to_gds()`, `to_gds_file()`, `to_gdspy()`, and `to_gdstk()`.
- `RunTimeSpec` accepted by `Simulation.run_time` to adaptively set the run time based on Q-factor, propagation length, and other factors.
- `JaxDataArray` now supports selection by nearest value via `JaxDataArray.sel(..., method="nearest")`.
- Convenience method `constant_loss_tangent_model` in `FastDispersionFitter` to fit constant loss tangent material model.
- Classmethods in `DispersionFitter` to load complex-valued permittivity or loss tangent data.
- Pre-upload validator to check that mode sources overlap with more than 2 grid cells.
- Support `2DMedium` for `Transformed`/`GeometryGroup`/`ClipOperation` geometries.
- `num_proc` argument to `tidy3d.plugins.adjoint.web.run_local` to control the number of processes used on the local machine for gradient processing.
- Support for complex and self-intersecting polyslabs in adjoint module via `JaxComplexPolySlab`.
- Support for `.gz` files in `Simulation` version updater.
- Warning if a nonuniform custom medium is intersecting `PlaneWave`, `GaussianBeam`, `AstigmaticGaussianBeam`, `FieldProjectionCartesianMonitor`, `FieldProjectionAngleMonitor`, `FieldProjectionKSpaceMonitor`, and `DiffractionMonitor`.
- Added a `CoaxialLumpedPort` and `CoaxialLumpedResistor` for coaxial type transmission lines and excitations.
- Automatic differentiation supported natively in Tidy3D components, and through the `web.run()` and `web.run_async()` functions through `autograd`.
- A batch of `ModeSolver` objects can be run concurrently using `tidy3d.plugins.mode.web.run_batch()`
- `RectangularWaveguide.plot_field` optionally draws geometry edges over fields. 
- `RectangularWaveguide` supports layered cladding above and below core.
- `SubpixelSpec` accepted by `Simulation.subpixel` to select subpixel averaging methods separately for dielectric, metal, and PEC materials. Specifically, added support for conformal mesh methods near PEC structures that can be specified through the field `pec` in the `SubpixelSpec` class. Note: previously, `subpixel=False` was implementing staircasing for every material except PEC. Now, `subpixel=False` implements direct staircasing for all materials. For PEC, the behavior of `subpixel=False` in Tidy3D < 2.7 is now achieved through `subpixel=SubpixelSpec(pec=HeuristicPECStaircasing())`, while `subpixel=True` in Tidy3D < 2.7 is now achieved through `subpixel=SubpixelSpec(pec=Staircasing())`. The default is `subpixel=SubpixelSpec(pec=PECConformal())` for more accurate PEC modelling.
- Lossless `Green2008` variant for crystalline silicon added to material library.
- `GridSpec` supports `snapping_points` that enforce grid boundaries to pass through them.
- Support for unstructured datasets (`TriangularGridDataset` and `TetrahedralGridDataset`) in custom medium classes.
- Support for `Transformed`/`GeometryGroup`/`ClipOperation` geometries in heat solver.
- Parameter `relative_min_dl` in `UniformUnstructuredGrid` and `DistanceUnstructuredGrid` to control minimal mesh size.
- Improved validation for `HeatSimulation`.
- Functions to clean unstructured datasets from degenerate cells and unused points.

### Changed
- IMPORTANT NOTE: differentiable fields in the `adjoint` plugin (`JaxBox.size`, `JaxBox.center`, `JaxPolySlab.vertices`) no longer store the derivative information after the object is initialized. For example, if using JaxPolySlab.vertices directly in an objective function, the vertices will have no effect on the gradient. Instead, this information is now stored in a field of the same name with `_jax` suffix, eg. `JaxPolySlab.vertices_jax`. For some applications, such as evaluating penalty values, please change to `radius_penalty.evaluate(polyslab.vertices_jax)` or use the vertices as generated by your parameterization functions (`make_vertices(params)`).
- `run_time` of the adjoint simulation is set more robustly based on the adjoint sources and the forward simulation `run_time` as `sim_fwd.run_time + c / fwdith_adj` where `c=10`.
- `FieldTimeMonitor` restriction to record at a maximum of 5000 time steps if the monitor is not zero-dimensional, to avoid creating unnecessarily large amounts of data.
- Bumped `trimesh` version to `>=4,<4.2`.
- Make directories in path passed to `.to_gds_file()` methods, if they don't exist.
- Changed sign of mode solver PML conductivity to match the `exp(-1j * omega * t)` convention used everywhere else. Previously, loss due to PML was coming out as negative `k_eff` of the mode, while now it comes out as positive `k_eff`, in line with material loss.
- Augmented the PML mode solver profile to the form `s(x) = kappa(x) + 1j * sigma(x) / (omega * EPSILON_0)`. Previously, `kappa` was not used. The parameters are currently set to `kappa_min = 1`, `kappa_max = 3`, `sigma_max = 2`. These are not yet exposed to the user.
- Also scaling `sigma_max` by the average relative impedance in the mode solver PML region.
- `tidy3d convert` from `.lsf` files to tidy3d scripts has moved to another repository at `https://github.com/hirako22/Lumerical-to-Tidy3D-Converter`.
- Relax validation of smallest triangle area in `TriangleMesh`.
- Default variant for silicon dioxide in material library switched from `Horiba` to `Palik_Lossless`.
- Sources and monitors which are exactly at the simulation domain boundaries will now error. They can still be placed very close to the boundaries, but need to be on the inside of the region.
- Relaxed `dt` stability criterion for 1D and 2D simulations.
- Switched order of angle and bend transformations in mode solver when both are present.

### Fixed
- Bug in PolySlab intersection if slab bounds are `inf` on one side.
- Better error message when trying to transform a geometry with infinite bounds.
- `JaxSimulation.epsilon` properly handles `input_structures`.
- `FieldData.flux` in adjoint plugin properly returns `JaxDataArray` containing frequency coordinate `f` instead of summing over values.
- Proper error message when trying to compute Poynting vector for adjoint field data.
- Bug in plotting and computing tilted plane intersections of transformed 0 thickness geometries.
- `Simulation.to_gdspy()` and `Simulation.to_gdstk()` now place polygons in GDS layer `(0, 0)` when no `gds_layer_dtype_map` is provided, instead of erroring.
- `task_id` now properly stored in `JaxSimulationData`.
- Bug in `FastDispersionFitter` when poles move close to input frequencies.
- Bug in plotting polarization vector of angled sources.
- Bug in `SpatialDataArray.reflect()` that was causing errors for older versions of `xarray`.
- `ModeSolver.plot_field` correctly returning the plot axes.
- Avoid error if non-positive refractive index used for integration resolution in adjoint.
- Make `Batch.monitor` robust if the run status is not found.
- Bugs in slicing unstructured datasets along edges.
- Correct behavior of temperature monitors in the presence os symmetry.

## [2.6.4] - 2024-04-23

### Fixed
- Set `importlib-metadata`, `boto3`, `requests`, and `click` version requirements to how they were in Tidy3D v2.5.
- Bug in `td.FullyAnisotropicMedium` when added to `adjoint.JaxStructureStaticMedium`.
- Bug when extra `**kwargs` passed to `Design.run_batch()`.

## [2.6.3] - 2024-04-02

### Added
- Added new validators in `HeatSimulation`: no structures with dimensions of zero size, no all-Neumann boundary conditions, non-empty simulation domain.
- `web.Batch` uses multi-threading for upload and download operations. The total time for many tasks is reduced by an order of magnitude.

### Changed
- Revert forbidden `"` in component names.

## [2.6.2] - 2024-03-21

### Changed
- Characters `"` and `/` not allowed in component names.
- Change error when `JaxPolySlab.sidewall_angle != 0.0` to a warning, enabling optimization with slanted sidewalls if a lower accuracy gradient is acceptable.
- Simplified output and logging in `web.Batch` with `verbose=True`.

### Fixed
- Compute time stepping speed shown `tidy3d.log` using only the number of time steps that was run in the case of early shutoff. Previously, it was using the total number of time steps.
- Bug in PolySlab intersection if slab bounds are `inf` on one side.
- Divergence in the simultaneous presence of PML, absorber, and symmetry.
- Fixed validator for `ModeSpec.bend_radius == 0`, which was not raising an error.

## [2.6.1] - 2024-03-07

### Added
- `tidy3d.plugins.design.Results` store the `BatchData` for batch runs in the `.batch_data` field.
- Prompting user to check solver log when loading solver data if warnings were found in the log, or if the simulation diverged or errored.

### Changed
- Slightly reorganized `web.run` logging when `verbose=True` to make it clearer.

### Fixed
- Fix to 3D surface integration monitors with some surfaces completely outside of the simulation domain which would sometimes still record fields.
- Better error handling if remote `ModeSolver` creation gives response of `None`.
- Validates that certain incompatible material types do not have intersecting bounds.
- Fixed handling of the `frequency` argument in PEC medium.
- Corrected plotting cmap if `val='re'` passed to `SimulationData.plot_field`.
- Bug when converting point `FieldMonitor` data to scalar amplitude for adjoint source in later jax versions.
- Handle special case when vertices overlap in `JaxPolySlab` to give 0 grad contribution from edge.
- Corrected some mistakes in the estimation of the solver data size for each monitor type, which affects the restrictions on the maximum monitor size that can be submitted.
- Bug in visualizing a slanted cylinder along certain axes in 2D.
- Bug in `ModeSolver.reduced_simulation_copy` that was causing mode profiles to be all `NaN`.
- Bug in `Simulation.subsection()` that was causing zero-size dimensions not to be preserved.
- Bug in `CustomGrid` that was causing an error when using for zero-size dimensions.
- When downloading gzip-ed solver data, automatically create the parent folder of the `to_file` path if it does not exist.

## [2.6.0] - 2024-02-21

### Added
- Automatic subdivision of 2D materials with inhomogeneous substrate/superstrate.
- Mode field profiles can be stored directly from a `ModeMonitor` by setting `store_fields_direction`.
- Users can toggle https ssl version through `from tidy3d.web.core.environment import Env` and `Env.set_ssl_version(ssl_version: ssl.TLSVersion)`
- Free-carrier absorption (FCA) and free-carrier plasma dispersion (FCPD) nonlinearities inside `TwoPhotonAbsorption` class.
- `log_path` argument in `set_logging_file`, set to `False` by default.
- `ErosionDilationPenalty` to `tidy3d.plugins.adjoint.utils.penalty` to penalize parameter arrays that change under erosion and dilation. This is a simple and effective way to penalize features that violate minimum feature size or radius of curvature fabrication constraints in topology optimization.
- `tidy3d.plugins.design` tool to explore user-defined design spaces.
- `ModeData.dispersion` and `ModeSolverData.dispersion` are calculated together with the group index.
- A utility function `td.medium_from_nk()` that automatically constructs a non-dispersive medium when permittivity>=1, and a single-pole Lorentz medium when permittivity<1.
- Integration of the `documentation` alongside the main codebase repository.
- Integration of the `tidy3d-notebooks` repository.
- `tidy3d develop` CLI and development guide on the main documentation.
- Added a convenience method `Simulation.subsection()` to a create a new simulation based on a subregion of another one.
- Users can toggle task caching through `from tidy3d.web.core.environment import Env` and `Env.enable_caching(True)` to enable, `Env.enable_caching(False)` to disable, or `Env.enable_caching(None)` to use global setting from web client account page. 

### Changed
- `DataArray.to_hdf5()` accepts both file handles and file paths.
- `ModeSolverMonitor` is deprecated. Mode field profiles can be retrieved directly from `ModeMonitor` with `store_fields_direction` set.
- The log file for a simulation run has been modified to include more information including warnings collected during execution.
- `poetry` based installation. Removal of `setup.py` and `requirements.txt`.
- Upgrade to sphinx 6 for the documentation build, and change of theme.
- Remote mode solver web api automatically reduces the associated `Simulation` object to the mode solver plane before uploading it to server.
- All solver output is now compressed. However, it is automatically unpacked to the same `simulation_data.hdf5` by default when loading simulation data from the server.
- Internal refactor of `adjoint` plugin to separate `jax`-traced fields from regular `tidy3d` fields.
- Added an optional argument `field` in class method `.from_vtu()` of `TriangularGridDataset` and `TetrahedralGridDataset` for specifying the name of data field to load.

### Fixed
- Add dispersion information to dataframe output when available from mode solver under the column "dispersion (ps/(nm km))".
- Skip adjoint source for diffraction amplitudes of NaN.
- Helpful error message if `val` supplied to `SimulationData.plot_field` not supported.
- Fixed validator that warns if angled plane wave does not match simulation boundaries, which was not warning for periodic boundaries.
- Validates that no nans are present in `DataArray` values in custom components.
- Removed nans from Cartesian temperature monitors in thermal simulations by using nearest neighbor interpolation for values outside of heat simulation domain.
- Removed spurious warnings realted to reloading simulation containing `PerturbationMedium` with `CustomChargePerturbation`/`CustomHeatPerturbation`.

## [2.5.2] - 2024-01-11

### Fixed
- Internal storage estimation for 3D surface integration monitors now correctly includes only fields on the surfaces, and not the whole volume.

## [2.5.1] - 2024-01-08

### Added
- `ModeData.dispersion` and `ModeSolverData.dispersion` are calculated together with the group index.
- String matching feature `contains_str` to `assert_log_level` testing utility.
- Warning in automatic grid generation if a structure has a non-zero size along a given direction that is too small compared to a single mesh step.
- `assert_log_level` adds string matching with `contains_str` and ensures no higher log level recorded than expected.
- `AssertLogLevel` context manager for testing log level and automatically clearing capture.
- More robust validation for boundary conditions and symmetry in 1D and 2D simulations.

### Changed
- `jax` and `jaxlib` versions bumped to `0.4.*`.
- Improved and unified warning message for validation failure of dependency fields in validators.

### Fixed
- Error in automatic grid generation in specific cases with multiple thin structures.

## [2.5.0] - 2023-12-13

### Added
- Ability to mix regular mediums and geometries with differentiable analogues in `JaxStructure`. Enables support for shape optimization with dispersive mediums. New classes `JaxStructureStaticGeometry` and `JaxStructureStaticMedium` accept regular `Tidy3D` geometry and medium classes, respectively.
- Warning if nonlinear mediums are used in an `adjoint` simulation. In this case, the gradients will not be accurate, but may be approximately correct if the nonlinearity is weak.
- Validator for surface field projection monitors that warns if projecting backwards relative to the monitor's normal direction.
- Validator for field projection monitors when far field approximation is enabled but the projection distance is small relative to the near field domain.
- Ability to manually specify a medium through which to project fields, when using field projection monitors.
- Added support for two-photon absorption via `TwoPhotonAbsorption` class. Added `KerrNonlinearity` that implements Kerr effect without third-harmonic generation.
- Can create `PoleResidue` from LO-TO form via `PoleResidue.from_lo_to`.
- Added `TriangularGridDataset` and `TehrahedralGridDataset` for storing and manipulating unstructured data.
- Support for an anisotropic medium containing PEC components.
- `SimulationData.mnt_data_from_file()` method to load only a single monitor data object from a simulation data `.hdf5` file.
- `_hash_self` to base model, uses `hashlib` to hash a Tidy3D component the same way every session.
- `ComponentModeler.plot_sim_eps()` method to plot the simulation permittivity and ports.
- Support for 2D PEC materials.
- Ability to downsample recorded near fields to speed up server-side far field projections.
- `FieldData.apply_phase(phase)` to multiply field data by a phase.
- Optional `phase` argument to `SimulationData.plot_field` that applies a phase to complex-valued fields.
- Ability to window near fields for spatial filtering of far fields for both client- and server-side field projections.
- Support for multiple frequencies in `output_monitors` in `adjoint` plugin.
- GDSII export functions `to_gds_file`, `to_gds`, `to_gdspy`, and `to_gdstk` to `Simulation`, `Structure`, and `Geometry`.
- `verbose` argument to `estimate_cost` and `real_cost` functions such that the cost is logged if `verbose==True` (default). Additional helpful messages may also be logged.
- Support for space-time modulation of permittivity and electric conductivity via `ModulationSpec` class. The modulation function must be separable in space and time. Modulations with user-supplied distributions in space and harmonic modulation in time are supported.
- `Geometry.intersections_tilted_plane` calculates intersections with any plane, not only axis-aligned ones.
- `Transformed` class to support geometry transformations.
- Methods `Geometry.translated`, `Geometry.scaled`, and `Geometry.rotated` can be used to create transformed copies of any geometry.
- Time zone in webAPI logging output.
- Class `Scene` consisting of a background medium and structures for easier drafting and visualization of simulation setups as well as transferring such information between different simulations.
- Solver for thermal simulation (see `HeatSimulation` and related classes).
- Specification of material thermal properties in medium classes through an optional field `.heat_spec`.

### Changed
- Credit cost for remote mode solver has been modified to be defined in advance based on the mode solver details. Previously, the cost was based on elapsed runtime. On average, there should be little difference in the cost.
- Mode solves that are part of an FDTD simulation (i.e. for mode sources and monitors) are now charged at the same flex credit cost as a corresponding standalone mode solver call.
- Any `FreqMonitor.freqs` or `Source.source_time.freq0` smaller than `1e5` now raise an error as this must be incorrect setup that is outside the Tidy3D intended range (note default frequency is `Hz`).
- When using complex fields (e.g. with Bloch boundaries), FluxTimeMonitor and frequency-domain fields (including derived quantities like flux) now only use the real part of the time-domain electric field.
- Indent for the json string of Tidy3D models has been changed to `None` when used internally; kept as `indent=4` for writing to `json` and `yaml` files.
- API for specifying one or more nonlinear models via `NonlinearSpec.models`.
- `freqs` and `direction` are optional in `ModeSolver` methods converting to monitor and source, respectively. If not supplied, uses the values from the `ModeSolver` instance calling the method.
- Removed spurious ``-1`` factor in field amplitudes injected by field sources in some cases. The injected ``E``-field should now exactly match the analytic, mode, or custom fields that the source is expected to inject, both in the forward and in the backward direction.
- Restriction on the maximum memory that a monitor would need internally during the solver run, even if the final monitor data is smaller.
- Restriction on the maximum size of mode solver data produced by a `ModeSolver` server call.
- Updated versions of `boto3`, `requests`, and `click`.
- python 3.7 no longer tested nor supported.
- Removed warning that monitors now have `colocate=True` by default.
- If `PML` or any absorbing boundary condition is used along a direction where the `Simulation` size is zero, an error will be raised, rather than just a warning.
- Remove warning that monitors now have `colocate=True` by default.
- Internal refactor of Web API functionality.
- `Geometry.from_gds` doesn't create unnecessary groups of single elements.

### Fixed
- Fixed energy leakage in TFSF when using complex fields.
- Fixed the duplication of log messages in Jupyter when `set_logging_file` is used.
- If input to circular filters in adjoint have size smaller than the diameter, instead of erroring, warn user and truncate the filter kernel accordingly.
- When writing the json string of a model to an `hdf5` file, the string is split into chunks if it has more than a set (very large) number of characters. This fixes potential error if the string size is more than 4GB.
- Proper equality checking between `Tidy3dBaseModel` instances, which takes `DataArray` values and coords into account and handles `np.ndarray` types.
- Correctly set the contour length scale when exporting 2D (or 1D) structures with custom medium to GDSII.
- Improved error handling if file can not be downloaded from server.
- Fix for detection of file extensions for file names with dots.
- Restrict to `matplotlib` >= 3.5, avoiding bug in plotting `CustomMedium`.
- Fixes `ComponentModeler` batch file being different in different sessions by use of deterministic hash function for computing batch filename.
- Can pass `kwargs` to `ComponentModeler.plot_sim()` to use in `Simulation.plot()`.
- Ensure that mode solver fields are returned in single precision if `ModeSolver.ModeSpec.precision == "single"`.
- If there are no adjoint sources for a simulation involved in an objective function, make a mock source with zero amplitude and warn user.

## [2.4.3] - 2023-10-16

### Added
- `Geometry.zero_dims` method that uses `Geometry.bounds` and speeds up the validator for zero-sized geometries.

### Changed
- Limited number of distinct sources to 1000. In cases where a complicated spatial dependence of the source is desired, a ``CustomFieldSource`` or a ``CustomCurrentSource`` can be used instead of multiple distinct sources.

### Fixed
- Properly handle `.freqs` in `output_monitors` of adjoint plugin.
- Simulation updater for `.hdf5` files with custom data.
- Fix to solver geometry parsing in some edge cases of slanted polyslab and STL geometries that could lead to an error or divergence.
- Fix to errors in some edge cases of a TFSF source setup.

## [2.4.2] - 2023-9-28

### Added
- Warnings for too many frequencies in monitors; too many modes requested in a ``ModeSpec``; too many number of grid points in a mode monitor or mode source.

### Changed
- Time domain monitors warn about data usage if all defaults used in time sampler specs.

### Fixed
- Faster sorting of modes in `ModeSolverData.overlap_sort` by avoiding excessive data copying.
- Ensure same `Grid` is generated in forward and adjoint simulations by setting `GridSpec.wavelength` manually in adjoint.
- Properly handling of `JaxBox` derivatives both for multi-cell and single cell thickness.
- Properly handle `JaxSimulation.monitors` with `.freqs` as `np.ndarray` in adjoint plugin.
- Properly handle `JaxDataArray.sel()` with single coordinates and symmetry expansion.
- Properly handle `JaxDataArray * xr.DataArray` broadcasting.
- Stricter validation of `JaxDataArray` coordinates and values shape.


## [2.4.1] - 2023-9-20

### Added
- `ModeSolverData.pol_fraction` and `ModeSolverData.pol_fraction_waveguide` properties to compute polarization fraction of modes using two different definitions.
- `ModeSolverData.to_dataframe()` and `ModeSolverData.modes_info` for a convenient summary of various modal properties of the computed modes.
- Loss upper bound estimation in `PoleResidue` material model.
- Command line tool `tidy3d convert` for conversion from `.lsf` project files into equivalent python scripts implementing the project in Tidy3D. Usage:  `tidy3d convert example.lsf tidy3d_script.py`.

### Changed
- Output task URL before and after simulation run and make URLs blue underline formatting.
- Support to load and save compressed HDF5 files (`.hdf5.gz`) directly from `Tidy3dBaseModel`.
- Line numbers no longer printed in webapi log output.
- Empty list returned if the folder cannot be queried in `web.get_tasks()`.

### Fixed
- Filtering based on `ModeSpec.filter_pol` now uses the user-exposed `ModeSolverData.pol_fraction` property. This also fixes the previous internal handling which was not taking the nonuniform grid, as well as and the propagation axis direction for modes in angled waveguides. In practice, the results should be similar in most cases.
- Bug with truly anisotropic `JaxCustomMedium` in adjoint plugin.
- Bug in adjoint plugin when `JaxBox` is less than 1 grid cell thick.
- Bug in `adjoint` plugin where `JaxSimulation.structures` did not accept structures containing `td.PolySlab`.

## [2.4.0] - 2023-9-11

### Added
- Configuration option `config.log_suppression` can be used to control the suppression of log messages.
- `web.abort()` and `Job.abort()` methods allowing user to abort running tasks without deleting them. If a task is aborted, it cannot be restarted later, a new one needs to be created and submitted.
- `FastDispersionFitter` for fast fitting of material dispersion data.
- `Simulation.monitors_data_size` property mapping monitor name to its data size in bytes.
- Source with arbitrary user-specified time dependence through `CustomSourceTime`.
- Interface for specifying material heat and charge perturbation models. 
Specifically, non-dispersive and dispersive mediums with heat and/or charge perturbation models can be defined through classes `PerturbationMedium` and `PerturbationPoleResidue`, 
where perturbations to each parameter is specified using class `ParameterPerturbation`.
A convenience function `Simulation.perturbed_mediums_copy` is added to class `Simulation` which applies heat and/or charge fields to mediums containing perturbation models.
- Added `hlim` and `vlim` kwargs to `Simulation.plot()` and `Simulation.plot_eps()` for setting horizontal and vertical plot limits.
- Added support for chi3 nonlinearity via `NonlinearSusceptibility` class.
- Spatial downsampling allowed in ``PermittivityMonitor`` through the ``interval_space`` argument.
- `ClipOperation` geometry type allows the construction of complex geometries through boolean operations.
- Operations on geometry (`+`, `|`, `*`, `&`, `-`, `^`) will create the appropriate `ClipOperation` or `GeometryGroup`.
- `Geometry.from_shapely` to extrude shapely primitives into Tidy3D geometry.
- `Geometry.from_gds` to extrude GDS cells into geometry groups with support for holes.
- `components.geometry.utils.traverse_geometry` used internally to traverse geometry trees.
- `components.geometry.utils.flatten_groups` used internally to validate large geometry trees.

### Changed
- Add `width` and `height` options to `Simulation.plot_3d()`.
- `sim_with_source()`, `sim_with_monitor()`, and `sim_with_mode_solver_monitor()` methods allowing the `ModeSolver` to create a copy of its `Simulation` with an added `ModeSource`, `ModeMonitor`, or `ModeSolverMonitor`, respectively.
- `nyquist_step` also taking the frequency range of frequency-domain monitors into account.
- Added option to allow DC component in `GaussianPulse` spectrum, by setting `remove_dc_component=False` in `GaussianPulse`.
- Jax installation from `pip install "tidy3d[jax]"` handled same way on windows as other OS if python >= 3.9.
- `colocate` introduced as an argument to `ModeSolver` and a Field in `ModeSolverMonitor`, set to True by default.
- `FieldTimeMonitor`-s and `FieldMonitor`-s that have `colocate=True` return fields colocated to the grid boundaries rather than centers. This matches better user expectations for example when the simulation has a symmetry (explicitly defined, or implicit) w.r.t. a given axis. When colocating to centers, fields would be `dl / 2` away from that symmetry plane, and components that are expected to go to zero do not (of course, they still do if interpolated to the symmetry plane). Another convenient use case is that it is easier to place a 2D monitor exactly on a grid boundary in the automatically generated grid, by simply passing an override structure with the monitor geometry.
- In these monitors, `colocate` is now set to `True` by default. This is to avoid a lot of potential confusion coming from returning non-colocated fields by default, when colocated fields need to be used when computing quantities that depend on more than one field component, like flux or field intensity.
- Field colocation for computations like flux, Poynting, and modal overlap also happen to cell boundaries rather than centers. The effect on final results
should be close to imperceptible as verified by a large number of backend tests and our online examples. Any difference can be at most on the scale of
the difference that can be observed when slightly modifying the grid resolution.
- `GeometryGroup` accepts other `GeometryGroup` instances as group elements.
- FDTD and mode solver tasks always upload `hdf5.gz` file instead of `hdf5` or `json`.
- `web.download_json()` will download `simulation.hdf5.gz` and unzip it, then load the json from the hdf5 file.
- `SimulationTask.get_simulation_hdf5()` will download simulation.hdf5.gz and unzip it to hdf5 file.
- The width of Tidy3D logging output is changed to 80 characters.
- Upgrades to `pydantic==2.*` with `pydantic.v1` imports.
- Uses `ruff` for linting instead of `pylint`.
- Added lower bound validator to `freqs` in mode solver.
- Added lower bound validator to `group_index_step` in `ModeSpec`.
- `ComponentModeler.freqs` now a `FreqArray`, so it can be initialized from numpy, list or tuple.

### Fixed
- Handles `TIDY3D_ENV` properly when set to `prod`.
- Redundant phase compensation for shifted source in smatrix plugin.
- Bug in angled mode solver with negative `angle_theta`.
- Properly include `JaxSimulation.input_structures` in `JaxSimulationData.plot_field()`.
- Numerically stable sigmoid function in radius of curvature constraint.
- Spatial monitor downsampling when the monitor is crossing a symmetry plane or Bloch boundary conditions.
- Cast `JaxDataArray.__abs__` output to `jnp.array`, reducing conversions needed in objective functions.
- Correct color and zorder for line segments when plotting 2D geometries.
- Bug in `Simulation.eps_bounds` that was always setting the lower bound to 1.
- Extended lower limit of frequency range for `Graphene` to zero.
- Improved warnings for `Medium2D`.
- Improved mode solver handling of 1D problem to avoid singular matrix issue.
- Set `colocate=False` automatically in output `FieldMonitor` objects in adjoint plugin, warning instead of erroring for backwards compatibility.
- Validation for `ModeSpec.group_index_step` working incorrectly if the value was set to 1.


## [2.3.3] - 2023-07-28

### Added

### Changed

### Fixed
- Adjoint plugin import failures after `jax.numpy.DeviceArray` -> `jax.Array` in jax 0.4.14.
- Fixed definition of shapely box bounds in Geometry.intersections_2dbox()

## [2.3.2] - 2023-7-21

### Added

### Changed
- Surface integration monitor validator changed to error only if *all* integration surfaces are outside of the simulation domain.
- Inverse design convenience utilities in `plugins.adjoint.utils` for image filtering (`ConicFilter`), projection (`BinaryProjector`), and radius of curvature penalization (`RadiusPenalty`).

### Fixed
- Properly handle sign flip in `ModeMonitor` outputs with `direction="-"` in adjoint plugin.
- Bug in `CustomMedium.grids`, which was giving incorrect grid boundaries at edges.
- Out of date endpoint in `migrate` option of webapi.

## [2.3.1] - 2023-7-14

### Added
- Support for differentiating with respect to `JaxMedium.conductivity`.
- Validating that every surface (unless excluded in ``exclude_surfaces``) of a 3D ``SurfaceIntegrationMonitor`` (flux monitor or field projection monitor) is not completely outside the simulation domain.

### Changed
- Source validation happens before simulation upload and raises an error if no source is present.
- Warning instead of error if structure out of simulation bounds.
- Internal refactor of `ComponentModeler` to simplify logic.
- (Changed effective 2.3.0) Backward-direction mode amplitudes from `ModeMonitors` have a flipped sign compared to previous versions. 
Previously, the amplitudes were computed directly through the dot product ``amp = (mode, field)``, while, since 2.3.0, we use ``amp = (mode, field) / (mode, mode)`` instead.
The modes are already normalized to unit directed flux, such that ``(mode, mode) = 1`` for forward modes, and ``(mode, mode) = -1`` for backward modes
(the dot product of a mode with itself is equal to the flux through the mode plane). Therefore, the change in the ``amp`` formula is equivalent to a sign change for
backward modes. This makes their interpretation more intuitive, since the amplitude is now ``1`` if the recorded field and a mode match exactly. A ``-1`` amplitude means
that the fields match exactly except for a ``pi`` phase shift. This interpretation is also now independent of forward/backward direction.

### Fixed
- Point-like objects correctly appear as single points using `plt.scatter`.
- Cleaner display of `ArrayLike` in docs.
- `ArrayLike` validation properly fails with `None` or `nan` contents.
- Apply finite grid correction to the fields when calculating the Poynting vector from 2D monitors.
- `JaxCustomMedium` properly handles complex-valued permittivity.


## [2.3.0] - 2023-6-30

### Added
- Specification of spatial permittivity distribution of dispersive material using user-supplied data through `CustomPoleResidue`, `CustomSellmeier`, `CustomLorentz`, `CustomDebye`, and `CustomDrude` components.
- `CustomAnisotropicMedium` where each component can take user-supplied data to define spatial permittivity distribution of non-dispersive or dispersive material.
- `Coords.spatial_interp` to interpolate spatial data avoiding pitfalls of `xarray.interp` in directions where there is a single data point.
- All medium types accept `allow_gain` which is `False` by default, but allows the medium to be active if `True`.
- Causality validation in `PoleResidue` model.
- Group index calculation added to mode monitors and available through `ModeData.n_group`.
- Support for `JaxGeometryGroup` in adjoint plugin.
- Support for gradients w.r.t. `FieldMonitor` data from `output_monitors` in adjoint plugin.
- `estimate_cost()` method for `Job` and `Batch`.
- `s3utils.get_s3_sts_token` accepts extra query parameters.
- `plugins.mode.web` to control the server-side mode solver.
- Support for `JaxDataArray.interp`, allowing differentiable linear interpolation.
- Support for `JaxSimulationData.get_intensity()`, allowing intensity distribution to be differentiated in `adjoint` plugin.
- Support for `JaxFieldData.flux` to compute differentiable flux value from `FieldData`.
- All custom medium types accept `subpixel` which is `False` by default, but applies subpixel averaging of the permittivity on the interfaces of the structure (including exterior boundary and intersection interfaces with other structures) if `True` and simulation's `subpixel` is also `True`.

### Changed
- Add `Medium2D` to full simulation in tests.
- `DispersionFitter` and `StableDispersionFitter` unified in a single `DispersionFitter` interface.
- `StableDispersionFitter` deprecated, with stable fitter now being run instead through `plugins.dispersion.web.run(DispersionFitter)`.
- Removed validator from `CustomFieldSource` that ensured data spanned the source geometry. Now the current values are extrapolated outside of the supplied data ranges.
- `CustomMedium` now take fields `permittivity` and `conductivity`. `eps_dataset` will be deprecated in v3.0.
- Moved `CustomMedium._interp` to `Coords.spatial_interp` to be used by custom current sources.
- Adjoint simulations no longer contain unused gradient permittivity monitors, reducing processing time.
- `Batch` prints total estimated cost if `verbose=True`.
- Unified config and authentication.
- Remove restriction that `JaxCustomMedium` must not be a 3D pixelated array.
- Limit total number of input structures in adjoint plugin for performance reasons.
- `ModeSolver` and `ModeSolverMonitor` now contain `direction` field that explicitly specifies the mode propagation direction (default is "+").
- Added an `interpolate` option to `CustomCurrentSource` and `UniformCurrentSource`, which uses linear interpolation to emulate exact placement of the source along directions where the source has zero size, rather than snapping to the nearest grid location, similar to the behavior for `PointDipole`. Default: `True`.

### Fixed
- Plotting 2D materials in `SimulationData.plot_field` and other circumstances.
- More segments in plotting of large cylinder and sphere cross-sections.
- Proper handling of nested list of custom data components in IO, needed for custom dispersive medium coefficients.
- `ElectromagneticFieldData.outer_dot` now works correctly for `FieldData`, not only `ModeSolverData`.
- Fix to the setting of the mode solver PML parameters that produces better results for modes which do not decay in the PML, and fewer spurious modes.
- Fix to single-precision mode solver to do the type conversion on the final matrix only rather than at intermediate steps, which improves accuracy in some cases.
- Improvements to graphene medium fit.
- Schema titles in `ArrayLike` fields.
- Fix `web.estimate_cost` error/time-out for large simulations, it should now always work but may take some time for complex cases.
- A more accurate injection and decomposition of backward propagating waveguide modes in lossy and gyrotropic systems.

## [2.2.3] - 2023-6-15

### Added

### Changed

### Fixed
- Callback URL: "call_back_url" replaced with proper "callbackUrl".

## [2.2.2] - 2023-5-25

### Added

### Changed
- Tidy3D component `.to_hdf5()` and `.from_hdf5()` now accept custom encoder and decoder functions for more flexible IO.

### Fixed
- `JaxDataArrays` are properly handled when reading and writing to file, dramatically reducing the VJP simulation download size in server-side adjoint.
- A bug in a total-field scattered-field (TFSF) validator which was causing unnecessary errors when a TFSF surface intersected with 2D materials.
- CI tests working with binary installation of gdstk instead of compiling from source.

## [2.2.1] - 2023-5-23

### Added

### Changed

### Fixed
- Downgrade `typing_extensions` to `<=4.5.0` to avoid bug with pydantic for python <= 3.9.
- Bug in `ModeSolver` which was causing a preconditioner to be applied even when it is not needed.

## [2.2.0] - 2023-5-22

### Added
- Fully anisotropic medium class (`FullyAnisotropicMedium`) that allows to simulate materials with permittivity and conductivity tensors oriented arbitrary with respect to simulation grid.
- Adjoint processing is done server side by default, to avoid unnecessary downloading of data.
- `JaxPolySlab` in `adjoint` plugin, which can track derivatives through its `.vertices`.
- `run_local` and `run_async_local` options in `tidy3d.plugins.adjoint.web` to provide way to run adjoint processing locally.
- `web.test()` to simply test if the authentication is configured correctly and raise exception otherwise.
- `SimulationTask.get_running_tasks()` to get a list of running tasks from the server.
- Retry for set number of seconds in web functions if internet connection errors.
- Argument `scale` to `ModeSolver.plot_field` to control plot scaling.
- `Simulation.plot_3d()` method to make 3D rendering of simulation.


### Changed
- Perfect electric conductors (PECs) are now modeled as high-conductivity media in both the frontend and backend mode solvers, and their presence triggers the use of a preconditioner to improve numerical stability and robustness. Consequently, the mode solver provides more accurate eigenvectors and field distributions when PEC structures are present.
- Include source amplitude in `amp_time`.
- Increased the maximum allowed estimated simulation data storage to 50GB. Individual monitors with projected data larger than 10GB will trigger a warning.
- `PolySlab.inside` now uses `matplotlib.path.contains_points` for better performance.
- `JaxCustomMedium` accepts a maximum of 250,000 grid cells to avoid slow server-side processing.
- `PolySlab.inside` now uses `matplotlib.path.contains_points`.
- `JaxCustomMedium` accepts a maximum of 250,000 grid cells.
- Logging messages are suppressed and summarized to avoid repetitions.

### Fixed
- Log messages provide the correct caller origin (file name and line number).
- `Medium2D` is removed from the list of allowed options for `Simulation.medium` in the documentation.
- Symmetry works properly in `adjoint` plugin.

## [2.1.1] - 2023-4-25

### Added

### Changed
- `adjoint` plugin now filters out adjoint sources that are below a threshold in amplitude relative to the maximum amplitude of the monitor data, reducing unnecessary processing by eliminating sources that won't contribute to the gradient.
- `web.run_async` uses `Batch` under the hood instead of `asyncio`.

### Fixed
- More helpful error messages from HTTP responses.
- Bug in `_validate_no_structures_pml`, which was using wrong pml thicknesses.
- Broken `callback_url` in webapi.

## [2.1.0] - 2023-4-18

### Added
- Group index calculation added to ModeSpec.
- Waveguide plugin for quickly calculating modes in dielectric waveguides.
- `ElectromagneticFieldData.dot_intep` to calculate mode overlap integrals between modes with different discretizations.
- `ElectromagneticFieldData.mode_area` to calculate the effective mode area.
- `ElectromagneticFieldData.intensity` returns the sum of the squared components of the electric field.
- Group index calculation added to ModeSolver.
- Web interface prints clickable link to task on Tidy3D web interface.
- Allow configuration through API key in python via `tidy3d.web.configure(apikey: str)` function.

### Changed
- `adjoint` plugin now filters out adjoint sources that are below a threshold in amplitude relative to the maximum amplitude of the monitor data, reducing unnecessary processing by eliminating sources that won't contribute to the gradient.
- `ArrayLike` fields use `np.ndarray` internally instead of `TidyNDArray` subclass. Tidy3D objects are no longer hashable, instead, hash the `obj.json()` string.

### Fixed
- `web.Batch` monitoring is more robust, will not raise exception if a job errors or diverges. In this case, the progressbar text will render in red.
- More robust handling for 2D structure validator.

## [2.0.3] - 2023-4-11

### Added

### Changed
- Times logged in `tidy3d.log` during solver run now split into `Solver time` (time-stepping only), `Field projection time` (after the time stepping if any field projection monitors present) and `Data write time` (when the raw data is packaged to disk). Previously, `Solver time` used to include the `Field projection time` and not `Data write time`.

### Fixed
- Port name duplication in smatrix plugin for multimode ports.
- Web functions create the leading directories for the supplied filename if they don't exist.
- Some docstring examples that were giving warnings.
- `web.monitor()` only prints message when condition met.
- PML boxes have non-zero extent along any dimensions where the simulation has 0 size, to fix plotting issues for 2D simulations.
- Improved PEC handling around curved interfaces and structure intersections. Old handling accessible with `subpixel=False` (previously, it was independent of the subpixel setting).
- Fix to field projections sometimes containing `NaN` values.

## [2.0.2] - 2023-4-3

### Added

### Changed

### Fixed
- Bug in web interface when `Simulation` upload was not putting quotes around `td.inf` values.

## [2.0.1] - 2023-3-31

### Added

### Changed
- Default Tidy3D logging level is now set to `'WARNING'`.
- Tidy3D is no longer pip installable from `tidy3d-beta` on PyPI.
- Plugins must be imported from their respective directories, eg. `from tidy3d.plugins.mode import ModeSolver`.
- Removed `Geometry.intersections()`.
- Log level only accepts upper case strings.
- `PolySlab` `reference_plane` is `"middle"` by default.
- Boundary conditions are now `PML()` by default.
- `PointDipole` sources now have a continuous dependence on the source position, as opposed to snapping to Yee grid locations. Behavior is controlled by the `interpolate` argument, set to `True` by default. 
- `smatrix` plugin accepts list of frequencies and returns data as an `xarray.DataArray` instead of a nested `dict`.
- `importlib-metadata` version set to `>= 6.0.0`.

### Fixed
- Helpful error message if user has insufficient credits.

## [1.10.0] - 2023-3-28

### Added
- `TriangleMesh` class for modeling geometries specified by triangle surface meshes, with support for STL file import.
- Total-field scattered-field (TFSF) source which allows angled plane waves to be injected into a finite region of space (the total-field region), such that only scattered fields exist outside this region (scattered-field region).
- `Medium2D` class for surface conductivity model of a 2D material.
- Entries in `material_library` for graphene and some common TMDs.
- Ability to create a 2D representation of a thin 3D material.
- `SimulationData.plot_field` accepts new field components and values, including the Poynting vector.
- `SimulationData.get_poynting_vector` for calculating the 3D Poynting vector at the Yee cell centers.
- Post-init validation of Tidy3D components.
- Validate post-Simulation init to error if any structures have bounds that terminate inside of the PML.
- Validate `slab_bounds` for `PolySlab`.

### Changed
- Tidy3D account authentication done solely through API key. Migration option offered for users with old username / password authentication.
- `export_matlib_to_file` in `material_library` exports material's full name in addition to abbreviation.
- Simpler progress bars for `run_async`.
- Medium property `n_cfl` added to adjust time step size according to CFL condition.
- In the mode solver plugin, regular methods in `solver.py` transformed into classmethods.
- `ArrayLike` types are stored internally as `np.ndarray` and written to json as lists. `constrained_array()` provides way to validate `ArrayLike` values based on `ndim` and `dtype`.
- Pip installing tidy3d automatically creates `~/.tidy3d` directory in home directory.
- Percentage done and field decay determined through http request.
- `SourceTime` plotting methods `.plot()` and `.plot_spectrum()` accept a `val` kwarg, which selects which part of the data (`'real'`, `'imag'`, or `'abs'`) to plot, rather than plotting all at once.

### Fixed
- Bug in remote file transfer when client environment has no correct certificate authority pem file install locally. 
- Tidy3D exceptions inherit from `ValueError` so they are handled properly by pydantic.
- Two unstable materials in `material_library`: `Cu_JohnsonChristy1972` and `Ni_JohnsonChristy1972`. `TiOx_HoribStable` added for improved stability.
- Bug in infinite long cylinder when the `reference_plane` is not at the bottom or the cylinder is slanted.

## [1.9.3] - 2023-3-08

### Fixed
- Allow new `tidy3d.config.logging_level` to accept lower case for backwards compatibility.

## [1.9.2] - 2023-3-08

### Added
- `set_logging_console` allows redirection of console messages to stderr.

### Changed
- Use custom logger to avoid changing global logging state when importing tidy3d.
- Separate logging configuration from custom exception definitions.

## [1.9.1] - 2023-3-06

### Fixed
- Avoiding shapely warning in some cases when checking intersection with an empty shape.
- `Medium.eps_model` error when supplied a list of frequencies rather than a numpy array.
- Set install requirement `rich<12.6.0` to fix double output in webapi functions.

## [1.9.0] - 2023-3-01

### Added
- Specification of relative permittivity distribution using raw, user-supplied data through a `CustomMedium` component.
- Automatic differentiation through `Tidy3D` simulations using `jax` through `tidy3d.plugins.adjoint`.
- New Drude model variants for Gold and Silver in the `material_library`.
- Plugin `ComplexPolySlab` for supporting complex polyslabs containing self-intersecting polygons during extrusion.
- Asynchronous running of multiple simulations concurrently using `web.run_async`.
- Jax-compatible `run_async` in the `adjoint` plugin for efficiently running multi-simulation objectives concurrently and differentiating result.
- Warning in `Simulation.epsilon` if many grid cells and structures provided and slow run time expected as a result.
- `verbose` option in `tidy3d.web` functions and containers. If `False`, there will be no non-essential output when running simulations over web api.
- Warning if PML or absorbing boundaries are used along a simulation dimension with zero size.

### Changed
- Saving and loading of `.hdf5` files is made orders of magnitude faster due to an internal refactor.
- `PolySlab.from_gds` supports `gds_cell` from both `gdspy` and `gdstk`, both packages are made optional requirements.
- Adjoint plugin `JaxCustomMedium` is made faster and can handle several thousand pixels without significant overhead.
- Jax is made an optional requirement. The adjoint plugin supports jax versions 0.3 and 0.4 for windows and non-windows users, respectively.
- Issue a deprecation warning that `Geometry.intersections` will be renamed to `Geometry.intersections_plane` in 2.0.
- Limit some warnings to only show for the first structure for which they are encountered.
- Billed flex unit no longer shown at the end of `web.run` as it may take a few seconds until it is available. Instead, added a `web.real_cost(task_id)` function to get the cost after a task run.
- Refactored `tidy3d.web` for more robustness and test coverage.

### Fixed
- Progressbars always set to 100% when webapi functions are finished.
- Faster handling of `Geometry.intersects` and `Geometry.inside` by taking into account geometry bounds.
- Numpy divide by zero warning in mode solver fixed by initializing jacobians as real instead of complex.
- Bug in validators for 2D objects being in homogeneous media which were looking at the infinite plane in which the objects lie. This can also significantly speed up some validators in the case of many structures.
- Sources and monitors with bend radii are displayed with curved arrows.

## [1.8.4] - 2023-2-13

### Fixed
- Error importing `Axes` type with most recent `matplotlib` release (3.7).

## [1.8.3] - 2023-1-26

### Fixed
- Bug in `Simulation.epsilon` with `coord_key="centers"` in which structures were not rendered.
- Missing `@functools.wrap` in `ensure_freq_in_range` decorator from `medium.py` causing incorrect docstrings.

## [1.8.2] - 2023-1-12

### Added
- Warning if users install via `tidy3d-beta` on pip, from now on, best to use `tidy3d` directly.
- Support for dispersive media in `AnisotropicMedium`

### Changed
- Support shapely version >=2.0 for all python versions.
- Internally refactor `Simulation.epsilon` and move `eps_diagonal` to `Structure` in preparation for future changes.
- Readme displays updated instructions for installing tidy3d (remove beta version mention).

### Fixed
- Field decay warning in mode solver when symmetry present.
- Formatting bug in Tidy3d custom exceptions.

### Removed

## [1.8.1] - 2022-12-30

### Added
- Environment variable `TIDY3D_SSL_VERIFY` to optionally disable SSL authentication (default is `True`).
- Billed FlexUnit cost displayed at the end of `web.monitor`.

### Fixed
- Bug on Windows systems with submitting `CustomFieldSource` data to the server.
- Fix to `FieldData.symmetry_expanded_copy` for monitors with `colocate=True`.

### Changed
- The `Simulation` version updater is called every time a `Simulation` object is loaded, not just `from_file`.
- Boundary specifications that rely on the default `Periodic` boundary now print a deprecation warning, as the default boundaries will change to
 `PML` in Tidy3D 2.0. 

## [1.8.0] - 2022-12-14

### Added
- `CustomFieldSource` that can inject arbitrary source fields.
- `ElectromagneticFieldData.flux` property for data corresponding to 2D monitors, and `ElectromagneticFieldData.dot`
method for computing the overlap integral over two sets of frequency-domain field data.
- Data corresponding to 2D `FieldMonitor` and `FieldTimeMonitor`, as well as to `ModeSolverMonitor`, now also stores `grid_correction` data
 related to the finite grid along the normal direction. This needs to be taken into account to avoid e.g. numerical oscillations of the flux
 with the exact position of the monitor that is due to the interpolation from the grid cell boundaries. These corrections are automatically
 applied when using the `flux` and `dot` methods.
- Resonance finding plugin for estimating resonance frequency and Q-factor of multiple resonances from time-domain data.
 Accessed through `tidy3d.plugins.ResonanceFinder`.
- New `.updated_copy(**kwargs)` method to all tidy3d objects to add a more convenient shortcut to copying an instance with updated fields, 
 i.e. `med.copy(update=dict(permittivity=3.0))` becomes `med.updated_copy(permittivity=3.0)`.
- Test support for python 3.11.
- `sidewall_angle` option for `Cylinder` that allows a `Cylinder` to be tuned into a conical frustum or a cone.
- `reference_plane` for `PolySlab` that provides options to define the vertices at the bottom, middle, or top of the `PolySlab`.
- Automesh generation: `MeshOverrideStructure` that allows for a direct grid size specification in override structures,
 and `dl_min` that bounds the minimal grid size.
- More material models to the material database such as gold from Olman2012.
- In `AdvancedFitterParam` for `StableDispersionFitter`, `random_seed` option to set the random seed,
 and `bound_f_lower` to set the lower bound of pole frequency.
- Introduced the option to project fields at near, intermediate, and far distances using an exact Green's function formalism which does not
 make far-field approximations. This can be enabled in any `AbstractFieldProjectionMonitor` by setting `far_field_approx=False`. A tutorial notebook
 as a comprehensive reference for field projections was added to the documentation.
- Tracking of modes in `ModeSolverData` based on overlap values, controlled through `ModeSpec.track_freq`.
- Native broadband support for `GassuainBeam` `AstigmaticGaussianBeam`, and `ModeSource` through the `num_freqs` argument.
- Apodization option for frequency-domain monitors to ignore temporal data in the beginning and/or end of a simulation


### Changed
- Minimum flex unit charge reduced from `0.1` to `0.025`.
- Default Courant factor was changed from `0.9` to `0.99`.
- A point dipole source placed on a symmetry plane now always has twice the amplitude of the same source in a simulation without the 
 symmetry plane, as expected by continuity with the case when the dipole is slightly off the symmetry plane, in which case 
 there are effectively two dipoles, the original one and its mirror image. Previously, the amplitude was only doubled for dipoles polarized normal 
 to the plane, because of Yee grid specifics.
- `FluxMonitor` and `FluxTimeMonitor` no longer snap fields to centers, but instead provide continuous interpolation of the flux over the
 exact geometry of the monitor.
- Major refactor to internal handling of data structures, including pure `Dataset` components that do not depend on other `Tidy3D` components and may
 therefore be used to define custom data in `Tidy3D` models.
- Speed and memory usage improvement when writing and reading Tidy3d models to and from `.hdf5` files.
- Writing `Tidy3D` models containing custom data to `.json` file will log a warning and exclude the raw data from the file for performance reasons.
- Material database reorganization and fixing a few references to the dispersion data.
- The name `Near2Far` has been replaced with `FieldProjection`. For example, `Near2FarAngleMonitor` is now `FieldProjectionAngleMonitor`.
- The API for far field projections has been simplified and several methods have now become properties. 
 For example, the radar cross section is now accessed as `.radar_cross_section`, not `.radar_cross_section()`.
- Added a method `renormalize_fields` to `AbstractFieldProjectionData` to re-project far fields to different projection distances.
- The API for `DiffractionData` was refactored to unify it with the API for `AbstractFieldProjectionData`.
- The user no longer needs to supply `orders_x` and `orders_y` when creating a `DiffractionMonitor`; all allowed orders are automatically
generated and returned in the resulting `DiffractionData`.
- The user no longer needs to supply a `medium` when creating a `DiffractionMonitor` or any `AbstractFieldProjectionMonitor`; the medium through
which fields are to be projected is now determined automatically based on the medium in which the monitor is placed.
- The following attributes of `AbstractFieldProjectionMonitor` are now properties rather than methods:
`fields_spherical`, `fields_cartesian`, `power`, `radar_cross_section`.


### Fixed
- Some issues in `DiffractionMonitor` that is not `z`-normal that could lead to solver errors or wrong results.
- Bug leading to solver error when `Absorber` boundaries with `num_layers = 0` are used.
- Bug leading to solver error when a `FieldMonitor` crosses a `BlochBoundary` and not all field components are recorded.
- When running a `Batch`, `path_dir` is created if not existing.
- Ignore shapely `STRtree` deprecation warning.
- Ignore x axis when plotting 1D `Simulation` cross sections to avoid plot irregularities.
- Local web api tests.
- Use Tidy3D logger for some warnings that used to use default python logging.
 
### Changed

- Replaced `gdspy` dependency with `gdstk`.

## [1.7.1] - 2022-10-10

### Added
- `medium` field in `DiffractionMonitor` for decomposition of fields that are not in vacuum.

### Fixed
- Bug in meshing an empty simulation with zero size along one dimension.
- Bug causing error in the solver when a `PermittivityMonitor` is present in the list of monitors and is not at the end of the list.

## [1.7.0] - 2022-10-03

### Added
- `DiffractionMonitor` to compute the power amplitude distribution in all diffraction orders in simulations of periodic structures.
- `PolySlab` can be oriented along `x` or `y`, not just `z`.

### Removed
- Loading components without validation no longer supported as it is too unpredictable.
- Webplots plugin was removed as it was cumbersome to maintain and no longer used in web UI.

## [1.6.3] - 2022-9-13

### Added
- Type field for `DataArray` subclasses written to `hdf5`.

### Fixed
- Docstring for `FluxMonitor` and `FluxTimeMonitor`.

### Removed
- Explicit error message about `grid_size` deprecation.

## [1.6.2] - 2022-9-6

### Added
- Support for `Near2Far` monitors in the presence of simulation symmetries.

### Fixed
- Bug in 3D `Near2Far` monitors where surfaces defined in `exclude_surfaces` will no actually be excluded.
- Bug in getting angles from `k`-space values in `Near2FarKSpaceMonitor`.
- Bug in `SimulationData.plot_field` when getting the position along the normal axis for a 2D plot.

## [1.6.1] - 2022-8-31

### Fixed
- Bug in new simulation upload on Windows machines.

## [1.6.0] - 2022-8-29

### Added
- New classes of near-to-far monitors for server-side computation of the near field to far field projection.
- Option to exclude `DataArray` Fields from a `Tidy3dBaseModel` json.
- Option to save/load all models to/from `hdf5` format.
- Option to load base models without validation.
- Support negative sidewall angle for slanted `PolySlab`-s.
- Option to specify only a subset of the S-matrix to compute in the S-matrix plugin, as well as to provide mapping between elements (due to symmetries).
- More Lorentz-Drude material models to the material database.

### Fixed
- Raise a more meaningful error if login failed after `MAX_ATTEMPTS`.
- Environment login credentials set to `""` are now ignored and credentials stored to file are still looked for.
- Improved subpixel coefficients computation around sharp edges, corners, and three-structure intersections.

### Changed
- Major refactor of the way data structures are used internally.
- `ModeFieldMonitor` -> `ModeSolerMonitor` with associated `ModeSolverData`. `ModeSolverData` is now also stored internally in `ModeSolver`, 
 and the `plot_field` method can be called directly from `ModeSolver` instead of `ModeSolverData`.
- Field data for monitors that have a zero size along a given dimension is now interpolated to the exact `monitor.center` along that dimension.
- Removed `nlopt` from requirements, user-side material fitting now uses `scipy`.
- New Field `normalize_index` in `Simulation` - used to be input parameter when loading simulation data. A given `SimulationData` 
 can still be renormalized to a different source later on using the new `SimulationData.renormalize`.
- `FluxMonitor` and `FluxTimeMonitor`-s can now have a 3D box geometry, in which case the flux going out of all box surfaces is computed (optionally, 
 some surfaces can be excluded).
- Frequency-domain monitors require a non-empty list of frequencies.
- Reduced the minimum flex unit cost to run a simulation to `0.1`.
- Reduced the premium cost for dispersive materials in typical cases.
- Added a cost for monitors that should be negligible in typical cases but affects large monitors that significantly slow down the simulation.

## [1.5.0] - 2022-7-21

### Fixed
- Bug in computing the `bounds` of `GeometryGroup`.
- Bug in auto-mesh generation.

### Added
- Ability to compute field projections server-side.

### Changed
- All Tidy3D components apart from data structures are now fully immutable.
- Stopped support for python 3.6, improved support for python 3.10.
- Web material fitter for lossless input data (no `k` data provided) will now return a lossless medium.
- `sort_by` changed to `filter_pol` in `ModeSpec`.
- `center` no longer a field of all `Geometry` components, instead only present when needed, 
 removed in `PolySlab` and `GeometryGroup`. `Planar` geometries no longer have a mandatory `length` field, but 
 have `center_axis` and `lengt_axis` properties for the center and length along the extrusion axis. `PolySlab` now defined exclusively through `slab_bounds`, 
 while `Cylinder` through `center` and `length`.
- In mode solver, allow precision to switch between double and single precision.
- Near-to-far transformation tool is no longer a plugin, but is now part of Tidy3D's new core data structures


## [1.4.1] - 2022-6-13

### Fixed
- Bug in plotting polarization of a normal incidence source for some `angle_phi`.
- Bloch vector values required to be real rather than complex.
- Web security mitigation.

## [1.4.0] - 2022-6-3

### Fixed
- Bug in plotting when alpha is turned off in permittivity overlay.
- Bug in plotting polarization of an angled incidence source (S,P -> P,S).
- Throw warning if user tries to download data instances in `yaml` or `json` format. 
- Arrow length plotting issues for infinite sources.
- Issues with nonuniform mesh not extending exactly to simulation boundaries.

### Added
- Bloch periodic boundary conditions, enabling modeling of angled plane wave.
- `GeometryGroup` object to associate several `Geometry` instances in a single `Structure` leading to improved performance for many objects.
- Ability to uniquely specify boundary conditions on all 6 `Simulation` boundaries.
- Options in field monitors for spatial downsampling and evaluation at Yee grid centers.
- `BatchData.load()` can load the data for a batch directly from a directory.
- Utility for updating `Simulation` objects from old versions of `Tidy3d` to current version.
- Explicit `web.` functions for downloading only `simulation.json` and `tidy3d.log` files.

### Changed
- `Batch` objects automatically download their json file upon `download` and `load`.
- Uses `shapely` instead of `gdspy` to merge polygons from a gds cell.
- `ComponentModeler` (S matrix tool) stores the `Batch` rather than the `BatchData`.
- Custom caching of properties to speed up subsequent calculations.
- Tidy3D configuration now done through setting attributes of `tidy3d.config` object.

## [1.3.3] - 2022-5-18

### Fixed

 - Bug in `Cylinder.inside` when `axis != 2`.

### Added

 - `AstigmaticGaussianBeam` source.

### Changed

 - Internal functions that may require hashing the simulation many times now use a `make_static` decorator. This pre-computes the simulation hash and stores it,
 and makes sure that the simulation has not changed at the beginning and end of the function execution.
 - Speeding up initialization of `PolySlab` when there is no dilation or slant angle.
 - Allow customizing data range that colormap covers in `plot_field`.
 - Speeding up of the automatic grid generation using Rtree and other improvements.
 - Better handling of http response errors.
 - In `web.monitor`, the estimated cost is only displayed when available; avoid "Unable to get cost" warning.
 - In `PolySlab.from_gds`, the selected polygons are first merged if possible, before the `PolySlab`-s are made. This avoids bugs e.g. in the case of slanted walls.

## [1.3.2] - 2022-4-30

### Fixed

 - Bug in nonuniform mesh where the simulation background medium may be taken into account if higher than other structures overriding it.

## [1.3.1] - 2022-4-29

### Added

### Changed

 - The `copy()` method of Tidy3D components is deep by default.
 - Maximum allowed number of distinct materials is now 65530.

### Fixed

 - Monitor/source opacity values also applied to associated arrows.
 - Auto meshing in the presence of symmetries ignores anything outside of the main symmetry quadrant.
 - If an interface is completely covered by another structure, it is ignored by the mesher.

## [1.3.0] - 2022-4-26

### Added

- New `grid_spec` Field in `Simulation` that allows more flexibility in defining the mesh.
- `GridSpec1d` class defining how the meshing along each dimension should be done, with subclasses `UniformGrid` and `CustomGrid` that cover the functionality 
  previously offered by supplying a float or a list of floats to `Simulation.grid_size`. New functionality offered by `AutoGrid` subclass, with the 
  mesh automatically generated based on the minimum required steps per wavelength.
- New `PointDipole` source.
- Opacity kwargs for monitor and source in `sim.plot`.
- Separated `plotly`-based requirements from core requirements file, can be added with `"pip install tidy3d-beta[plotly]"`.

### Changed
- `Simulation.grid_spec` uses the default `GridSpec`, which has `AutoGrid(min_steps_per_wvl=10)` in each direction. To initialize a `Simulation` then it is no 
  longer needed to provide grid information, if sources are added to the simulation. Otherwise an error will be raised asking to provide a wavelength for the auto mesh.
- `VolumeSource` is now called `UniformCurrentSource`.
- S-matrix module now places the monitors exactly at the port locations and offsets the source slightly for numerical reasons (more accurate).
- Fixed bug in `PolySlab` visualization with sidewalls.
- Inheritance structure of `Source` reorganized.
- Better handling of only one `td.inf` in `Box.from_bounds`.
- Added proper label to intensity plots.
- Made all attributes `Field()` objects in `data.py` to clean up docs.
- Proper handling of `Medium.eps_model` at frequency of `td.inf` and `None`.

### Removed
- `Simulation.grid_size` is removed in favor of `Simulation.grid_spec`.

## [1.2.2] - 2022-4-16

### Added
- `SimulationDataApp` GUI for visualizing contents of `SimulationData` in `tidy3d.plugings`.
- `SimulationPlotly` interface for generating `Simulation.plot()` figures using `plotly` instead of `matplotlib`.
- New `PermittivityMonitor` and `PermittivityData` to store the complex relative permittivity as used in the simulation.
- The maximum credit cost for a simulation can now be queried using `web.estimate_cost`. It is also displayed by default during `web.upload`.

### Changed
- Faster plotting for matplotlib and plotly.
- `SimulationData` normalization keeps track of source index and can be normalized when loading directly from .hdf5 file.
- Monitor data with symmetries now store the minimum required data to file and expands the symmetries on the fly.
- Significant speedup in plotting complicated simulations without patch transparency.
- When a list of `dl` is provided as a `grid_size` along a given direction, the grid is placed such that the total size `np.sum(dl)` is centered at the simulation center.
  Previously, a grid boundary was always placed at the simulation center.

## [1.2.1] - 2022-3-30

### Added

### Changed

- `webapi` functions now only authenticate when needed.
- Credentials storing folder only created when needed.
- Added maximum number of attempts in authentication.
- Made plotly plotting faster.
- Cached Simulation.medium and Simulation.medium_map computation.

## [1.2.0] - 2022-3-28

### Added
- `PolySlab` geometries support dilation and angled sidewalls.
- Percent done monitoring of jobs running longer than 10 seconds.
- Can use vectorized spherical coordinates in `tidy3d.plugins.Near2Far`.
- `ModeSolver` returns a `ModeSolverData` object similar to `SimulationData`, containing all the information about the modes.
- `ModeFieldMonitor` and `ModeFieldData` allow the results of a mode solve run server-side to be stored.
- Plotting of `ModeFieldData` fields in `SimulationData.plot_field` and `ModeSolverData.plot_field`.
- Ordering of modes by polarization fraction can be specified in `ModeSpec`.
- Angled mode sources.

### Changed
- Significant speed improvement for `Near2Far` calculations.
- `freq` no longer passed to `ModeSolver` upon init, instead a list of `freqs` passed to `ModeSolver.solve`.
- Mode solver now returns `ModeSolverData` object containing information about the mode fields and propagation constants as data arrays over frequency and mode index.
- Reorganized some of the internal `Source` classes.
- Major improvements to `Batch` objects. `Batch.run()` returns a `BatchData` object that maps `task_name` to `SimulationData`.
- Infinity stored as `str` in json outputs, conforming to json file specifications.
- No longer need to specify one of `x/y/z` in `SimulationData.plot_field` if the monitor has a zero-sized dimension.
- `Simulation.run_time` but must be > 0 to upload to server.

## [1.1.1] - 2022-3-2

### Added

### Changed

- Fixed issue where smatrix was not uploaded to pyPI.

## [1.1.0] - 2022-3-1

### Added

- `Simulation` symmetries now fully functional.
- Ability to perform near-to-far transformations from multiple surface monitors oriented along the x, y or z directions using `tidy3d.plugins.Near2Far`.
- `tidy3d.plugins.ComponentModeler` tool for scattering matrix calculations.

### Changed

- Major enhancements to near field to far field transformation tool: multiple monitors supported with arbitrary configuration, user control over sampling point density.
- Fixed visualization bug in symmetry.

## [1.0.2] - 2022-2-24

### Added
 - Clarified license terms to not include scripts written using the tidy3d python API.
 - Simulation symmetries are now enabled but currently only affect the mode solver, if the mode plane lies on the simulation center and there's a symmetry.
 - Validator that mode objects with symmetries are either entirely in the main quadrant, or lie on the symmetry axis.
- `Simulation.plotly()` makes a plotly figure of the cross section.
- Dispersion fitter can parse urls from refractiveindex.info
 - Clarified license terms to not include scripts written using the tidy3d python API.

### Changed
- Fixed a bug in python 3.6 where polyslab vertices loaded differently from file.

## [1.0.1] - 2022-2-16

### Added
- `Selmeier.from_dispersion()` method to quickly make a single-pole fit for lossless weakly dispersive materials.
- Stable dispersive material fits via webservice.
- Allow to load dispersive data directly by providing URL to txt or csv file
- Validates simulation based on discretized size.

### Changed
- `Polyslab.from_gds` returns a list of `PolySlab` objects imported from all polygons in given layer and dtype, can optionally specify single dtype.
- Warning about structure close to PML disabled if Absorber type.
- Source dft now ignores insignificant time amplitudes for speed.
- New color schemes for plots.

## [1.0.0] - 2022-1-31

### Added
- Stable dispersive material fits via webservice.

### Changed
- Refined and updated documentation.

## [0.2.0] - 2022-1-29

### Added

- `FieldMonitor.surface()` to split volume monitors into their surfaces.
- Units and annotation to data.
- Faster preprocessing.
- Web authentication using environment variables `TIDY3D_USER` and `TIDY3D_PASS`.
- `callback_url` in web API to put job metadata when a job is finished.
- Support for non uniform grid size definition.
- Gaussian beam source.
- Automated testing through tox and github actions.

## [0.1.1] - 2021-11-09
### Added

- PML parameters and padding Grid with pml pixels by [@momchil-flex](https://github.com/momchil-flex) in #64
- Documentation by [@tylerflex](https://github.com/tylerflex) in #63
- Gds import from [@tylerflex](https://github.com/tylerflex) in #69
- Logging, by [@tylerflex](https://github.com/tylerflex) in #70
- Multi-pole Drude medium by [@weiliangjin2021](https://github.com/weiliangjin2021) in #73
- Mode Solver: from [@tylerflex](https://github.com/tylerflex) in #74
- Near2Far from [@tylerflex](https://github.com/tylerflex) in #77

### Changed
- Separated docs from [@tylerflex](https://github.com/tylerflex) in #78

## [0.1.0] - 2021-10-21

### Added
- Web API implemented by converting simulations to old tidy3D

## Alpha Release Changes

### 22.1.1
- Solver speed improvement (gain depending on simulation + hardware details).
- Bringing the speed of the non-angled mode solver back to pre-21.4.2 levels.

### 21.4.4
- Improvements to subpixel averaging for dispersive materials.
- Enabled web login using environment variables ``TIDY3D_USER`` and ``TIDY3D_PASS``.

### 21.4.3
- Bugfix when running simulation with zero ``run_time``.
- More internal logging.
- Fixed unstable ``'Li1993_293K'`` variant of ``cSi`` in the material library.

### 21.4.2.2
- Bugfix when downloading data on Windows.
- Bugfix in material fitting tool when target tolerance is not reached.

### 21.4.2
- New Gaussian beam source and `example usage <examples/GratingCoupler.html>`__.
- Modal sources and monitors in bent and in angled waveguides with `tutorial <examples/Modes_bent_angled.html>`__.
- Nyquist-limit sampling in frequency-domain monitors (much faster without loss of accuracy).
- Support for Drude model of material dispersion.
- Small bugfixes to some of the other dispersion models.
- PEC boundaries applied by default at the truncation of any boundary with PML, avoiding potential
   issues with using periodic boundaries under the PML instead.
- Source normalization no longer adding a spurious frequency-dependent phase to the fields.
- Fixed bug in unpacking monitor fields with symmetries and ``interpolate=False``.
- Lots of streamlining on the backend side.

### 21.4.1
- Fixed bug with zero-size monitor plotting.
- Fixed bug with empty simulation run introduced in 21.4.0.

### 21.4.0
- A few small fixes.


### 21.3.1.6
- Fixed nonlinear constraint in dispersive material fitting tool.
- Fixed potential issue when a monitor stores neither `'E'` nor `'H'`.
- Fixed some backwards compatibility issues introduced in 21.3.1.5.


### 21.3.1.5
 - Frequency monitors can now optionally store the complex permittivity at the same locations where 
   the E-fields are recorded, at the monitor frequencies.
 - Frequency monitors now also have an `'interpolate'` keyword, which defaults to `True` and 
   reproduces the behavior of previous versions. If set to `False`, however, the raw fields 
   evaluated at their corresponding Yee grid locations are returned, instead of the fields interpolated 
   to the Yee cell centers. This also affects the returned permittivity, if requested.
 - Reorganized internal source time dependence handling, enabling more complicated functionality 
   in the future, like custom source time.
 - Total field in the simulation now sampled at the time step of the peak of the source time dependence,
   for better estimation of the shutoff factor.
 - A number of bug fixes, especially in the new plotting introduced in 21.3.1.4.

### 21.3.1.4
- Reorganized plotting:
- Speeding up structure visualizations.
- Structures now shown based on primitive definitions rather than grid discretization. This 
    then shows the physical structures but not what the simulation "sees". Will add an option to 
    display the grid lines in next version.
- Bumped down matplotlib version requirement to 3.2 and python version requirement to 3.6.
- Improved handling of PEC interfaces.- Reorganized and extended internal logging.
- Added ``tidy3d.__version__``.
- A number of fixes to the example notebooks and the colab integration.

### 21.3.1.3
- Bumping back python version requirement from 3.8 to 3.7.

### 21.3.1.2
- Hotfix to an internal bug in some simulation runs.

### 21.3.1.1
- New dispersion fitting tool for material data and accompanying `tutorial <examples/Fitting.html>`__.
- (`beta`) Non-uniform Cartesian meshing now supported. The grid coordinates are provided
   by hand to `Simulation`. Next step is implementing auto-meshing.
- `DispersionModel` objects can now be directly used as materials.
- Fixed bug to `Cylinder` subpixel averaging.
- Small bugs fixes/added checks for some edge cases.

### 21.3.1.0
- Rehash of symmetries and support for mode sources and monitors with symmetries.
- Anisotropic materials (diagonal epsilon tensor).
- Rehashed error handling to output more runtime errors to tidy3d.log.
- Job and Batch classes for better simulation handling (eventually to fully replace webapi functions).
- A large number of small improvements and bug fixes.

[Unreleased]: https://github.com/flexcompute/tidy3d/compare/v2.10.1...develop
[2.10.1]: https://github.com/flexcompute/tidy3d/compare/v2.10.0...v2.10.1
[2.10.0]: https://github.com/flexcompute/tidy3d/compare/v2.9.3...v2.10.0
[2.9.3]: https://github.com/flexcompute/tidy3d/compare/v2.9.2...v2.9.3
[2.9.2]: https://github.com/flexcompute/tidy3d/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/flexcompute/tidy3d/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/flexcompute/tidy3d/compare/v2.8.5...v2.9.0
[2.8.5]: https://github.com/flexcompute/tidy3d/compare/v2.8.4...v2.8.5
[2.8.4]: https://github.com/flexcompute/tidy3d/compare/v2.8.3...v2.8.4
[2.8.3]: https://github.com/flexcompute/tidy3d/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/flexcompute/tidy3d/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/flexcompute/tidy3d/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/flexcompute/tidy3d/compare/v2.7.9...v2.8.0
[2.7.9]: https://github.com/flexcompute/tidy3d/compare/v2.7.8...v2.7.9
[2.7.8]: https://github.com/flexcompute/tidy3d/compare/v2.7.7...v2.7.8
[2.7.7]: https://github.com/flexcompute/tidy3d/compare/v2.7.6...v2.7.7
[2.7.6]: https://github.com/flexcompute/tidy3d/compare/v2.7.5...v2.7.6
[2.7.5]: https://github.com/flexcompute/tidy3d/compare/v2.7.4...v2.7.5
[2.7.4]: https://github.com/flexcompute/tidy3d/compare/v2.7.3...v2.7.4
[2.7.3]: https://github.com/flexcompute/tidy3d/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/flexcompute/tidy3d/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/flexcompute/tidy3d/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/flexcompute/tidy3d/compare/v2.6.4...v2.7.0
[2.6.4]: https://github.com/flexcompute/tidy3d/compare/v2.6.3...v2.6.4
[2.6.3]: https://github.com/flexcompute/tidy3d/compare/v2.6.2...v2.6.3
[2.6.2]: https://github.com/flexcompute/tidy3d/compare/v2.6.1...v2.6.2
[2.6.1]: https://github.com/flexcompute/tidy3d/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/flexcompute/tidy3d/compare/v2.5.2...v2.6.0
[2.5.2]: https://github.com/flexcompute/tidy3d/compare/v2.5.1...v2.5.2
[2.5.1]: https://github.com/flexcompute/tidy3d/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/flexcompute/tidy3d/compare/v2.4.3...v2.5.0
[2.4.3]: https://github.com/flexcompute/tidy3d/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/flexcompute/tidy3d/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/flexcompute/tidy3d/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/flexcompute/tidy3d/compare/v2.3.3...v2.4.0
[2.3.3]: https://github.com/flexcompute/tidy3d/compare/v2.3.2...v2.3.3
[2.3.2]: https://github.com/flexcompute/tidy3d/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/flexcompute/tidy3d/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/flexcompute/tidy3d/compare/v2.2.3...v2.3.0
[2.2.3]: https://github.com/flexcompute/tidy3d/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/flexcompute/tidy3d/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/flexcompute/tidy3d/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/flexcompute/tidy3d/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/flexcompute/tidy3d/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/flexcompute/tidy3d/compare/v2.0.3...v2.1.0
[2.0.3]: https://github.com/flexcompute/tidy3d/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/flexcompute/tidy3d/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/flexcompute/tidy3d/compare/v1.10.0...v2.0.1
[1.10.0]: https://github.com/flexcompute/tidy3d/compare/v1.9.3...v1.10.0
[1.9.3]: https://github.com/flexcompute/tidy3d/compare/v1.9.2...v1.9.3
[1.9.2]: https://github.com/flexcompute/tidy3d/compare/v1.9.1...v1.9.2
[1.9.1]: https://github.com/flexcompute/tidy3d/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/flexcompute/tidy3d/compare/v1.8.4...v1.9.0
[1.8.4]: https://github.com/flexcompute/tidy3d/compare/v1.8.3...v1.8.4
[1.8.3]: https://github.com/flexcompute/tidy3d/compare/v1.8.2...v1.8.3
[1.8.2]: https://github.com/flexcompute/tidy3d/compare/v1.8.1...v1.8.2
[1.8.1]: https://github.com/flexcompute/tidy3d/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/flexcompute/tidy3d/compare/v1.7.1...v1.8.0
[1.7.1]: https://github.com/flexcompute/tidy3d/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/flexcompute/tidy3d/compare/v1.6.3...v1.7.0
[1.6.3]: https://github.com/flexcompute/tidy3d/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/flexcompute/tidy3d/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/flexcompute/tidy3d/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/flexcompute/tidy3d/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/flexcompute/tidy3d/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/flexcompute/tidy3d/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/flexcompute/tidy3d/compare/v1.3.3...v1.4.0
[1.3.3]: https://github.com/flexcompute/tidy3d/compare/v1.3.2...v1.3.3
[1.3.2]: https://github.com/flexcompute/tidy3d/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/flexcompute/tidy3d/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/flexcompute/tidy3d/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/flexcompute/tidy3d/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/flexcompute/tidy3d/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/flexcompute/tidy3d/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/flexcompute/tidy3d/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/flexcompute/tidy3d/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/flexcompute/tidy3d/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/flexcompute/tidy3d/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/flexcompute/tidy3d/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/flexcompute/tidy3d/compare/0.1.1...v0.2.0
[0.1.1]: https://github.com/flexcompute/tidy3d/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/flexcompute/tidy3d/releases/tag/0.1.0
