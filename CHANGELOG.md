# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Autograd support for local field projections using `FieldProjectionKSpaceMonitor`.
- Function `components.geometry.utils.flatten_groups` now also flattens transformed groups when requested.

### Fixed
- Regression in local field projection leading to incorrect results for `far_field_approx=True`.
- Bug when differentiating with respect to `Cylinder.center`.


## [2.7.6] - 2024-10-30

### Added
- Users can pass the `shading` argument in the `SimulationData.plot_field` to `Xarray.plot` method. When `shading='gouraud'`, the image is interpolated, producing a smoother visualization.
- Users can manually specify the background medium for a structure to be used for geometry gradient calculations by supplying `Structure.background_permittivity`. This is useful when there are overlapping structures or structures embedded in other mediums.
- Autograd functions can now be called directly on `DataArray` (e.g., `np.sum(data_array)`) in objective functions.
- Automatic differentiation support for local field projections with `FieldProjectionAngleMonitor` and `FieldProjectionCartesianMonitor` using `FieldProjector.project_fields(far_field_monitor)`.

### Changed
- Improved autograd tracer handling in `DataArray`, resulting in significant speedups for differentiation involving large monitors.
- Triangulation of `PolySlab` polygons now supports polygons with collinear vertices.

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

[Unreleased]: https://github.com/flexcompute/tidy3d/compare/v2.7.6...develop
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
