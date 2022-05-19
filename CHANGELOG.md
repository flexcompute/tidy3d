All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

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

 - The `copy()` method of Tidy3d components is deep by default.
 - Maximum allowed number of distinct materials is now 65530.

### Fixed

 - Monitor/source opacity values also applied to associated arrows.
 - Auto meshing in the presence of symmetries ignores anything outside of the main symmetry quadrant.
 - If an interface is completely covered by another structure, it is ignored by the mesher.

## [1.3.0] - 2022-4-26

### Added

- New `grid_spec` Field in `Simulation` that allows more flexibility in defining the mesh.
- `GridSpec1d` class defining how the meshing along each dimension should be done, with sublcasses `UniformGrid` and `CustomGrid` that cover the functionality 
  previously offered by supplying a float or a list of floats to `Simulation.grid_size`. New functionality offered by `AutoGrid` subclass, with the 
  mesh automatically generated based on the minimum required steps per wavelength.
- New `PointDipole` source.
- Opacity kwargs for monitor and source in `sim.plot`.
- Separated `plotly`-based requirements from core requrements file, can be added with `"pip install tidy3d-beta[plotly]"`.

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
- Added maximum number of attemtps in authentication.
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
- Loggin by [@tylerflex](https://github.com/tylerflex) in #70
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

[Unreleased]: https://github.com/flexcompute/tidy3d/compare/v1.3.2...develop
[1.3.3]: https://github.com/flexcompute/tidy3d/compare/v1.3.2...1.3.3
[1.3.2]: https://github.com/flexcompute/tidy3d/compare/v1.3.1...1.3.2
[1.3.1]: https://github.com/flexcompute/tidy3d/compare/v1.3.0...1.3.1
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
