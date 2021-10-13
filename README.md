# Tidy3D Client Revamp

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-client-revamp/HEAD?filepath=notebooks)

## Flow

### Client (Pre)

- Make `tidy3d.components` to define simulation.
	- Using builtin `tidy3d` imports (`td.PlaneWave`, `td.ModeMonitor`, etc.)
	- Using `tidy3d.plugins` to construct more specialized components (mode solver, dispersion fitter, etc.).
- Create a `td.Simulation` object containing all simulation parameters (pydantic will automaticall validate all components).
- Upload `Simulation` to server using `tidy3d.web`.
	- Export `Simulation` to a .json file format with `Simulation.json()`
	- Save as .json file.
	- Upload to server using http request, authenticate, etc.
	- Use `plugins.batch_processor` to submit batches of simulations.
- Manage task with `tidy3d.web`
	- Run task explicitly (if draft).
	- Monitor progress.
	- Cancel / delete task.

### Core

- Validate received .json file using `Simulation` schema file from our local copy of client.
- Load `.json` into `Simulation` object using local copy of client.
- Preprocess `Simulation` into files needed for core solver (need momchil's help).
- Run solver, export solver data files (need momchil's help).
- Postprocess solver data files into `SimulationData` containing `MonitorData` using definitions from local copy of client (need momchil's help).
- Export data file visualizations as .html to display on browser.
- Store data objects as .hdf5 files on server for download.

### Client (Post)

- Monitor solver progress, manage tasks.
- Load results into `SimulationData` and `MonitorData` objects locally.
	- Download .hdf5 files locally.
	- Open .hdf5 files as `SimulationData` and `MonitorData`.
- Visualize / post process results
	- data management using `xarray.DataArray` operations like `.sel()`, `.iterpolate()`.
	- simple plots with `MonitorData.plot()`
	- interactive plots with `MonitorData.visualize()` 
	- plugins for `near2far`, spectral analysis.

## Roadmap (113.5 days = 16.2 weeks = 3.7 months ~ jan 1)

**Bold** = in progress this week-ish.

### Stage 1: Definition

#### Component / API definition

- [x] Organize repo in basic structure we want moving forward (.5 days)
- [x] Decide on schema for all fields in tidy3d public (1 week)
	- [x] Simulation
	- [x] Geometry
	- [x] Medium
	- [x] MaterialLibrary
	- [x] Structure
	- [x] Source
	- [x] Monitor
	- [x] Mode
- [x] Write documentation explaining all components & design choices.
- [x] Write boilerplate for all plugins / packages (1 week)
	- [x] Dispersion fit (create `DispersiveMedium()` objects from nk data)
	- [x] Mode Solver (create specifications for waveguide modes using the solver)
	- [x] Batch Processor (submit and manage batches of jobs, used by below)
	- [x] Component Modeler (wrapper for analyzing S-matrix of device using ports)
	- [x] Device Optimizer (wrapper for parameter sweep / adjoint optimization routines)
---

#### Web Emulation

- [x]  Write simple tidy3d_core
	- [x]  Preprocess simulation.json into Simulation
	- [x]  "solver" that generates fake data.
	- [x]  Postprocess solver data into tidy3d data objects.
	- [x]  Export and load Solver data into files for download.
- [x]  Write emulated webAPI to transfer files back and forth (1 day)
	- [x] Make `task` its own dataclass.
	- [x] Create basic mock data creation.
	- [x] Create set of basic API calls.
	- [x] Simple batch / job interface.
---

### Stage 2: Integration

#### Monitor Data
- [x] Define monitor data.
- [x] Export and Load
- [x] Expose especially useful xarray methods to MonitorData API.
- [x] Define Permittivity Monitor
- [x] Expose arguments to each MonitorData
- [x] How to more cleanly specify what fields are required to construct each MonitorData?
---

#### Plugins

- [x] Flesh out all details for plugins
	- [x] Make dispersion fitting tool compatible
	- [x] Make mode solver compatible
	- [x] Batch processor <- done in webapi
	- [x] Near2Far

---

#### IO
- [x] Supply other ways to create simulation files (2 days)
	- [x] yaml -> json conversion 
	- [x] command line interface to submit yaml or json.
---

#### Visualization
- [x] write all visuaization routines
	- [x] MonitorData
	- [x] Basic sliding window for cross section.
	- [x] Structure cross section
	- [x] Simulation epsilon discretization from structure polygons.
	- [x] Medium-aware structure / simulation plotting.
	- [x] Add Monitor / Source plotting
	- [x] Visualize symmetry in ``Simulation.plot``
	- [x] Add "plot" to all tidy3d objects.
		- [x] source time
		- [x] medium
		- [x] sampler
	- [x] Overlay structures onto fields.
	- [x] add .plot method to `SimulationData`
---

#### Solver Integration

- [x] Add version to simulation.json
- [ ] **Make tidy3d_core functional with existing (or slightly modified) solver.**
	- [x] Load simulation.json into `tidy3d.Simulation`.
	- [x] IO SimulationData
	- [ ] Process Simulation to solver inputs. (momchil)
	- [ ] Write solver outputs to SimulationData. (momchil)
- [x] Convert to old .json conversion to use old solver code?
- [ ] Test test test.
---

#### Web Integration (requires solver integration)
- [ ] Get webAPI working with conversion
	- [ ] refactor webapi internals.
	- [ ] add http / authentication etc.
	- [ ] hook webapi to conversion.
	- [ ] Test with simple run.
	- [ ] Add some example notebooks using supported sources / monitors.
	- [ ] Include Nice web api plotting / progressbars using `rich`.
- [ ] Get webAPI working without conversion.
	- [ ] Use native `Simulation.export()` or `Simulation.json()` for ``upload()`.
	- [ ] Use native `SimulationData.load()` for `load()`.
- [ ] Test test test and then test more.
---

#### Finishing Integration
- [x] Separate FieldData into different xr.DataArrays with different xs, ys, zs at positions in yee lattice
- [ ] ModeSolver uses Permittivity Monitor for discretization / eps_cross.
- [ ] Support for Nonuniform Mesh
- [ ] Handling symmetries properly
- [ ] Handling client-side discretization properly.
---

#### "Tidying" Up
- [x] clean up data.plot arguments. <- using xarray.
- [x] Replace tqdm with rich
- [x] Add tests for near2far.
- [x] Allow numpy for non-data inputs.
- [x] Refactor material library.
- [x] **Refactor Monitor / MonitorData**
	- [x] Combine fields components (Ex, Ey, Ez, ...)
	- [x] Put permittivity monitor into freq-domain field data.
- [x] Get rid of pydantic? <- Not doing it
- [ ] Simplify MonitorData
    - [ ] remove MonitorData.monitor  and MonitorData.monitor_name attributes.
    - [ ] remove MonitorData.load() and MonitorData.export()
    - [ ] assert all SimulationData.monitor_data.keys() are in SimulationData.simulation.monitor.keys()
    - [ ] provide optional args to SimulationData.export() to only export some MonitorData by key
    - [ ] Move any interfaces in MonitorData to SimulationData (geometry, plotting, etc).
    - [ ] Remove unneeded data or monitor maps.
    - [ ] Make monitordata.load_from_data aware of lists
- [x] Use shapely for geometry ops / plotting?  `Geometry.geo(x=0)` -> shapely representation.
	- [x] Fix all tests.
	- [x] Integrate shapely plotting / kwargs.
	- [x] Catch edge case where new shape intersects with **two** shapes of the same medium.
	- [x] Integrate overlapping shape plotting using intersection.
	- [x] Clean up arguments to polyslab.
	- [x] Clean up pylint and org issues.
	- [x] use xmin, ymin, ... = `Geometry.intersections(axis).bounds` to ``get_bounds()``
	- [x] Integrate shapely ops into bounds checking / intersection checking.
- [x] Fix MyPy warnings. <- too many stupid ones? ..
- [ ] Add Logging.
- [ ] Migrate notebooks into static tests.
- [ ] Add PEC PMC.
- [ ] Handle Inf as it's own keyword that depends on simulation bounds.
- [ ] Make Uniform Samplers separate classes? (dont evaluate / store freqs, times)?
- [ ] setup.cfg for installing packages with more granularity (base, docs, tests)
---

#### Momchil's To Do Notes
- [ ] Front end
	- [ ] More discussion in Simulation docs.
	- [ ] Courant 0.99 stable? Adjust range of acceptable values.
	- [ ] Support nonuniform coordinates.
	- [ ] Support Medium names eventually.
	- [ ] Support diagonal anisotropy.
	- [ ] GDS slab import (+shapely import).
	- [ ] If using "polarization" kwarg, specify 'J' or 'E' in source?.
- [ ] Conversion
	- [ ] Find proper way to link tidy3d and tidy3d core. (core imports local copy of Simulation always updated and on main branch by default?).
	- [ ] Convert dispersive models to PoleResidue.
	- [ ] Support plane wave, gaussian beam, modesource in JSON conversion.
	- [ ] Support mode monitor, permittivity monitor in JSON conversion.	
- [ ] Data
	- [ ] Provide optional args to only export the `MonitorData` of selected `Monitor`s.
	- [ ] Wrap string conversion in MonitorLoad in helper function.
	- [ ] Assert SimulationData's monitor names exist in original Simulation.
	- [ ] Solver field data evaluated at yee cell locations, sufficient to interpolate values anywhere in monitor volume.
	- [ ] Eliminate need for storing `monitor_name` in `MonitorData`.
---

### Stage 3: Refining

A good template:
https://github.com/crusaderky/python_project_template

#### Documentation

- [ ] Finalize writing of internal documentation (1 week)
	- [ ] Add and edit docstrings to all.
	- [ ] Add example code snippets to everything.
- [ ] Set up documentation (1 week)
	- [x] Make pydantic autosummaries more pretty.
	- [ ] Move Docs into repo.
	- [ ] write tutorial notebooks for
		- [x] Visualizing simulation.
		- [x] Loading data.
		- [x] Visualizing data.
		- [ ] Batch simulation.
	- [ ] Include material library.
	- [ ] make docs pretty,
	- [ ] Move tidy3d notebooks into repo, make them work with new code.
	- [ ] set up and test binder links.
	- [ ] Separate notebooks / docs into its own repo?
---

#### Testing

- [ ] Add extensive amount of tests (2 weeks)
	- [x] Test simulation.json validation / error catching.
	- [x] Test plugins.
	- [x] Test notebooks.
	- [ ] Test submitting jobs if possible <- do this on tidy3dcore tests?
	- [ ] Import and integrate tests from existing solvers.
---

#### Github Integration
- [ ] Automate everything using GitHub extensions (2 weeks)
	- [ ] CI / tests.
	- [ ] Version / releases.
	- [ ] changelog
	- [ ] Contributing guide / developer documentation.
	- [ ] What else?
---

#### Refine

- [ ] Add finishing touches (2 days)
	- [ ] Logo (1 day, outsource?)
	- [ ] Github issue templates (.5 days)
	- [ ] Releases, PyPI setup (.5 days)
	- [ ] Wipe commit history
	- [ ] decide how to manage branches
---

#### Final
- [ ] Finding bugs and fixing things, testing (2 weeks)
- [ ] White paper on arxiv?
- [ ] Release publicly! :partying_face:
---

### Extensions
- [ ] Geometry
	- [ ] Erosion / Dilation of polyslabs
	- [ ] Vectorize / automatic `inside` based on `intersections`
	- [ ] Rotations of shapes (x=None, y=None, z=None -> ax+by+cz=0)
	- [ ] Angled sidewalls.
	- [ ] Angled sources, monitors.
- [ ] Visualization
	- [ ] Simple gui?  https://github.com/hoffstadt/DearPyGui
	- [ ] 3 panel lumerical style plot `Simulation.visualize()`
	- [ ] 3D structure plotting (matplotlib?)
	- [ ] Output Simulation as blender file
	- [ ] Interactive visualization.
	- [ ] html export plots.
- [ ] Plugins
	- [ ] S matrix plugin
	- [ ] Optimizer plugin
	- [ ] Simple yaml editor? flask app
