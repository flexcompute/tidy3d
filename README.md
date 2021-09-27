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

#### Solver Integration

- [x] Add version to simulation.json
- [ ] **Make tidy3d_core functional with existing (or slightly modified) solver.**
	- [x] Load simulation.json into `tidy3d.Simulation`.
	- [x] IO SimulationData
	- [ ] Process Simulation to solver inputs. (momchil)
	- [ ] Write solver outputs to SimulationData. (momchil)
	- [ ] Separate FieldData into different xr.DataArrays with different xs, ys, zs at positions in yee lattice
- [ ] Test test test.
---

#### Web Integration (requires solver integration)
- [ ] refactor webapi <- can do without integration?
- [ ] add http / authentication etc.
- [ ] set up so that daemon recgonizes new json files.
- [ ] Test test test.
---

#### Plugins

- [ ] Flesh out all details for plugins (18.5 days total)
	- [x] Make dispersion fitting tool compatible
	- [ ] **Make mode solver compatible (2)**
	- [x] Batch processor <- done in webapi
---

#### Visualization
- [ ] write all visuaization routines
	- [x] MonitorData
	- [x] Basic sliding window for cross section.
	- [ ] **Structure cross section, sliding bar**
	- [ ] **3 panel lumerical style plot `Simulation.visualize()`**
	- [ ] Overlay structures onto fields.
	- [ ] SimulationData
---

#### IO
- [x] Supply other ways to create simulation files (2 days)
	- [x] yaml -> json conversion 
	- [x] command line interface to submit yaml or json.
---

#### Finishing Integration
- [ ] Mode Sources / Monitors
- [ ] Nonuniform Mesh

### Stage 3: Refining

A good template:
https://github.com/crusaderky/python_project_template

#### Documentation

- [ ] Finalize writing of internal documentation (1 week)
	- [ ] Add and edit docstrings to all.
- [ ] Set up documentation (1 week)
	- [ ] Make pydantic autosummaries more pretty.
	- [ ] Move Docs into repo.
	- [ ] Move tidy3d notebooks into repo, make them work with new code.
	- [ ] test binder links.
---

#### Testing

- [ ] Add extensive amount of tests (2 weeks)
	- [x] Test simulation.json validation / error catching.
	- [ ] Test plugins.
	- [ ] Test submitting jobs if possible.
	- [x] Test notebooks.
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
- [ ] Release publicly! :partying_face:

---
### After this'
- [ ] 3D structure plotting (matplotlib?)
- [ ] S matrix plugin
- [ ] Optimizer plugin
- [ ] Mode solver API using web
- [ ] More advanced plotting
- [ ] Angled sources
- [ ] simple yaml editor?  flask app
- [ ] Output Simulation as blender file
