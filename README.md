# Tidy3D Client Revamp

<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-client-revamp/HEAD?filepath=notebooks) -->


## Developer

### Git Notes

https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
https://www.atlassian.com/git/tutorials/merging-vs-rebasing

### New feature

```
git checkout develop
git checkout -b feature
# develop and make some commits
git rebase -i develop
# squash, edit, commits
# submit PR on github.
# when ready, rebase and merge feature into develop
git checkout develop
git pull -or- git reset --hard origin/develop
```

### Release
```
git checkout main
git merge develop
git tag x.x.x
git push origin x.x.x
```


## Flow


### Client (Pre)

- Make `tidy3d.components` to define simulation.
	- Using builtin `tidy3d` imports (`td.PlaneWave`, `td.ModeMonitor`, etc.)
	- Using `tidy3d.plugins` to construct more specialized components (mode solver, dispersion fitter, etc.).
- Create a `td.Simulation` object containing all simulation parameters (pydantic will automatically validate all components).
- Upload `Simulation` to server using `tidy3d.web`.
	- Export `Simulation` to a .json file format with `Simulation.json()`
	- Save as .json file.
	- Upload to server using http request, authenticate, etc.
	- Use `plugins.batch_processor` to submit batches of simulations.
- Manage task with `tidy3d.web`
	- Run task explicitly (if draft).
	- Monitor progress.
	- Cancel / delete task.
- load results into `td.SimulationData` object, containing simulation and data for all of its monitors.
	- Look at log through `sim_data.log`.
	- Plot data.
	- Manipulate, interpolte, resample data.


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

### Stage 2: Basics

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

#### Solver Conversion

- [x] Add version to simulation.json
- [ ] **Make tidy3d_core functional with existing (or slightly modified) solver.**
	- [x] Load simulation.json into `tidy3d.Simulation`.
	- [x] IO SimulationData
- [x] Convert to old .json conversion to use old solver code?
---

#### Web Integration (requires solver integration)
- [x] Get webAPI working with conversion
	- [x] add http / authentication etc.
	- [x] hook webapi to conversion.
	- [x] Test with simple run.
	- [x] Fix issue with converting to complex.
	- [x] **Refactor some webapi internals.**
	- [x] **Add conversions for rest of objects.**
	- [x] **Containers (job batch).**
	- [x] **Better handling for runtime status using rich.**
	- [x] **Add example notebooks and make consistent.**
	- [x] **Comments / documentations**

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
- [x] Simplify MonitorData
    - [x] remove MonitorData.monitor  and MonitorData.monitor_name attributes.
    - [x] remove MonitorData.load() and MonitorData.export()
    - [x] Make monitordata.load_from_group aware of lists
- [x] Use shapely for geometry ops / plotting?  `Geometry.geo(x=0)` -> shapely representation.
	- [x] Fix all tests.
	- [x] Integrate shapely plotting / kwargs.
	- [x] Catch edge case where new shape intersects with **two** shapes of the same medium.
	- [x] Integrate overlapping shape plotting using intersection.
	- [x] Clean up arguments to polyslab.
	- [x] Clean up pylint and org issues.
	- [x] use xmin, ymin, ... = `Geometry.intersections(axis).bounds` to ``get_bounds()``
	- [x] Integrate shapely ops into bounds checking / intersection checking.
- [x] Fix MyPy warnings. <- too many stupid ones?

### Stage 3: Details (starting on 10/27)

#### Getting package to usable status (18 days)
- [x] Finishing main features (1 week)
 	- [x] `SimulationData` interface.  Any new methods to add? (plotting, etc?) (1 day)
	- [x] Near2far with new API (3 days)
	- [x] Mode Monitor consistent with new .epsilon() (1 day)
- [x] API changes (discuss first, implementation in 1 day)
	- [x] Freqs and times store start, end, stop / number instead of raw values.
	- [x] Change source polarization to E instead of J.
	- [x] named Meidums?
	- [x] Symmetry, PML, grid spec.  Less clunky interface? 
- [x] Covering features of existing code (1 day)
	- [x] support diagonal anisotropy (permittivity as 3-tuple)
	- [x] Conversion of dispersive materials into pole-residue.
	- [x] gaussian beam.
	- [x] option to display cell boundaries in plot.
	- [x] gds slab / gds importing.
	- [x] Add PEC medium
- [ ] Documentation (1 week)
	- [x] Add more discussion into Simulation docs.
	- [x] Write docstrings and examples for all callables.
	- [ ] How Do I?
	- [ ] Developer guide
	- [ ] Package structure guide / explanation.
	- [ ] Make all notebooks work with new version.
	- [ ] Material library.
- [ ] Improvement (2 days)
	- [x] Add more info / debug logging and more comprehensive error handling (file IO, etc).
	- [ ] Add more intelligent 'inf' handling.
	- [ ] setup.cfg for installing dependencies for different parts of the code (base, docs, tests)
	- [ ] web.monitor using running status for progress updates <- waiting on victor.

---

#### Integration (in parallel to above) (18 days?)
- [ ] Momchil will work on integration in tidy3d core while I do above.

---

#### Integration of integration (8 days)
After integration is complete, need to flesh out details and make them compatible with the main package.

- [ ] Make webAPI work without conversion (1 hour)
	- [ ] Use native `Simulation.export()` or `Simulation.json()` for `upload()`.
	- [ ] Use native `SimulationData.load()` for `load()`.
- [ ] Flesh out Mode solver details (discussion, then implemeent in 1 hour)
	- [ ] Change API?
- [ ] Flesh out Symmetry details (3 days?)
- [ ] Nonuniform mesh, make compatible with front end (few hours?)
- [ ] Add Permittivity monitor (2 days?)
- [ ] Organize tests into core and tidy3d tests (few hours?)

---

### Stage 4: Polishing for Release (21 days)

A good template:
https://github.com/crusaderky/python_project_template

#### Documentation

- [ ] Refine documentation (1 week)
	- [ ] Make pydantic autosummaries more pretty.
	- [x] Move Docs and notebooks into their own repos?
	- [ ] write as many tutorial notebooks as we can.
	- [ ] make docs pretty,
	- [ ] set up and test binder links.
---

#### Testing

- [ ] Add extensive amount of tests (1 week)
	- [x] Test simulation.json validation / error catching.
	- [x] Test plugins.
	- [x] Test notebooks.
	- [x] Test submitting jobs if possible <- do this on tidy3dcore tests?
	- [ ] Import and integrate tests from existing solvers.
---

#### Github Integration
- [ ] Automate everything using GitHub extensions (1 week)
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

#### Final (28 days)
- [ ] Finding bugs and fixing things, testing (2 weeks)
- [ ] White paper on arxiv? (2 weeks)
- [ ] Release publicly! :partying_face:
---

### Extensions
- [ ] Web
	- [ ] Nail down propper status messages.
	- [ ] Store server-side log metadata inside SimulationData (credits billed etc)
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
- [ ] Other
	- [ ] Courant 0.99 stable? Adjust range of acceptable values. <- weiliang.

