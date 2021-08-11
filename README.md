# Tidy3D Client Revamp

## Roadmap

---

### Basics

#### Data Structure
- [x] Implement basic `Tidy3d` structure using `pydantic`
- [x] Implement basic validation of fields using `pydantic.validators`
- [x] Write basic emulation of `td.web` interface for submitting simulations and loading results

#### Schema
- [x] Automatically generate schema from `Simulation` model
- [x] Generate JSON output from `Simulation` instance
- [x] Custom validatation of json file against schema using `jsonschema`
- [x] Integrate schema into `td.web` emulation

#### Data
- [x] Load `Tidy3d` monitor data (.hdf5) into `xarray` datastructure for ease of accessing various axes
- [ ] Store return of `data.load()` with `Simulation`.
- [ ] Implement basic viz and postprocessing `Tidy3d` functions for output data
- [ ] Integrate data loading into `td.web` emulation

#### Sources
- [ ] Define and implement basic (non-modal) `Sources`, make consistent with current schema.
- [ ] Define `td.viz` function for vieweing source data.

#### Monitors
- [ ] Define and implement basic (non-modal) `Monitors`, make consistent with current schema.

#### Modes
- [ ] Integrate mode solver into `Tidy3d`
- [ ] Define and implement function of `ModeMonitor` and `ModeSource`
- [ ] Integrate Mode objects with `xarray`

#### WebAPI
- [ ] Refactor existing webAPI interface.
- [ ] Port webAPI interface into `Tidy3d` and hook up with existing monitors and data.
- [ ] Improve or rework batch processing interface.

#### Structures
- [ ] Implement basic Polyslab
- [ ] Test all structure bounds
- [ ] Determine best plotting package for geometric shapes
- [ ] Write viz functions / geometrical plotting

#### Viz
- [ ] Implement complete geometric plotting of all `GeometryObject` attributes of `Simulation`.
- [ ] Add mesh overlay with 3D plots.
- [ ] Explore interactive plotting inline in notebook.

---

### Integration

#### Integration
- [ ] Incorporate most args, kwargs into API.
- [ ] Ensure the json is readable by our server, make consistent.
- [ ] Add tons of tests.

#### Docs
- [ ] Port docs into package.
- [ ] Update docs to be consistent with changes.
- [ ] Set up and document workflow for updating and pushing docs.
- [ ] Logo.

#### Open Source Tools
- [ ] Finalize testing setup.
- [ ] Set up continuous integration.
- [ ] Set up code quality checking, code reformating.
- [ ] Make standard python package, `__version__`, `changelog`, bug reporting, various metrics.

#### Testing
- [ ] Test all aspects of package.

#### Backend
- [ ] Streamline backend handling of data.
- [ ] IO of xarray objects instead of hdf5 files.
- [ ] Use schema from tidy3d to do validation
- [ ] No code copying, just import.
- [ ] Implement system to co-develop with two git branches.

#### Release
- [ ] Export to PyPI.
- [ ] Make public on github.
- [ ] existing `tidy3d` -> `tidy3d core`, make consistent with this schema.

---

### Enhacement

#### Adjoint
- [ ] Figure out basic adjoint problem for volume and shifting boundaries
- [ ] Come up with plan for how to track gradients to arg data in `Simulation`
- [ ] Implement adjoint wrapper

#### YAML input
- [ ] Load simulation from YAML file and validate.
- [ ] Explore web based YAML editors and integrate into workflow.
- [ ] Host YAML editor on web UI with 'export simulation'

#### Hosting Tidy3d Examples
- [ ] Explore ways to host notebooks with Tidy3d already installed, rather than !pip installing on colab.
- [ ] Share notebooks with links.

#### Integrated GUI
- [ ] Implement basic GUI with filesystem, code or YAML editor, and interactive plotting.
