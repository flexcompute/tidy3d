# Tidy3D Client Revamp

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-client-revamp/HEAD?filepath=notebooks)

## Roadmap (113.5 days = 16.2 weeks = 3.7 months ~ jan 1)

### Setup

- [x] Organize repo in basic structure we want moving forward (.5 days)
---
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
---
- [ ] Write boilerplate for all plugins / packages (1 week)
	- [ ] Dispersion fit
	- [ ] Mode Solver
	- [ ] Component Modeler
	- [ ] Adjoint Optimizer
	- [ ] Data Analyzer
---
- [ ] Write emulated webAPI to transfer files back and forth (1 day)
	- [ ] Make `task` its own pydantic dataclass.
	- [ ] Create basic mock data creation.
	- [ ] Create set of basic API calls.
	- [ ] Simple batch / job interface.
- [ ] Write tidy3d core and make webAPI functional for simplest cases (2-3 weeks, w/ Momchil’s help)
	- [ ] What are tidy3d core endpoints for C++?
	- [ ] Load simulation.json into `tidy3d.Simulation`.
	- [ ] Postprocess `tidy3d.Simulation` into solver files.
	- [ ] add http / authentication etc.
	- [ ] Test test test.
---
---
---
- [ ] Flesh out all details for plugins (18.5 days total)

	- [ ] Viz (1 week)
	- [ ] Dispersion fit (.5 days)
	- [ ] Mode solver (3 days)
	- [ ] Analyze (3 days)
		- [ ] Read output files in different formats.
	- [ ] S matrix (5 days)
	---
---
---
---
- [ ] Supply other ways to create simulation files (2 days)
	- [ ] yaml -> json conversion 
	- [ ] command line interface to submit yaml or json.
	- [ ] simple yaml editor?  flask app?
- [ ] Finalize writing of internal documentation (1 week)
	- [ ] Add and edit docstrings to all.
---
- [ ] Set up documentation (1 week)
	- [ ] Make pydantic autosummaries more pretty.
	- [x] Move Docs into repo.
	- [x] Move tidy3d notebooks into repo.
	- [ ] test binder links.
---
- [ ] Add extensive amount of tests (2 weeks)
	- [ ] Test simulation.json validation / error catching.
	- [ ] Test plugins.
	- [ ] Test submitting jobs if possible.
	- [ ] Test notebooks.
---
---
- [ ] Automate everything using GitHub extensions (2 weeks)
	- [ ] CI / tests.
	- [ ] Version / releases.
	- [ ] changelog
	- [ ] Contributing guide / developer documentation.
	- [ ] What else?
---
---
- [ ] Add finishing touches (2 days)
	- [ ] Logo (1 day, outsource?)
	- [ ] Github issue templates (.5 days)
	- [ ] Releases, PyPI setup (.5 days)
	- [ ] Wipe commit history
	- [ ] decide how to manage branches
---
- [ ] Finding bugs and fixing things, testing (2 weeks)
---
---
- [ ]  Release publicly! :partying_face:
