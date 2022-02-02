All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ``Selmeier.from_dispersion()`` method to quickly make a single-pole fit for lossless weakly dispersive materials.
- Stable dispersive material fits via webservice.
- Validates simulation based on discretized size.

### Changed
- Refined and updated documentation.
- New color schemes for plots.

## [0.2.0] - 2021-1-29

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

[Unreleased]: https://github.com/flexcompute/Tidy3D-client-revamp/compare/v0.2.0...develop
[0.2.0]: https://github.com/flexcompute/Tidy3D-client-revamp/compare/0.1.1...v0.2.0
[0.1.1]: https://github.com/flexcompute/Tidy3D-client-revamp/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/flexcompute/Tidy3D-client-revamp/releases/tag/0.1.0
