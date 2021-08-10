# Tidy3D Client Revamp

## Improvements

### JSON file creation

#### Requirements
- [x] Error checking / Type Checking / Field Checking
- [x] Reduce “conversion” code.
- [x] More streamlined from td objects to schema.

#### Constraints / Keep in Mind
- [x] object attributes dont always correspond to schema keys
- [x] changes in schema can affect C++ but also web UI, so be careful
- [x] numpy arrays convert to list or float
- [ ] Converting various things (GDS → Polyslabs)
- [ ] Unit conversion
- [ ] Backwards compatibility with default values

#### Implement
- [x] define “master” schema with
	- [x] Type annotation
	- [x] Conversion as needed
	- [x] Error checking
- [x] make compatible with existing tidy3d objects
- [ ] test all edge cases

### Python API
- [x] Simplify the objects to be data classes or similar so they work better with JSON.
- [x] `simulation.data(monitor).E`  instead of `[‘E’]`
- [x] grab monitors and sources by name `sim.monitors[‘name’]`
- [x] Refactor
	- [x] Geometry 
	- [x] Mesh

### Open Source Features
- [ ] Automatic code formatting
- [ ] Automatic testing / CI (add more tests)
- [x] Type annotation
- [ ] handling example notebooks and documentation
- [ ] Tags on GitHub, issue tracking, other nice features.

### Features

#### Visualization
- [x] Viz uses geometry alone.
- [ ] With the simulation can get mesh and discretized with DL showing.

#### Circuit Simulation
- [ ] port object
- [ ] S parameters
- [ ] interface with klayout, gdsfactory,

#### Adjoint wrapper
- [ ] Wrap tidy3d objects using JAX.
- [ ] Write primitives for derivative of port / monitor data w.r.t. structure, source info.
