# Tidy3D Client Revamp

## Improvements

### JSON file creation

### Requirements
* Error checking / Type Checking / Field Checking
* Reduce “conversion” code.
* More streamlined from td objects to schema.

### Constraints / Keep in Mind
* object attributes dont always correspond to schema keys
* changes in schema can affect C++ but also web UI, so be careful
* numpy arrays convert to list or float
* Converting various things (GDS → Polyslabs)
* Unit conversion
* Backwards compatibility with default values

### Implement
* “master” schema with
	* Type annotation
	* Conversion as needed
	* Error checking

### Python API
* Simplify the objects to be data classes or similar so they work better with JSON.
* `simulation.data(monitor).E`  instead of `[‘E’]`
* grab monitors and sources by name `sim.monitors[‘name’]`
* Refactor
	* Geometry 
	* Mesh
* 

### Open Source Stuff
* Automatic code formatting
* Automatic testing / CI (add more tests)
* Type Annotation
* Tags on GitHub, issue tracking.

### Features

#### Visualization
* Viz uses geometry alone.
* With the simulation can get mesh and discretized with DL showing.

#### Circuit Simulation
* port object
* S parameters
* interface with klayout, gdsfactory,

#### Adjoint wrapper
Wrap tidy3d objects using JAX.
Write primitives for derivative of port / monitor data w.r.t. structure, source info.
