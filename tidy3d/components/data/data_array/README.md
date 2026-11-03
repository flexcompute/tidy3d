# Data arrays

Data arrays are `xarray.DataArray` subclasses. Modules named after coordinates
(`spectral`, `spatial`, `unstructured`) hold arrays defined by their coordinates
and shared across products. `monitor` holds electromagnetic monitor outputs; the
modules named after products (`eme`, `tcad`, `microwave`) hold arrays that one
product produces or consumes.

- `base.py`: the public `DataArray` facade. Array behavior, dimensions and field
  parsing are inherited from the matching core array types. Public facades keep
  their import paths and class names, public errors, plotting and HDF5 readers. The shared implementation owns
  HDF5 writing and autograd-aware operations.
- `spectral.py`: frequency, time, mode, and terminal arrays.
- `abstract_spatial.py`: `AbstractSpatialDataArray`, the base class for arrays
  on structured spatial grids, with its bounds-aware selection, reflection, and
  downsampling methods. `_spatial_sampling.py` owns `MAX_SPATIAL_SAMPLES` and
  the grid builders used by `downsample`.
- `spatial.py`: fields on structured Cartesian or cylindrical grids, plus
  spherical angle sets. `SphericalAngleDataArray` is a plain `DataArray`: its
  coordinates are angle pairs, not grid axes.
- `unstructured.py`: point, cell, triangle-mesh, and index-keyed arrays. It
  keeps every member of `IndexedDataArrayTypes`.
- `monitor.py`: electromagnetic monitor arrays (flux, structure loss, mode,
  field projection, dipole emission, diffraction).
- `eme.py`: eigenmode-expansion arrays.
- `tcad.py`: TCAD results, plus the heat and charge lookup tables used by
  parameter perturbations.
- `microwave.py`: voltage, current, and impedance arrays and the integral result
  helpers.
- `registry.py`: `DATA_ARRAY_TYPES`, added to the `DATA_ARRAY_MAP` in
  `components/data/registry.py` that resolves serialized arrays by class name.
- `_constants.py`, `_plot.py`, `_traced.py`: shared constants, plot accessors,
  and the traced dataset wrapper.

The package initializer is the compatibility facade for the former monolithic
`data_array` module. It re-exports the names that module defined, public and
private. Use the focused modules for new imports.

Public arrays are subclasses of their corresponding `flexcompute.data.arrays`
classes. The shared implementation imports neither schema models nor the public
client. The schema package-level array exports alias these shared types.

Core parsing reuses exact public facades and checks dimensions on every call.
It writes canonical data and coordinate attributes onto the returned array,
including a reused array owned by the caller. Writable arrays must be checked
when used again; there is no persistent validation cache.
`DataArray.__init_subclass__` derives `_core_array_type` from each facade's direct,
same-name shared base. User subclasses of facades and arrays supplied for a
different field role convert to the canonical core type. Native core subclasses
are reused.

Reuse shares the wrapper, coordinates, attributes, name and data within the
Python process. Callers needing independent mutable state must copy it explicitly.
Loading native core results into public fields constructs the public facade
while sharing the numerical buffer. JSON uses the array's class name; both public
and core HDF5 writers use shuffle for payloads of at least 64 KiB. HDF5 is the
portable interchange format; a Python pickle containing public arrays requires
the public package when unpickling.
