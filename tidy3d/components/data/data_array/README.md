# Data arrays

Data arrays are `xarray.DataArray` subclasses. Modules named after coordinates
(`spectral`, `spatial`, `unstructured`) hold arrays defined by their coordinates
and shared across products. `monitor` holds electromagnetic monitor outputs; the
modules named after products (`eme`, `tcad`, `microwave`) hold arrays that one
product produces or consumes.

- `base.py`: the `DataArray` base class. It binds its HDF5 I/O methods from
  `_io.py` and its autograd-traced interpolation and update methods from
  `_autograd.py`.
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
