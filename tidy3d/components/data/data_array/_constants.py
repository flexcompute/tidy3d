"""Dimension attributes, units, and HDF5 names shared by data arrays."""

from __future__ import annotations

from tidy3d.constants import (
    COULOMB,
    HERTZ,
    MICROMETER,
    RADIAN,
    SECOND,
    STERADIAN,
    VOLT,
    WATT,
)

# maps the dimension names to their attributes
DIPOLE_EMISSION_INTENSITY_UNITS = f"{WATT}/({STERADIAN} * ({COULOMB}*{MICROMETER})^2)"


DIM_ATTRS = {
    "x": {"units": MICROMETER, "long_name": "x position"},
    "y": {"units": MICROMETER, "long_name": "y position"},
    "z": {"units": MICROMETER, "long_name": "z position"},
    "f": {"units": HERTZ, "long_name": "frequency"},
    "t": {"units": SECOND, "long_name": "time"},
    "v": {"units": VOLT, "long_name": "voltage"},
    "direction": {"long_name": "propagation direction"},
    "mode_index": {"long_name": "mode index"},
    "terminal_label": {"long_name": "terminal label"},
    "terminal_label_out": {"long_name": "output terminal label"},
    "terminal_label_in": {"long_name": "input terminal label"},
    "eme_port_index": {"long_name": "EME port index"},
    "eme_cell_index": {"long_name": "EME cell index"},
    "eme_interface_index": {"long_name": "EME interface index"},
    "mode_index_in": {"long_name": "mode index in"},
    "mode_index_out": {"long_name": "mode index out"},
    "trace_index_in": {"long_name": "trace index in"},
    "trace_index_out": {"long_name": "trace index out"},
    "sweep_index": {"long_name": "sweep index"},
    "theta": {"units": RADIAN, "long_name": "elevation angle"},
    "phi": {"units": RADIAN, "long_name": "azimuth angle"},
    "ux": {"long_name": "normalized kx"},
    "uy": {"long_name": "normalized ky"},
    "orders_x": {"long_name": "diffraction order"},
    "orders_y": {"long_name": "diffraction order"},
    "face_index": {"long_name": "face index"},
    "vertex_index": {"long_name": "vertex index"},
    "axis": {"long_name": "axis"},
    "dipole_axis": {"long_name": "dipole orientation axis"},
    "polarization": {"long_name": "emission polarization"},
    "angle": {"long_name": "emission angle index"},
    "spherical_coordinate": {"long_name": "spherical coordinate"},
}


# name of the DataArray.values in the hdf5 file (xarray's default name too)
DATA_ARRAY_VALUE_NAME = "__xarray_dataarray_variable__"


HDF5_SHUFFLE_MIN_BYTES = 64 * 1024
