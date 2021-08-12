import numpy as np
import xarray as xr
import h5py
import pydantic
from typing import Tuple, Dict


# if supplied keys are in keys(), convert to values()
KEY_CONVERSIONS = {
    'xmesh': 'x',
    'ymesh': 'y',
    'zmesh': 'z',
    'tmesh': 't',
    'freqs': 'freq',   
    'mode_amps': 'amps',
}


# coords that aren't defined
EXTRA_COORDS = {
    'component' : ['x', 'y', 'z'],
    'direction' : ['f', 'b']
}

# map data name to possible coordinate values at each index
DATA_COORD_MAP = {
    'E': (('component',), ('x',), ('y',), ('z',), ('freq', 't')),
    'H': (('component',), ('x',), ('y',), ('z',), ('freq', 't')),
    'S': (('component',), ('x',), ('y',), ('z',), ('freq', 't')),
    'flux': (('freq', 't'),),
    'amps': (('direction',), ('freq',), ('mode_orders'),),
    'modes': (('freq',), ('mode_orders',)),
    'indspan': None
}

# accounting of the keys and what groups they belong to
DEFAULT_COORD_KEYS = set(('x', 'y', 'z', 't', 'freq'))
EXTRA_COORD_KEYS = set(EXTRA_COORDS.keys())
COORD_KEYS = DEFAULT_COORD_KEYS.union(EXTRA_COORD_KEYS)
DATA_KEYS = set(('E', 'H', 'S', 'flux', 'amps', 'modes', 'indspan'))
ALL_KEYS = COORD_KEYS.union(DATA_KEYS)

def parse_group(hdf5_group) -> Tuple[Dict, Dict]:
    """ does conversions on HDF5 group and separates into coordinates and data """
    coords = {}
    data = {}
    for key, value in hdf5_group.items():

        # ignore empty values
        if len(value) <= 0:
            continue

        # rename keys and check
        if key in KEY_CONVERSIONS.keys():
            key = KEY_CONVERSIONS[key]
        assert (key in ALL_KEYS), f"key '{key}' not recognized"

        # separate into coords and data
        if key in COORD_KEYS:
            coords[key] = value
        elif key in DATA_KEYS:
            data[key] = np.array(value)

    return coords, data

def assemble_data_array(data_name: str, data_value: np.ndarray, coords: dict) -> xr.DataArray:
    """ creates data array using axis coordinate map """

    # get raw data and map
    coord_axis_option_map = DATA_COORD_MAP[data_name]

    # no map defined, just return without coordinates
    if coord_axis_option_map is None:
        return xr.DataArray(data_value)

    # construct coordinate dict
    axis_coords = {}
    for coord_axis_options in coord_axis_option_map:

        # loop through acceptable axis options
        for coord_option in coord_axis_options:

            # if supplied in coords, strip coords, convert to numpy array
            if coord_option in coords.keys():
                coord_vals = np.array(coords[coord_option])
                axis_coords[coord_option] = coord_vals

            # if given as default, just take that value
            elif coord_option in EXTRA_COORD_KEYS:
                axis_coords[coord_option] = EXTRA_COORDS[coord_option]

    # make suire axis coords have same number of entries as dimensions in
    if len(axis_coords) == data_value.ndim:
        return xr.DataArray(data_value, axis_coords)
    else:
        return xr.DataArray(data_value)



def load_monitor(hdf5_group) -> xr.Dataset:
    """ loads monitor's HDF5 group into xarray dataset """
    coords, data = parse_group(hdf5_group)
    dataset_dict = {}
    for data_name, data_value in data.items():
        data_value_raw = data
        data_array = assemble_data_array(data_name, data_value, coords)
        dataset_dict[data_name] = data_array
    return xr.Dataset(dataset_dict)

def load_hdf5(dfile: str = 'out/monitor_data.hdf5') -> Dict:
    """ loads HDF5 data file into dict of monitor datasets """
    mfile = h5py.File(dfile, 'r')

    monitor_data_dict = {}

    for key, mon_data in mfile.items():
        if key == 'diverged':
            if mon_data[0] == 1:
                print(mfile['diverged_msg'][0])
                raise ValueError('run diverged :(')
        else:
            mon_dataset = load_monitor(mon_data)
            monitor_data_dict[key] = mon_dataset

    mfile.close()
    return monitor_data_dict

if __name__ == '__main__':
    d = load_hdf5('out/monitor_data.hdf5')