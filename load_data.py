import numpy as np
import xarray as xr
import h5py

def field_to_DataArray(hdf5_dataset) -> xr.DataArray:
    shape = hdf5_dataset.shape

def flux_to_DataArray(hdf5_dataset) -> xr.DataArray:
    shape = hdf5_dataset.shape

def load_field_array(hdf5_dataset, coords):
    data = np.array(hdf5_dataset)
    axis_order = ["component", "xmesh", "ymesh", "zmesh", "freqs"]
    axis_coords = {k:coords[k] for k in axis_order}
    return xr.DataArray(data, coords=axis_coords)

def load_flux_array(hdf5_dataset, coords):
    data = np.array(hdf5_dataset)
    axis_order = ["freqs"]
    axis_coords = {k:coords[k] for k in axis_order}
    return xr.DataArray(data, coords=axis_coords)

def load_monitor(hdf5_group) -> xr.Dataset:

    keys = hdf5_group.keys()
    data_keys = [k for k in keys if k in ("E", "H", "flux")]
    coord_keys = [k for k in keys if k in ("xmesh", "ymesh", "zmesh", "freqs")]

    coords = {k: np.array(hdf5_group[k]) for k in coord_keys}

    coords["component"] = ["x", "y", "z"]

    dataset_dict = {}

    for key in data_keys:
        if key in ["H", "E"]:
            data_array = load_field_array(hdf5_group[key], coords)
        elif key in ["flux"]:
            data_array = load_flux_array(hdf5_group[key], coords)
        else:
            raise ValueError(f"key '{key}' not recognized")
        dataset_dict[key] = data_array
        return xr.Dataset(dataset_dict)

def load_hdf5(dfile: str = 'out/monitor_data.hdf5') -> dict:
    mfile = h5py.File(dfile, "r")

    monitor_data_dict = {}

    for key, mon_data in mfile.items():
        if key == "diverged":
            if mon_data[0] == 1:
                print(mfile["diverged_msg"][0])
                raise ValueError
        else:
            mon_dataset = load_monitor(mon_data)
            monitor_data_dict[key] = mon_dataset

    mfile.close()
    return monitor_data_dict

if __name__ == '__main__':
    d = load_hdf5('out/monitor_data.hdf5')