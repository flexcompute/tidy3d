import math
import pickle

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import tidy3d as td


def load_sim_data(RESOLUTION, fn_post, hdf5_files):
    raw_features = []
    raw_labels = []
    delta_n = []
    for sim_file in hdf5_files:
        sim_data = td.SimulationData.from_file(sim_file)
        raw_features.append(create_1d_array(sim_data.simulation, RESOLUTION))
        raw_labels.append(fn_post(sim_data))
        delta_n.append(calc_delta_n(sim_data.simulation))

    raw_features = np.array(raw_features)
    raw_labels = np.array(raw_labels)
    delta_n = np.array(delta_n)

    return raw_features, raw_labels, delta_n


def create_1d_array(sim, resolution):
    structure_info = [
        [struct.geometry.center[2], struct.geometry.size[2]] for struct in sim.structures
    ]
    start_z = structure_info[0][0] - structure_info[0][1] / 2
    end_z = structure_info[-1][0] + structure_info[-1][1] / 2

    structre_widths = np.array(
        [[struct[0] - struct[1] / 2, struct[0] + struct[1] / 2] for struct in structure_info]
    )
    search_points = np.linspace(start_z, end_z, resolution)

    permittivity_idx = []
    for point in search_points:
        for idx, widths in enumerate(structre_widths):
            if point >= widths[0] and point < widths[1]:
                permittivity_idx.append(idx)
                break

    permittivity_map = np.array(
        [sim.structures[idx].medium.permittivity for idx in permittivity_idx]
    )

    return permittivity_map


def calc_delta_n(sim):
    structures = sim.structures
    s1 = math.sqrt(structures[0].medium.permittivity)
    s2 = math.sqrt(structures[1].medium.permittivity)
    return abs(s1 - s2)


def split_data(dataArray, testPercent, validPercent):
    # Calculate indexes
    arrayLen = len(dataArray)
    trainFinishIndex = arrayLen - int(testPercent * arrayLen + validPercent * arrayLen)
    testStartIndex = trainFinishIndex
    testFinishIndex = arrayLen - int(validPercent * arrayLen)
    validationStartIndex = testFinishIndex

    # Divide up array
    trainArr = dataArray[:trainFinishIndex]
    testArr = dataArray[testStartIndex:testFinishIndex]
    validArr = dataArray[validationStartIndex:]

    return trainArr, testArr, validArr


def scale_label(y, pre_fit=None):
    data = y.reshape(-1, 1)

    if pre_fit is None:
        standardiser = QuantileTransformer()
        output = standardiser.fit_transform(data)

        return output, standardiser
    else:
        output = pre_fit.transform(data)
        return output


def scale_feature(x, pre_fit=None):
    original_shape = x.shape
    # Reduce to 1D array and scale
    flattened_x = x.reshape(-1, 1)

    if pre_fit is None:
        scaler = MinMaxScaler()
        scaled_flat_arr = scaler.fit_transform(flattened_x)
    else:
        scaled_flat_arr = pre_fit.transform(flattened_x)

    # Return to original shape
    scaledArr = scaled_flat_arr.reshape(original_shape)

    if pre_fit is None:
        return scaledArr, scaler
    else:
        return scaledArr


def pytorch_load(x, y, batch_size, shuffle_data=False):
    y_tensor = torch.Tensor(y)
    x_tensor = torch.Tensor(x)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loaded = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_data, pin_memory=True, num_workers=0
    )

    return loaded


def save_scalers(labelScaler, featScaler):
    """
    Save the scaler used for the labels and features
    """

    with open("scalers.pkl", "wb") as outFile:
        pickle.dump([labelScaler, featScaler], outFile)


def load_scalers():
    """
    Load scalers for labels and features
    """

    with open("scalers.pkl", "rb") as inFile:
        labelScaler, featScaler = pickle.load(inFile)

    return labelScaler, featScaler
