import os
import pickle

import torch
from sklearn.preprocessing import MinMaxScaler


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
        standardiser = MinMaxScaler()
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


def save_scalers(model_dir, label_scaler, feat_scaler):
    """
    Save the scaler used for the labels and features
    """
    os.chdir(model_dir)

    with open("scalers.pkl", "wb") as outFile:
        pickle.dump([label_scaler, feat_scaler], outFile)


def load_scalers():
    """
    Load scalers for labels and features
    """

    with open("scalers.pkl", "rb") as inFile:
        labelScaler, featScaler = pickle.load(inFile)

    return labelScaler, featScaler
