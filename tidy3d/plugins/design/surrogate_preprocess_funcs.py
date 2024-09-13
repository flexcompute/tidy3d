import os
import pickle

import numpy as np
import torch
import torch.utils
from sklearn.preprocessing import MinMaxScaler


def split_data(data_array, test_percent, valid_percent):
    # Calculate indexes
    array_len = len(data_array)
    train_finish_index = array_len - int(test_percent * array_len + valid_percent * array_len)
    test_start_index = train_finish_index
    test_finish_index = array_len - int(valid_percent * array_len)
    validation_start_index = test_finish_index

    # Divide up array
    train_arr = data_array[:train_finish_index]
    test_arr = data_array[test_start_index:test_finish_index]
    valid_arr = data_array[validation_start_index:]

    return train_arr, test_arr, valid_arr


def scale_label(y, pre_fit=None):
    if pre_fit is None:
        standardiser = MinMaxScaler()
        output = standardiser.fit_transform(y)

        return output, standardiser
    else:
        output = pre_fit.transform(y)
        return output


def scale_feature(x, individual_feature_scaling: bool, feature_scaler=None):
    original_shape = x.shape

    if len(original_shape) > 2:
        print(
            "Warning: scaling of features does not currently accept feature arrays with this many dimensions"
        )

    if individual_feature_scaling:
        scaler_dict = {}
        scaled_arrays = []
        for i in range(0, original_shape[1]):
            sub_array = x[:, i].reshape(-1, 1)
            if feature_scaler is None:
                scaler_dict[i] = MinMaxScaler()
                scaled_arrays.append(scaler_dict[i].fit_transform(sub_array))

            else:
                scaled_arrays.append(feature_scaler[i].transform(sub_array))

        scaled_arrays = np.concatenate(scaled_arrays, axis=1)

        if feature_scaler is None:
            return scaled_arrays, scaler_dict
        else:
            return scaled_arrays

    else:
        # Reduce to 1D array and scale
        flattened_x = x.reshape(-1, 1)

        if feature_scaler is None:
            scaler = MinMaxScaler()
            scaled_array = scaler.fit_transform(flattened_x)

        else:
            scaled_array = feature_scaler.transform(flattened_x)

        # Return to original shape
        scaled_array = scaled_array.reshape(original_shape)

        if feature_scaler is None:
            return scaled_array, scaler
        else:
            return scaled_array


def pytorch_load(x, y, batch_size, shuffle_data=False):
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            features = self.x[idx]
            target = self.y[idx]
            return features, target

    loaded = torch.utils.data.DataLoader(
        SampleDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle_data,
        pin_memory=True,
        num_workers=0,
    )

    return loaded


def save_scalers(model_dir, label_scaler, feat_scaler):
    """
    Save the scaler used for the labels and features
    """
    os.chdir(model_dir)

    with open("scalers.pkl", "wb") as out_file:
        pickle.dump([label_scaler, feat_scaler], out_file)


def load_scalers():
    """
    Load scalers for labels and features
    """

    with open("scalers.pkl", "rb") as in_file:
        label_scaler, feat_scaler = pickle.load(in_file)

    return label_scaler, feat_scaler


class Dummy_Scaler:
    def transform(self, arr):
        return arr

    def inverse_transform(self, arr):
        return arr
