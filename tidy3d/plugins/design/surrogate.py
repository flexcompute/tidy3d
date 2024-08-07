import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import Trial
from tqdm import tqdm

import tidy3d as td
from tidy3d.plugins.design.surrogate_model_funcs import (
    print_label_stats,
    train_regression_network,
    validate_regression,
)
from tidy3d.plugins.design.surrogate_models import Basic1DCNN as NN

# Local
from tidy3d.plugins.design.surrogate_preprocess_funcs import (
    load_sim_data,
    pytorch_load,
    save_scalers,
    scale_feature,
    scale_label,
    split_data,
)

# Set random seeds
SEED = 3
np.random.seed(SEED)
torch.cuda.empty_cache()
torch.manual_seed(SEED)

os.chdir("/home/matt/Documents/Flexcompute/bragg")
DATA_DIR = "/home/matt/Documents/Flexcompute/bragg/simple_fdtd"
OUTPUT_DIR = "/home/matt/Documents/Flexcompute/bragg"
PICKLE_DIR = "/home/matt/Documents/Flexcompute/bragg/pickle_data"

# Testing
label_stats = False
random_labels = False

# Model Constants
RESOLUTION = 65
BATCH_SIZE = 16
TEST_PERCENT = 0.05
VALID_PERCENT = 0.05
INIT_NEURONS = 1024
DROPOUT = 0.2
LEARNING_RATE = 5e-4
EPOCHS = 1000
PLOT_OUTPUT = False
TRIAL = 100

# Config for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


def objective(trial: Trial):
    # Setup NN
    cl1 = trial.suggest_int("cl1", 4, 16, step=4)
    cl2 = trial.suggest_int("cl2", 8, 32, step=4)
    c_kernal = trial.suggest_categorical("c_kernal", [3, 5, 7, 9])
    l1 = trial.suggest_int("l1", 64, 2048, step=64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    net = NN(RESOLUTION - 1, cl1, cl2, c_kernal, l1, dropout).to(device)
    trained_net = NN(RESOLUTION - 1, cl1, cl2, c_kernal, l1, dropout).to(device)
    optimiser = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    # Run training
    with tqdm(total=EPOCHS) as pbar:
        loss = train_regression_network(
            net,
            optimiser,
            train_loaded,
            test_loaded,
            loss_function,
            device,
            OUTPUT_DIR,
            epochs=EPOCHS,
            pbar=pbar,
            showPlot=PLOT_OUTPUT,
        )
        print(f" Best Test Loss: {loss}")

    trained_net.load_state_dict(torch.load(OUTPUT_DIR + "/model.pt"))

    test_rmse, test_mae, test_predictions = validate_regression(
        trained_net, test_loaded, scaled_test_labels, label_scaler, device
    )

    return test_rmse


def get_data(sim_data: td.SimulationData) -> dict[str, float]:
    flux = sim_data["flux"].flux
    reflectance = 1 - flux
    reflectance_max = float(np.max(reflectance).values)
    # TODO: compute the bandwidth Delta lambda
    return reflectance_max


if __name__ == "__main__":
    # Create data
    hdf5_files = [fileName for fileName in os.listdir(DATA_DIR) if ".hdf5" in fileName]
    modelName = f"bragg_{len(hdf5_files)}_{RESOLUTION}.pkl"

    try:
        os.chdir(PICKLE_DIR)
        with open(modelName, "rb") as d:
            raw_features, raw_labels, delta_n = pickle.load(d)
        print("Loaded data from pickle")
    except FileNotFoundError:
        print("Calculating data from scratch")
        os.chdir(DATA_DIR)
        raw_features, raw_labels, delta_n = load_sim_data(RESOLUTION, get_data, hdf5_files)

        os.chdir(PICKLE_DIR)
        with open(modelName, "wb") as file:
            pickle.dump((raw_features, raw_labels, delta_n), file)

    # Change shape for convolution
    raw_features = np.expand_dims(raw_features, axis=1)

    # Randomise labels for sanity check
    if random_labels:
        raw_labels = np.random.uniform(raw_labels.min(), raw_labels.max(), size=raw_labels.shape)
        # raw_labels = np.ones(shape=raw_labels.shape)

    # Shuffle and split data
    shuffle_index = np.random.permutation(len(raw_features))
    shuffle_features = raw_features[shuffle_index]
    shuffle_labels = raw_labels[shuffle_index]
    shuffle_delta_n = delta_n[shuffle_index]

    train_features, test_features, valid_features = split_data(
        shuffle_features, TEST_PERCENT, VALID_PERCENT
    )
    train_labels, test_labels, valid_labels = split_data(
        shuffle_labels, TEST_PERCENT, VALID_PERCENT
    )
    train_delta_n, test_delta_n, valid_delta_n = split_data(
        shuffle_delta_n, TEST_PERCENT, VALID_PERCENT
    )

    # Scale labels
    scaled_train_labels, label_scaler = scale_label(train_labels)
    scaled_test_labels = scale_label(test_labels, label_scaler)
    scaled_valid_labels = scale_label(valid_labels, label_scaler)

    # Scale features
    scaled_train_features, feature_scaler = scale_feature(train_features)
    scaled_test_features = scale_feature(test_features, feature_scaler)
    scaled_valid_features = scale_feature(valid_features, feature_scaler)

    # Load data into PyTorch DataLoaders
    train_loaded = pytorch_load(scaled_train_features, scaled_train_labels, BATCH_SIZE)
    test_loaded = pytorch_load(scaled_test_features, scaled_test_labels, BATCH_SIZE)
    valid_loaded = pytorch_load(scaled_valid_features, scaled_valid_labels, BATCH_SIZE)

    os.chdir(OUTPUT_DIR)
    save_scalers(label_scaler, feature_scaler)

    # Optuna
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=TRIAL, timeout=600)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    cl1 = 8
    cl2 = 32
    c_kernal = 9
    l1 = 1856
    dropout = 0.15
    learning_rate = 0.00055
    weight_decay = 1.34e-05
    net = NN(RESOLUTION - 1, cl1, cl2, c_kernal, l1, dropout).to(device)
    trained_net = NN(RESOLUTION - 1, cl1, cl2, c_kernal, l1, dropout).to(device)
    optimiser = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    # Run training
    with tqdm(total=EPOCHS) as pbar:
        loss = train_regression_network(
            net,
            optimiser,
            train_loaded,
            test_loaded,
            loss_function,
            device,
            OUTPUT_DIR,
            epochs=EPOCHS,
            pbar=pbar,
            showPlot=PLOT_OUTPUT,
        )
        print(f" Best Test Loss: {loss}")

    trained_net.load_state_dict(torch.load(OUTPUT_DIR + "/model.pt"))

    print("\nTest Data")
    test_rmse, test_mae, test_predictions = validate_regression(
        trained_net, test_loaded, scaled_test_labels, label_scaler, device
    )

    if random_labels:
        print("SANITY CHECK: Labels have totally random values")

    if label_stats:
        print("\nTrain Labels")
        print_label_stats(train_labels)

    if label_stats:
        print("\nTest Labels")
        print_label_stats(test_labels)
        print("\nTest Predictions")
        print_label_stats(test_predictions)

    print("\nValidation Data")
    valid_rmse, valid_mae, valid_predictions = validate_regression(
        trained_net, valid_loaded, scaled_valid_labels, label_scaler, device
    )

    if label_stats:
        print("\nValid Labels")
        print_label_stats(valid_labels)
        print("\nValid Predictions")
        print_label_stats(valid_predictions)

    _, ax = plt.subplots()

    def ref_max_model(delta_n):
        return np.tanh(10 * delta_n / 2.0) ** 2

    ax.scatter(test_delta_n, test_predictions, label="predicted values", c="tab:orange")

    dns = np.linspace(0, 0.5, 101)
    rmax = ref_max_model(dns)
    ax.plot(dns, rmax, label="analytical model")
    plt.ylabel("reflectance")
    plt.xlabel("delta n")
    plt.legend()
    plt.show()
