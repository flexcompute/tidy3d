import os
import pickle

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from tidy3d.plugins.design.surrogate_model_funcs import (
    print_label_stats,
    train_regression_network,
    validate_regression,
)
from tidy3d.plugins.design.surrogate_models import BasicReducingANN as NN

# Local
from tidy3d.plugins.design.surrogate_preprocess_funcs import (
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

os.chdir("/home/matt/Documents/Flexcompute/tidy3d/docs/notebooks")
OUTPUT_DIR = "/home/matt/Documents/Flexcompute/tidy3d/tidy3d/plugins/design"

# Testing
label_stats = True
random_labels = False

# Model Constants
BATCH_SIZE = 16
TEST_PERCENT = 0.20
VALID_PERCENT = 0.20
INIT_NEURONS = 1024
DROPOUT = 0.2
LEARNING_RATE = 5e-4
EPOCHS = 1000
PLOT_OUTPUT = True

# Config for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

with open("ysplit.pkl", "rb") as input:
    raw_df = pickle.load(input)

with open("ysplit_2.pkl", "rb") as input2:
    raw_df2 = pickle.load(input2)

raw_df = pd.concat([raw_df, raw_df2])

raw_labels = np.array(raw_df.pop("output"))
raw_features = np.array(raw_df)

# Randomise labels for sanity check
if random_labels:
    raw_labels = np.random.uniform(raw_labels.min(), raw_labels.max(), size=raw_labels.shape)
    # raw_labels = np.ones(shape=raw_labels.shape)

# Shuffle and split data
shuffle_index = np.random.permutation(len(raw_features))
shuffle_features = raw_features[shuffle_index]
shuffle_labels = raw_labels[shuffle_index]

train_features, test_features, valid_features = split_data(
    shuffle_features, TEST_PERCENT, VALID_PERCENT
)
train_labels, test_labels, valid_labels = split_data(shuffle_labels, TEST_PERCENT, VALID_PERCENT)

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

os.chdir("/home/matt/Documents/Flexcompute/tidy3d/tidy3d/plugins/design")
save_scalers(label_scaler, feature_scaler)

# Setup NN
net = NN(train_features.shape[1], INIT_NEURONS, DROPOUT).to(device)
trained_net = NN(train_features.shape[1], INIT_NEURONS, DROPOUT).to(device)
optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
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

if random_labels:
    print("SANITY CHECK: Labels have totally random values")

if label_stats:
    print("\nTrain Labels")
    print_label_stats(train_labels)

print("\nTest Data")
test_rmse, test_mae, test_predictions = validate_regression(
    trained_net, test_loaded, scaled_test_labels, label_scaler, device
)

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
