import numpy as np
import torch.nn as nn
import torch.optim as optim

import tidy3d as td
from tidy3d.plugins.design.surrogate_models import Basic1DCNN as NN
from tidy3d.plugins.design.surrogate_object import AI_Model

# Constants
output_dir = "/home/matt/Documents/Flexcompute/y_split"
data_dir = "/home/matt/Documents/Flexcompute/y_split/data"

test_percentage = 0.10
valid_percentage = 0.10
batch_size = 64
epochs = 50

trial_count = 10


def y_split_data(hdf5_files):
    """Load y_split data getting features and labels"""

    def create_features(sim):
        junction_vertices = sim.structures[1].geometry.vertices
        top_junction_boundary = junction_vertices[0:100]
        top_junction_vertices = top_junction_boundary[:, 1]

        return top_junction_vertices

    def fn_post(sim_data):
        power_reflected = np.squeeze(
            np.abs(sim_data["mode_11"].amps.sel(direction="-", mode_index=0)) ** 2
        )
        power_transmitted = np.squeeze(
            np.abs(sim_data["mode_12"].amps.sel(direction="+", mode_index=0)) ** 2
        )

        loss_fn = (
            1
            / 3
            * len(power_reflected)
            * np.sum(power_reflected**2 + 2 * (power_transmitted - 0.5) ** 2)
        )
        output = -float(loss_fn.values)  # Negative value as this is a minimizing loss function

        return output

    raw_features = []
    raw_labels = []
    for sim_file in hdf5_files:
        sim_data = td.SimulationData.from_file(sim_file)
        raw_features.append(create_features(sim_data.simulation))
        raw_labels.append(fn_post(sim_data))

    raw_features = np.array(raw_features)
    raw_labels = np.array(raw_labels)

    # Change shape for convolution
    raw_features = np.expand_dims(raw_features, axis=1)

    return raw_features, raw_labels


model = AI_Model(output_dir, data_dir, rng_seed=2)
model.load_data(
    y_split_data,
    test_percentage,
    valid_percentage,
    batch_size,
    pickle_name=f"y_split_{len(model.hdf5_files)}.pkl",
)

cl1 = 16
cl2 = 8
c_kernal = 7
l1 = 1344
dropout = 0.45
learning_rate = 0.000225
weight_decay = 5.99e-04
net = NN(100, cl1, cl2, c_kernal, l1, dropout)
optimiser = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = nn.MSELoss()
# trained_net = model.train_model("Model1", net, optimiser, loss_function, epochs, plot_output=True)
# model.validate_model(trained_net)

network_dict = {
    "network": NN,
    "kwargs": {"resolution": 100},
    "optimize_kwargs": {
        "cl1": {"name": "cl1", "low": 4, "high": 16, "step": 4},
        "cl2": {"name": "cl2", "low": 8, "high": 32, "step": 4},
        "c_kernal": {"name": "c_kernal", "choices": [3, 5, 7, 9]},
        "l1": {"name": "l1", "low": 64, "high": 2048, "step": 64},
        "dropout": {"name": "dropout", "low": 0.0, "high": 0.5, "step": 0.05},
    },
}

optimizer_dict = {
    "optimizer": optim.Adam,
    "kwargs": {},
    "optimize_kwargs": {
        "lr": {"name": "lr", "low": 1e-5, "high": 1e-1, "log": True},
        "weight_decay": {"name": "weight_decay", "low": 1e-5, "high": 1e-3, "log": True},
    },
}


best_network, best_optimizer = model.optimize_network(
    trial_count, "minimize", network_dict, optimizer_dict, loss_function, epochs
)

# trained_network = model.train_model("BestModel", best_network, best_optimizer, loss_function, 1000, True)
# model.validate_model(trained_network, "test")
# model.validate_model(trained_network, "valid")
