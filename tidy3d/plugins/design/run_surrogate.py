# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

import tidy3d as td
from tidy3d.plugins.design.surrogate_models import Basic1DCNN as NN
from tidy3d.plugins.design.surrogate_object import AI_Model

# Constants
output_dir = "/home/matt/Documents/Flexcompute/bragg"
data_dir = ["/home/matt/Documents/Flexcompute/bragg/cosine"]

test_percentage = 0.20
valid_percentage = 0.20
batch_size = 32
epochs = 100

trial_count = 40


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

        loss_fn = (1 / 3 * len(power_reflected)) * np.sum(
            power_reflected**2 + 2 * (power_transmitted - 0.5) ** 2
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


def bragg_grating_data(resolution, hdf5_files):
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

    def get_data(sim_data: td.SimulationData):
        flux = sim_data["flux"].flux
        reflectance = 1 - flux
        reflectance_max = float(np.max(reflectance).values)
        # TODO: compute the bandwidth Delta lambda
        return reflectance_max

    def calc_delta_n(sim):
        structures = sim.structures
        s1 = math.sqrt(structures[0].medium.permittivity)
        s2 = math.sqrt(structures[1].medium.permittivity)
        return abs(s1 - s2)

    raw_features = []
    raw_labels = []
    # delta_n = []
    for sim_file in hdf5_files:
        sim_data = td.SimulationData.from_file(sim_file)
        raw_features.append(create_1d_array(sim_data.simulation, resolution))
        raw_labels.append(get_data(sim_data))
        # delta_n.append(calc_delta_n(sim_data.simulation))

    raw_features = np.array(raw_features)
    raw_labels = np.array(raw_labels)
    # delta_n = np.array(delta_n)

    # Change shape for convolution
    raw_features = np.expand_dims(raw_features, axis=1)

    return raw_features, raw_labels  # , delta_n


# %%

model = AI_Model(output_dir, data_dir, rng_seed=2)
bragg_kwargs = {"resolution": 65}
model.load_data(
    bragg_grating_data,
    test_percentage,
    valid_percentage,
    batch_size,
    pickle_name="bragg_gaussian_750.pkl",
    fn_data_kwargs=bragg_kwargs,
)

plt.plot(model.train_loaded.dataset[0][0].tolist()[0])
plt.show()
# model.plot_label_distribution(model.train_labels)

# %%

# cl1 = 16
# cl2 = 8
# c_kernal = 7
# l1 = 1344
# dropout = 0.45
# learning_rate = 0.000225
# weight_decay = 5.99e-04
# net = NN(bragg_kwargs["resolution"] - 1, cl1, cl2, c_kernal, l1, dropout)
# optimiser = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = nn.MSELoss()
# trained_net = model.train_model("Model1", net, optimiser, loss_function, epochs, plot_output=True)
# test_rmse, _, test_predictions = model.validate_model(trained_net, "test")

# model.plot_label_distribution(model.test_labels, predictions=test_predictions, bin_count=20, plot_error=True)

# %%

network_dict = {
    "network": NN,
    "kwargs": {"resolution": bragg_kwargs["resolution"] - 1},
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

# %%

trained_network = model.train_model(
    "BestModel", best_network, best_optimizer, loss_function, 1000, True
)
test_rmse, _, test_predictions = model.validate_model(trained_network, "test")
valid_rmse, _, valid_predictions = model.validate_model(trained_network, "valid")

model.print_array_stats(model.test_labels, test_rmse)
model.print_array_stats(test_predictions)
model.print_array_stats(model.valid_labels, valid_rmse)
model.print_array_stats(valid_predictions)

model.plot_label_distribution(
    model.test_labels, predictions=test_predictions, bin_count=10, plot_error=True
)
# %%
