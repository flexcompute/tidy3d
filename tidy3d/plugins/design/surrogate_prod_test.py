# %% General
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

import tidy3d as td
from tidy3d.plugins.design.surrogate_model_funcs import calc_rmse
from tidy3d.plugins.design.surrogate_models import Basic1DCNN as NN

# Local
from tidy3d.plugins.design.surrogate_preprocess_funcs import (
    create_1d_array,
    load_scalers,
    load_sim_data,
    scale_feature,
)
from tidy3d.web.api.container import Batch

# Set random seeds
SEED = 5
np.random.seed(SEED)
torch.cuda.empty_cache()
torch.manual_seed(SEED)

MODEL_DIR = "/home/matt/Documents/Flexcompute/bragg/model/third_model"
DATA_DIR = "/home/matt/Documents/Flexcompute/bragg/model/data"
os.chdir(MODEL_DIR)

MODEL_NAME = "model.pt"
SAMPLE_POINTS = 25
RESOLUTION = 65

# Config for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

cl1 = 8
cl2 = 32
c_kernal = 9
l1 = 1856
dropout = 0.15

label_scaler, feature_scaler = load_scalers()
trained_net = NN(RESOLUTION - 1, cl1, cl2, c_kernal, l1, dropout).to(device)
trained_net.load_state_dict(torch.load(MODEL_NAME))


# Post function used to get real values
def get_data(sim_data: td.SimulationData) -> dict[str, float]:
    flux = sim_data["flux"].flux
    reflectance = 1 - flux
    reflectance_max = float(np.max(reflectance).values)
    # TODO: compute the bandwidth Delta lambda
    return reflectance_max


# %% Run pre-calculated
os.chdir(DATA_DIR)
hdf5_files = [fileName for fileName in os.listdir() if ".hdf5" in fileName]
raw_features, results, _ = load_sim_data(RESOLUTION, get_data, hdf5_files)

# %% Run from scratch

lambda0 = 1.0

freq0 = td.C_0 / lambda0
fwidth = freq0 / 2
run_time = 150 / fwidth

num_freqs = 101
freqs = np.linspace(freq0 - fwidth / 2, freq0 + fwidth / 2, num_freqs)

num_layers = 10
min_steps_per_wvl = 15

buffer = 3 * lambda0

n_avg = 2.0
periodicity = lambda0 / 2 / n_avg

mnt_name = "flux"


def make_sim(delta_n: float, sigma: float) -> td.Simulation:
    # n1 = n_avg - delta_n / 2
    # n2 = n_avg + delta_n / 2

    Lz = periodicity * num_layers + 2 * buffer

    def gaussian_modulation(z: float, sigma: float) -> float:
        return np.exp(-(abs(z) ** 2) / (2 * sigma**2))

    layers = []
    for i in range(num_layers):
        for j, sign in enumerate((-1, 1)):
            z_center = -Lz / 2 + buffer + periodicity / 2 + i * periodicity + j * periodicity / 2
            n = n_avg + sign * delta_n / 2 * gaussian_modulation(z=z_center, sigma=sigma)

            layer = td.Structure(
                geometry=td.Box(
                    size=(td.inf, td.inf, periodicity / 2),
                    center=(0, 0, z_center),
                ),
                medium=td.Medium(permittivity=n**2),
            )
            layers.append(layer)

    src = td.PlaneWave(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -Lz / 2 + buffer / 2),
        direction="+",
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    )

    mnt = td.FluxMonitor(
        size=(td.inf, td.inf, 0), center=(0, 0, +Lz / 2 - buffer / 2), freqs=freqs, name=mnt_name
    )

    return td.Simulation(
        size=(0, 0, Lz),
        structures=layers,
        monitors=[mnt],
        sources=[src],
        grid_spec=td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=min_steps_per_wvl),
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True),
        run_time=run_time,
        medium=td.Medium(permittivity=n_avg**2),
    )


simulations = {}
raw_features = []
for i in range(SAMPLE_POINTS):
    delta_n = np.random.uniform(0.0, 0.5, 1)[0]
    sigma = np.random.uniform(0.1, 0.9, 1)[0]
    sim = make_sim(delta_n=delta_n, sigma=sigma)
    simulations[i] = sim
    raw_features.append(create_1d_array(sim, RESOLUTION))

# %% Make prediction

# Change shape for convolution
raw_features = np.expand_dims(raw_features, axis=1)
raw_features = np.expand_dims(raw_features, axis=0)

scaled_features = scale_feature(raw_features, feature_scaler)

torch_tensor = torch.Tensor(scaled_features)
torch_data = torch.utils.data.TensorDataset(torch_tensor)


def make_prediction(tensor):
    with torch.no_grad():
        trained_net.eval()
        netout = trained_net(tensor[0].to(device)).cpu().numpy()

        prediction = netout.reshape(-1, 1)
        prediction = label_scaler.inverse_transform(prediction)
        prediction = prediction.reshape(prediction.shape[0])

        return prediction.tolist()


predictions = make_prediction(torch_data[0])

# %% Run Tidy3D
# Could be moved above prediction

batch_data = Batch(simulations=simulations).run(path_dir=DATA_DIR)

results = []
for _, sim in batch_data.items():
    reflectance = get_data(sim)
    results.append(reflectance)


# %% Analyse

pred_array = np.array(predictions).reshape(-1, 1)
valid_array = np.array(results).reshape(-1, 1)

RMSE = round(calc_rmse(pred_array, valid_array), 6)
MAE = round(mean_absolute_error(valid_array, pred_array), 6)

print(f"Prediction Mean: {round(valid_array.mean(), 3)}")
print(f"RMSE: {RMSE}; MAE: {MAE}")
print(f"% RMSE: {round(RMSE / valid_array.mean() * 100, 1)}%")

line = np.linspace(0, 0.9, 100)

plt.scatter(valid_array, pred_array)
plt.plot(line, line)
plt.xlabel("Real Value")
plt.ylabel("Predicted Value")
plt.show()
# %%
