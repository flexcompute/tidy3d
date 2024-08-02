# %%
import os

import numpy as np
import torch
import torch.utils

import tidy3d.plugins.design as tdd
from tidy3d.plugins.design.surrogate_models import BasicReducingANN as NN
from tidy3d.plugins.design.surrogate_preprocess_funcs import load_scalers, scale_feature

INIT_NEURONS = 1024
DROPOUT = 0.2
num_d = 13

os.chdir("/home/matt/Documents/Flexcompute/tidy3d/tidy3d/plugins/design/")
label_scaler, feature_scaler = load_scalers()

# Config for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

trained_net = NN(num_d, INIT_NEURONS, DROPOUT).to(device)
trained_net.load_state_dict(torch.load("model.pt"))


def eval_fn(**params):
    param_arr = np.array(list(params.values()))
    param_arr = param_arr.reshape((1, -1))
    scaled_params = scale_feature(param_arr, feature_scaler)

    torch_tensor = torch.Tensor(scaled_params)
    torch_data = torch.utils.data.TensorDataset(torch_tensor)

    with torch.no_grad():
        trained_net.eval()
        netout = trained_net(torch_data[0][0].to(device)).cpu().numpy()

    prediction = netout.reshape(-1, 1)
    prediction = label_scaler.inverse_transform(prediction)

    return prediction[0][0]


method = tdd.MethodMonteCarlo(
    num_points=1000,
    rng_seed=1,
)
num_d = 13
parameters = [tdd.ParameterFloat(name=f"w_{i}", span=(0.5, 1.6)) for i in range(num_d)]

design_space = tdd.DesignSpace(method=method, parameters=parameters, task_name="bay_opt_notebook")

results = design_space.run(eval_fn)
df = results.to_dataframe()
print()
# %%
