# Autograd to PyTorch Wrapper for Tidy3D

This wrapper allows you to seamlessly convert autograd functions to PyTorch functions, enabling the use of Tidy3D simulations within PyTorch.

## Examples

### Basic Usage

This module can be used to convert any autograd function to a PyTorch function:

```python
import torch
import autograd.numpy as anp

from tidy3d.plugins.pytorch.wrapper import to_torch

@to_torch
def my_function(x):
    return anp.sum(anp.sin(x)**2)

x = torch.rand(10, requires_grad=True)
y = my_function(x)
y.backward()  # backward works as expected, even though the function is defined in terms of autograd.numpy
print(x.grad)  # gradients are available in the input tensor
```

### Usage with Tidy3D

The `to_torch` wrapper can be used to convert an objective function that depends on Tidy3D simulations to a PyTorch function:

```python
import torch
import autograd.numpy as anp

import tidy3d as td
import tidy3d.web as web

from tidy3d.plugins.pytorch.wrapper import to_torch

@to_torch
def tidy3d_objective(params):
    sim = make_sim(params)
    sim_data = web.run(sim, task_name="pytorch_example")
    flux = sim_data["flux"].flux.values
    return anp.sum(flux)

params = torch.rand(10, requires_grad=True)
y = tidy3d_objective(params)
y.backward()
print(params.grad)
```
