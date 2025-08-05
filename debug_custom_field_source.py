#!/usr/bin/env python3
"""
Debug script to understand why CustomFieldSource is returning zero gradients.
"""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np

import tidy3d as td
from tidy3d.web import run


def debug_custom_field_source():
    """Debug why CustomFieldSource is returning zero gradients."""

    def create_simple_simulation(val):
        """Create a simulation with a traced CustomFieldSource."""

        # Create traced field data
        x = np.linspace(-0.5, 0.5, 5)
        y = np.linspace(-0.5, 0.5, 5)
        z = np.array([0])
        f = [2e14]
        coords = {"x": x, "y": y, "z": z, "f": f}

        # Create traced field data - this is what we want to differentiate
        field_data = val * np.ones((5, 5, 1, 1))
        scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)

        # Create field dataset with traced data
        field_dataset = td.FieldDataset(Ex=scalar_field)

        # Create CustomFieldSource with traced dataset
        custom_source = td.CustomFieldSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            field_dataset=field_dataset,
        )

        # Create simulation
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[custom_source],
            monitors=[
                td.FieldMonitor(
                    size=(1.0, 1.0, 0.0), center=(0, 0, 0), freqs=[2e14], name="field_monitor"
                )
            ],
        )

        return sim

    def objective(val):
        """Objective function that depends on CustomFieldSource parameters."""

        sim = create_simple_simulation(val)
        sim_data = run(sim, task_name="debug_field_source", local_gradient=True)

        # Extract field data
        field_data = sim_data.load_field_monitor("field_monitor")

        if hasattr(field_data, "Ex") and field_data.Ex is not None:
            # Compute objective from field data
            field_value = field_data.Ex.isel(x=2, y=2, z=0, f=0).values
            objective_value = anp.abs(field_value) ** 2
        else:
            # Fallback objective
            objective_value = val**2

        return objective_value

    # Test gradient computation
    val = 1.0
    grad = ag.grad(objective)(val)

    print(f"CustomFieldSource gradient: {grad}")

    # Test numerical derivative to verify
    delta = 1e-4
    obj_plus = objective(val + delta)
    obj_minus = objective(val - delta)
    grad_num = (obj_plus - obj_minus) / (2 * delta)

    print(f"Numerical gradient: {grad_num}")
    print(f"Relative error: {abs(grad - grad_num) / (abs(grad) + 1e-10)}")

    # Check if the gradient is exactly zero
    if grad == 0.0:
        print(
            "❌ CustomFieldSource gradient is exactly 0.0 - this suggests an architectural issue!"
        )
        print(
            "   The issue is likely that no field components are being found in derivative_info.E_adj"
        )
    else:
        print("✅ CustomFieldSource gradient is non-zero")

    return grad, grad_num


if __name__ == "__main__":
    print("=" * 60)
    print("Debugging CustomFieldSource Zero Gradient Issue")
    print("=" * 60)

    grad, grad_num = debug_custom_field_source()

    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    print("If the gradient is exactly 0.0, it means:")
    print("1. The CustomFieldSource._compute_derivatives method is being called")
    print("2. But no field components are found in derivative_info.E_adj")
    print("3. So it falls back to setting gradients to 0.0")
    print("\nThis suggests the autograd system isn't properly detecting")
    print("the traced parameters in the CustomFieldSource field_dataset.")
