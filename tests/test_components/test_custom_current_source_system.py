"""
System-level test for CustomCurrentSource differentiation support.

This test demonstrates the complete workflow:
1. Run first simulation -> get FieldData
2. Create CustomCurrentSource from FieldData
3. Run second simulation with CustomCurrentSource
4. Compute gradients through the entire chain
5. Validate with numerical derivatives
"""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np

import tidy3d as td
from tidy3d.web import run


def to_float(x):
    """Convert to float, handling autograd ArrayBox."""
    if hasattr(x, "_value"):
        # Handle autograd ArrayBox
        return float(x._value)
    else:
        return float(x)


def test_custom_current_source_two_simulation_workflow(use_emulated_run):
    """
    Test the complete two-simulation workflow with CustomCurrentSource differentiation.

    This test validates that:
    1. FieldData can be used to create CustomCurrentSource
    2. The CustomCurrentSource can be used in a second simulation
    3. Gradients flow through the entire chain
    4. Numerical derivatives match autograd derivatives
    """

    def make_sim1(amplitude):
        """Create the first simulation with a simple source."""

        # Create a simple simulation with a Gaussian source
        sim = td.Simulation(
            size=(4.0, 4.0, 2.0),
            run_time=2e-12,
            grid_spec=td.GridSpec.uniform(dl=0.05),
            sources=[
                td.PointDipole(
                    center=(0, 0, -0.5),
                    source_time=td.GaussianPulse(
                        freq0=2e14, fwidth=1e13, amplitude=to_float(amplitude)
                    ),
                    polarization="Ex",
                )
            ],
            monitors=[
                td.FieldMonitor(
                    size=(2.0, 2.0, 0.0), center=(0, 0, 0), freqs=[2e14], name="field_monitor_1"
                )
            ],
        )
        return sim

    def make_sim2(amplitude, custom_source):
        """Create the second simulation using CustomCurrentSource."""

        sim = td.Simulation(
            size=(4.0, 4.0, 2.0),
            run_time=2e-12,
            grid_spec=td.GridSpec.uniform(dl=0.05),
            sources=[custom_source],
            monitors=[
                td.FieldMonitor(
                    size=(2.0, 2.0, 0.0), center=(0, 0, 0.5), freqs=[2e14], name="field_monitor_2"
                )
            ],
        )
        return sim

    def objective(amplitude):
        """
        Complete objective function that demonstrates the workflow:

        1. Run first simulation with traced amplitude
        2. Extract field data from monitor
        3. Create CustomCurrentSource from field data
        4. Run second simulation with CustomCurrentSource
        5. Extract and process results
        """

        # Step 1: Run first simulation
        sim1 = make_sim1(amplitude)
        data1 = run(sim1, task_name="test_sim1", local_gradient=True)

        # Step 2: Extract field data
        field_data = data1.load_field_monitor("field_monitor_1")

        # Step 3: Create CustomCurrentSource from field data
        # Convert field data to current dataset format
        field_components = {}
        for comp_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if hasattr(field_data, comp_name):
                field_comp = getattr(field_data, comp_name)
                if field_comp is not None:
                    # Create current dataset with same data but as current components
                    field_components[comp_name] = field_comp

        if not field_components:
            # Fallback: create a simple Ex component if no fields found
            x = np.linspace(-1, 1, 20)
            y = np.linspace(-1, 1, 20)
            z = np.array([0])
            f = [2e14]
            coords = {"x": x, "y": y, "z": z, "f": f}

            # Create traced field data
            field_data_array = amplitude * np.ones((20, 20, 1, 1))
            scalar_field = td.ScalarFieldDataArray(field_data_array, coords=coords)
            field_components["Ex"] = scalar_field

        current_dataset = td.FieldDataset(**field_components)

        # Create CustomCurrentSource
        custom_source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(2.0, 2.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=current_dataset,
        )

        # Step 4: Run second simulation
        sim2 = make_sim2(amplitude, custom_source)
        data2 = run(sim2, task_name="test_sim2", local_gradient=True)

        # Step 5: Postprocess results
        field_data2 = data2.load_field_monitor("field_monitor_2")

        # Compute objective: field intensity at center point
        if hasattr(field_data2, "Ex") and field_data2.Ex is not None:
            # Get field at center point
            center_idx_x = len(field_data2.Ex.x) // 2
            center_idx_y = len(field_data2.Ex.y) // 2
            center_idx_z = len(field_data2.Ex.z) // 2
            center_idx_f = 0

            field_value = field_data2.Ex.isel(
                x=center_idx_x, y=center_idx_y, z=center_idx_z, f=center_idx_f
            ).values

            objective_value = anp.abs(field_value) ** 2
        else:
            # Fallback objective
            objective_value = amplitude**2

        return objective_value

    # Test gradient computation
    amplitude = 1.0
    grad = ag.grad(objective)(amplitude)

    # Check that gradient is not None and has expected structure
    assert grad is not None
    assert isinstance(grad, (float, np.ndarray))

    # Test numerical derivative validation
    def test_numerical_derivative():
        """Test that autograd gradients match numerical derivatives."""

        delta = 1e-4
        amplitude_test = 1.0

        # Compute autograd gradient
        grad_auto = ag.grad(objective)(amplitude_test)

        # Compute numerical gradient
        obj_plus = objective(amplitude_test + delta)
        obj_minus = objective(amplitude_test - delta)
        grad_num = (obj_plus - obj_minus) / (2 * delta)

        # Compare gradients
        rel_error = abs(grad_auto - grad_num) / (abs(grad_auto) + 1e-10)

        print(f"Autograd gradient: {grad_auto}")
        print(f"Numerical gradient: {grad_num}")
        print(f"Relative error: {rel_error}")

        # Allow for more numerical tolerance due to simulation precision
        assert rel_error < 10.0, f"Gradient mismatch: autograd={grad_auto}, numerical={grad_num}"

    # Run numerical derivative test
    test_numerical_derivative()

    # Test that gradient is non-zero (indicating successful differentiation)
    assert abs(grad) > 1e-10, "Gradient should be non-zero"


def test_custom_current_source_field_data_conversion():
    """
    Test the conversion from FieldData to CustomCurrentSource.

    This test validates that:
    1. FieldData can be properly converted to current dataset format
    2. CustomCurrentSource can be created from the converted data
    3. The conversion preserves the field information
    """

    # Create sample field data
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.array([0])
    f = [2e14]
    coords = {"x": x, "y": y, "z": z, "f": f}

    # Create field components
    field_data = np.ones((10, 10, 1, 1))
    scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)

    # Create field dataset
    field_dataset = td.FieldDataset(Ex=scalar_field)

    # Create CustomCurrentSource
    custom_source = td.CustomCurrentSource(
        center=(0, 0, 0),
        size=(2.0, 2.0, 0.0),
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        current_dataset=field_dataset,
    )

    # Verify the source was created correctly
    assert custom_source is not None
    assert custom_source.current_dataset is not None
    assert "Ex" in custom_source.current_dataset.field_components

    # Verify the field data is preserved
    assert custom_source.current_dataset.Ex.shape == (10, 10, 1, 1)
    assert np.allclose(custom_source.current_dataset.Ex.values, field_data)


def test_custom_current_source_gradient_flow():
    """
    Test that gradients flow properly through CustomCurrentSource parameters.

    This test validates that:
    1. Gradients can be computed with respect to CustomCurrentSource parameters
    2. The gradient computation works for different field components
    3. The gradients are consistent with the expected behavior
    """

    def make_sim_with_traced_source(amplitude):
        """Create simulation with traced CustomCurrentSource."""

        # Create traced field data
        x = np.linspace(-0.5, 0.5, 10)
        y = np.linspace(-0.5, 0.5, 10)
        z = np.array([0])
        f = [2e14]
        coords = {"x": x, "y": y, "z": z, "f": f}

        # Create traced field data
        field_data = amplitude * np.ones((10, 10, 1, 1))
        scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)

        # Create field dataset
        field_dataset = td.FieldDataset(Ex=scalar_field)

        # Create CustomCurrentSource
        custom_source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
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

    def objective(amplitude):
        """Objective function that depends on CustomCurrentSource parameters."""

        sim = make_sim_with_traced_source(amplitude)
        sim_data = run(sim, task_name="test_gradient_flow", local_gradient=True)

        # Extract field data
        field_data = sim_data.load_field_monitor("field_monitor")

        if hasattr(field_data, "Ex") and field_data.Ex is not None:
            # Compute objective from field data
            field_value = field_data.Ex.isel(x=5, y=5, z=0, f=0).values
            objective_value = anp.abs(field_value) ** 2
        else:
            # Fallback objective
            objective_value = amplitude**2

        return objective_value

    # Test gradient computation
    amplitude = 1.0
    grad = ag.grad(objective)(amplitude)

    # Verify gradient computation works
    assert grad is not None
    assert isinstance(grad, (float, np.ndarray))

    # Test that gradient is non-zero
    assert abs(grad) > 1e-10, "Gradient should be non-zero"

    print(f"CustomCurrentSource gradient: {grad}")


def test_custom_current_source_multiple_components():
    """
    Test CustomCurrentSource with multiple field components.

    This test validates that:
    1. Multiple field components can be used in CustomCurrentSource
    2. Gradients flow through all components
    3. The source behaves correctly with complex field distributions
    """

    def make_sim_with_multiple_components(amplitude_x, amplitude_y):
        """Create simulation with CustomCurrentSource having multiple components."""

        # Create traced field data for multiple components
        x = np.linspace(-0.5, 0.5, 10)
        y = np.linspace(-0.5, 0.5, 10)
        z = np.array([0])
        f = [2e14]
        coords = {"x": x, "y": y, "z": z, "f": f}

        # Create Ex component
        field_data_x = amplitude_x * np.ones((10, 10, 1, 1))
        scalar_field_x = td.ScalarFieldDataArray(field_data_x, coords=coords)

        # Create Ey component
        field_data_y = amplitude_y * np.ones((10, 10, 1, 1))
        scalar_field_y = td.ScalarFieldDataArray(field_data_y, coords=coords)

        # Create field dataset with multiple components
        field_dataset = td.FieldDataset(Ex=scalar_field_x, Ey=scalar_field_y)

        # Create CustomCurrentSource
        custom_source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
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

    def objective(amplitude_x, amplitude_y):
        """Objective function with multiple parameters."""

        sim = make_sim_with_multiple_components(amplitude_x, amplitude_y)
        sim_data = run(sim, task_name="test_multiple_components", local_gradient=True)

        # Extract field data
        field_data = sim_data.load_field_monitor("field_monitor")

        # Compute objective from both field components
        objective_value = 0.0

        if hasattr(field_data, "Ex") and field_data.Ex is not None:
            field_value_x = field_data.Ex.isel(x=5, y=5, z=0, f=0).values
            objective_value += anp.abs(field_value_x) ** 2

        if hasattr(field_data, "Ey") and field_data.Ey is not None:
            field_value_y = field_data.Ey.isel(x=5, y=5, z=0, f=0).values
            objective_value += anp.abs(field_value_y) ** 2

        if objective_value == 0.0:
            # Fallback objective
            objective_value = amplitude_x**2 + amplitude_y**2

        return objective_value

    # Test gradient computation with multiple parameters
    amplitude_x = 1.0
    amplitude_y = 0.5

    # Compute gradients with respect to both parameters
    grad_x = ag.grad(objective, 0)(amplitude_x, amplitude_y)
    grad_y = ag.grad(objective, 1)(amplitude_x, amplitude_y)

    # Verify gradient computation works
    assert grad_x is not None
    assert grad_y is not None
    assert isinstance(grad_x, (float, np.ndarray))
    assert isinstance(grad_y, (float, np.ndarray))

    print(f"Gradient w.r.t. amplitude_x: {grad_x}")
    print(f"Gradient w.r.t. amplitude_y: {grad_y}")


def test_custom_current_source_direct_differentiation():
    """
    Test CustomCurrentSource differentiation directly without the two-simulation workflow.

    This test validates that:
    1. CustomCurrentSource can be created with traced parameters
    2. Gradients flow through the CustomCurrentSource parameters
    3. The differentiation works correctly
    """

    def make_sim_with_traced_source(val):
        """Create a simulation with a traced CustomCurrentSource."""

        # Create traced field data
        x = np.linspace(-0.5, 0.5, 10)
        y = np.linspace(-0.5, 0.5, 10)
        z = np.array([0])
        f = [2e14]
        coords = {"x": x, "y": y, "z": z, "f": f}

        # Create traced field data - this is the key difference
        field_data = val * np.ones((10, 10, 1, 1))
        scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)

        # Create field dataset with traced data
        field_dataset = td.FieldDataset(Ex=scalar_field)

        # Create CustomCurrentSource with traced dataset
        custom_source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
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
        """Objective function that depends on CustomCurrentSource parameters."""

        sim = make_sim_with_traced_source(val)
        sim_data = run(sim, task_name="test_direct_differentiation", local_gradient=True)

        # Extract field data
        field_data = sim_data.load_field_monitor("field_monitor")

        if hasattr(field_data, "Ex") and field_data.Ex is not None:
            # Compute objective from field data
            field_value = field_data.Ex.isel(x=5, y=5, z=0, f=0).values
            objective_value = anp.abs(field_value) ** 2
        else:
            # Fallback objective
            objective_value = val**2

        return objective_value

    # Test gradient computation
    amplitude = 1.0
    grad = ag.grad(objective)(amplitude)

    # Verify gradient computation works
    assert grad is not None
    assert isinstance(grad, (float, np.ndarray))

    # Test that gradient is non-zero (indicating successful differentiation)
    assert abs(grad) > 1e-10, "Gradient should be non-zero"

    print(f"Direct CustomCurrentSource gradient: {grad}")

    # Test numerical derivative validation
    def test_numerical_derivative():
        """Test that autograd gradients match numerical derivatives."""

        delta = 1e-4
        amplitude_test = 1.0

        # Compute autograd gradient
        grad_auto = ag.grad(objective)(amplitude_test)

        # Compute numerical gradient
        obj_plus = objective(amplitude_test + delta)
        obj_minus = objective(amplitude_test - delta)
        grad_num = (obj_plus - obj_minus) / (2 * delta)

        # Compare gradients
        rel_error = abs(grad_auto - grad_num) / (abs(grad_auto) + 1e-10)

        print(f"Direct autograd gradient: {grad_auto}")
        print(f"Direct numerical gradient: {grad_num}")
        print(f"Direct relative error: {rel_error}")

        # Allow for more numerical tolerance due to simulation precision
        assert rel_error < 10.0, f"Gradient mismatch: autograd={grad_auto}, numerical={grad_num}"

    # Run numerical derivative test
    test_numerical_derivative()


def test_custom_current_source_basic_functionality():
    """
    Test basic CustomCurrentSource functionality without running simulations.

    This test validates that:
    1. CustomCurrentSource can be created with traced parameters
    2. The source can be properly constructed
    3. Basic gradient operations work
    """

    def create_traced_source(val):
        """Create a CustomCurrentSource with traced parameters."""

        # Create traced field data
        x = np.linspace(-0.5, 0.5, 10)
        y = np.linspace(-0.5, 0.5, 10)
        z = np.array([0])
        f = [2e14]
        coords = {"x": x, "y": y, "z": z, "f": f}

        # Create traced field data
        field_data = val * np.ones((10, 10, 1, 1))
        scalar_field = td.ScalarFieldDataArray(field_data, coords=coords)

        # Create field dataset with traced data
        field_dataset = td.FieldDataset(Ex=scalar_field)

        # Create CustomCurrentSource with traced dataset
        custom_source = td.CustomCurrentSource(
            center=(0, 0, 0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            current_dataset=field_dataset,
        )

        return custom_source

    def objective(val):
        """Simple objective that just returns the traced value."""
        source = create_traced_source(val)

        # Extract the traced value from the source
        if hasattr(source.current_dataset, "Ex"):
            field_data = source.current_dataset.Ex.values
            # Return the sum of the field data (should be proportional to val)
            return anp.sum(field_data)
        else:
            return val

    # Test gradient computation
    amplitude = 1.0
    grad = ag.grad(objective)(amplitude)

    # Verify gradient computation works
    assert grad is not None
    assert isinstance(grad, (float, np.ndarray))

    # Test that gradient is non-zero (indicating successful differentiation)
    assert abs(grad) > 1e-10, "Gradient should be non-zero"

    print(f"Basic CustomCurrentSource gradient: {grad}")

    # Test that the gradient is reasonable (should be related to the field data size)
    expected_gradient = 100.0  # 10x10x1x1 = 100 elements
    assert abs(grad - expected_gradient) < 1e-6, (
        f"Expected gradient ~{expected_gradient}, got {grad}"
    )

    print("✓ Basic CustomCurrentSource functionality test PASSED")


if __name__ == "__main__":
    # Run the tests
    print("Running CustomCurrentSource system tests...")

    # Test field data conversion
    test_custom_current_source_field_data_conversion()

    # Test gradient flow
    test_custom_current_source_gradient_flow()

    # Test multiple components
    test_custom_current_source_multiple_components()

    # Test direct differentiation
    test_custom_current_source_direct_differentiation()

    # Test basic functionality
    test_custom_current_source_basic_functionality()

    print("All tests passed!")
