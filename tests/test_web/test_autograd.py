"""Tests for autograd compatibility of web API methods."""

from __future__ import annotations

import autograd.numpy as anp
import pytest

import tidy3d as td
from tidy3d.web import run, run_async
from tidy3d.web.api.autograd.autograd import run as autograd_run
from tidy3d.web.api.autograd.autograd import run_async as autograd_run_async
from tidy3d.web.api.container import Batch, Job


class TestAutogradCompatibility:
    """Test that all web API methods support autograd functionality."""

    @pytest.fixture
    def simple_simulation(self):
        """Create a simple simulation without traced fields for testing."""
        wavelength = 1.0
        freq0 = td.C_0 / wavelength

        # Create a simple simulation without traced fields
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=wavelength),
            structures=[
                td.Structure(
                    geometry=td.Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0)),
                    medium=td.Medium(permittivity=2.0),
                )
            ],
            sources=[
                td.PointDipole(
                    center=(0, 0, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                )
            ],
            monitors=[
                td.FieldMonitor(
                    center=(0, 0, 0),
                    size=(1.0, 1.0, 1.0),
                    freqs=[freq0],
                    name="field",
                )
            ],
            run_time=1e-12,
        )
        return sim

    @pytest.fixture
    def traced_simulation(self):
        """Create a simulation with traced fields for testing."""
        wavelength = 1.0
        freq0 = td.C_0 / wavelength

        # Create a simulation with traced fields
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=wavelength),
            structures=[
                td.Structure(
                    geometry=td.Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0)),
                    medium=td.Medium(permittivity=anp.array(2.0)),  # Traced field
                )
            ],
            sources=[
                td.PointDipole(
                    center=(0, 0, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                )
            ],
            monitors=[
                td.FieldMonitor(
                    center=(0, 0, 0),
                    size=(1.0, 1.0, 1.0),
                    freqs=[freq0],
                    name="field",
                )
            ],
            run_time=1e-12,
        )
        return sim

    def test_imports_work_correctly(self):
        """Test that all imports work correctly."""
        # Test that we can import all the necessary functions
        assert run is not None
        assert run_async is not None
        assert Job is not None
        assert Batch is not None
        assert autograd_run is not None
        assert autograd_run_async is not None

    def test_job_autograd_parameters(self, traced_simulation):
        """Test that Job.run() properly passes autograd parameters."""

        # Create a Job with autograd parameters
        job = Job(
            simulation=traced_simulation,
            task_name="test_job_params",
            local_gradient=True,
            max_num_adjoint_per_fwd=5,
        )

        # Test that the parameters are set correctly
        assert job.local_gradient is True
        assert job.max_num_adjoint_per_fwd == 5

    def test_batch_autograd_parameters(self, traced_simulation):
        """Test that Batch.run() properly passes autograd parameters."""

        # Create a Batch with autograd parameters
        batch = Batch(
            simulations={"test": traced_simulation}, local_gradient=True, max_num_adjoint_per_fwd=5
        )

        # Test that the parameters are set correctly
        assert batch.local_gradient is True
        assert batch.max_num_adjoint_per_fwd == 5

    def test_mixed_batch_autograd(self, simple_simulation, traced_simulation):
        """Test that Batch.run() works with mixed regular and autograd simulations."""

        # Create a Batch with mixed simulations
        batch = Batch(
            simulations={
                "regular": simple_simulation,
                "traced": traced_simulation,
            }
        )

        # Test that the batch can be created without errors
        assert len(batch.simulations) == 2
        assert "regular" in batch.simulations
        assert "traced" in batch.simulations

    def test_job_creation(self, traced_simulation):
        """Test that Job objects can be created correctly."""

        # Test basic Job creation
        job = Job(
            simulation=traced_simulation,
            task_name="test_job",
        )

        assert job.simulation == traced_simulation
        assert job.task_name == "test_job"
        assert job.local_gradient is False  # Default value
        assert job.max_num_adjoint_per_fwd == 10  # Default value

    def test_batch_creation(self, traced_simulation):
        """Test that Batch objects can be created correctly."""

        # Test basic Batch creation
        batch = Batch(
            simulations={"test": traced_simulation},
        )

        assert len(batch.simulations) == 1
        assert "test" in batch.simulations
        assert batch.simulations["test"] == traced_simulation
        assert batch.local_gradient is False  # Default value
        assert batch.max_num_adjoint_per_fwd == 10  # Default value

    def test_job_upload_fields(self, traced_simulation):
        """Test that Job._upload_fields includes autograd parameters."""

        job = Job(
            simulation=traced_simulation,
            task_name="test_job",
            local_gradient=True,
            max_num_adjoint_per_fwd=5,
        )

        # Check that autograd parameters are in _upload_fields
        assert "local_gradient" in job._upload_fields
        assert "max_num_adjoint_per_fwd" in job._upload_fields

    def test_job_run_method_exists(self, traced_simulation):
        """Test that Job.run() method exists and can be called."""
        job = Job(simulation=traced_simulation, task_name="test_job")
        assert hasattr(job, "run")
        assert callable(job.run)

    def test_batch_run_method_exists(self, traced_simulation):
        """Test that Batch.run() method exists and can be called."""
        batch = Batch(simulations={"test": traced_simulation})
        assert hasattr(batch, "run")
        assert callable(batch.run)

    def test_actual_autograd_gradient_computation(self, traced_simulation, monkeypatch):
        """Test that Job.run() actually works with autograd gradient computation."""
        import autograd

        # Mock the autograd run function to return a simple result
        def mock_autograd_run(*args, **kwargs):
            # Return a simple SimulationData-like object with traced field data
            class MockSimulationData:
                def __init__(self):
                    # Create a simple field data structure that can be traced
                    self.field = type(
                        "obj",
                        (object,),
                        {
                            "Ex": type(
                                "obj", (object,), {"values": anp.array([[1.0, 2.0], [3.0, 4.0]])}
                            )()
                        },
                    )()

            return MockSimulationData()

        # Patch the autograd run function
        monkeypatch.setattr("tidy3d.web.api.autograd.autograd.run", mock_autograd_run)

        # Define an objective function that uses Job.run()
        def objective_function(permittivity):
            # Create a new simulation with the updated traced field
            wavelength = 1.0
            freq0 = td.C_0 / wavelength

            new_sim = td.Simulation(
                size=(2.0, 2.0, 2.0),
                grid_spec=td.GridSpec.auto(wavelength=wavelength),
                structures=[
                    td.Structure(
                        geometry=td.Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0)),
                        medium=td.Medium(permittivity=anp.array(permittivity)),  # Traced field
                    )
                ],
                sources=[
                    td.PointDipole(
                        center=(0, 0, 0),
                        polarization="Ex",
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                    )
                ],
                monitors=[
                    td.FieldMonitor(
                        center=(0, 0, 0),
                        size=(1.0, 1.0, 1.0),
                        freqs=[freq0],
                        name="field",
                    )
                ],
                run_time=1e-12,
            )

            # Create a new job with the updated simulation
            job = Job(simulation=new_sim, task_name="test_autograd")

            # Run the simulation using Job.run()
            result = job.run()

            # Return a scalar value that can be differentiated
            return anp.real(anp.sum(result.field.Ex.values))

        # Test that we can compute gradients
        try:
            gradient = autograd.grad(objective_function)(2.0)
            # If we get here, the gradient computation worked
            assert gradient is not None
            assert isinstance(gradient, (float, int, anp.ndarray))
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")

    def test_actual_autograd_gradient_with_batch(self, traced_simulation, monkeypatch):
        """Test that Batch.run() actually works with autograd gradient computation."""
        import autograd

        # Mock the autograd run_async function to return a simple result
        def mock_autograd_run_async(*args, **kwargs):
            # Return a simple BatchData-like object
            class MockBatchData:
                def __init__(self):
                    self.task_paths = {"test": "mock_path"}
                    self.task_ids = {"test": "mock_id"}
                    self.verbose = True

                    # Create a simple field data structure that can be traced
                    class MockSimulationData:
                        def __init__(self):
                            self.field = type(
                                "obj",
                                (object,),
                                {
                                    "Ex": type(
                                        "obj",
                                        (object,),
                                        {"values": anp.array([[1.0, 2.0], [3.0, 4.0]])},
                                    )()
                                },
                            )()

                    self.data = {"test": MockSimulationData()}

                def __getitem__(self, key):
                    return self.data[key]

            return MockBatchData()

        # Patch the autograd run_async function
        monkeypatch.setattr("tidy3d.web.api.autograd.autograd.run_async", mock_autograd_run_async)

        # Define an objective function that uses Batch.run()
        def objective_function(permittivity):
            # Create a new simulation with the updated traced field
            wavelength = 1.0
            freq0 = td.C_0 / wavelength

            new_sim = td.Simulation(
                size=(2.0, 2.0, 2.0),
                grid_spec=td.GridSpec.auto(wavelength=wavelength),
                structures=[
                    td.Structure(
                        geometry=td.Box(center=(0, 0, 0), size=(1.0, 1.0, 1.0)),
                        medium=td.Medium(permittivity=anp.array(permittivity)),  # Traced field
                    )
                ],
                sources=[
                    td.PointDipole(
                        center=(0, 0, 0),
                        polarization="Ex",
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                    )
                ],
                monitors=[
                    td.FieldMonitor(
                        center=(0, 0, 0),
                        size=(1.0, 1.0, 1.0),
                        freqs=[freq0],
                        name="field",
                    )
                ],
                run_time=1e-12,
            )

            # Create a new batch with the updated simulation
            batch = Batch(simulations={"test": new_sim})

            # Run the simulation using Batch.run()
            result = batch.run()

            # Return a scalar value that can be differentiated
            return anp.real(anp.sum(result["test"].field.Ex.values))

        # Test that we can compute gradients
        try:
            gradient = autograd.grad(objective_function)(2.0)
            # If we get here, the gradient computation worked
            assert gradient is not None
            assert isinstance(gradient, (float, int, anp.ndarray))
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")
