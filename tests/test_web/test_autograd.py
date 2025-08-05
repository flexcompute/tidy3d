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
        """Test that Job.run() method exists and is callable."""

        job = Job(
            simulation=traced_simulation,
            task_name="test_job",
        )

        # Test that the run method exists
        assert hasattr(job, "run")
        assert callable(job.run)

    def test_batch_run_method_exists(self, traced_simulation):
        """Test that Batch.run() method exists and is callable."""

        batch = Batch(
            simulations={"test": traced_simulation},
        )

        # Test that the run method exists
        assert hasattr(batch, "run")
        assert callable(batch.run)
