"""Additional workflow coverage for multi-step execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.workflow import (
    HeatChargeWorkflow,
    Step,
    StepInput,
    StepOutput,
    Workflow,
    resolve_workflow,
)
from tidy3d.exceptions import DataError, WebError
from tidy3d.web.api import task_api
from tidy3d.web.api.container import Batch, Job, JobState, WebContainer
from tidy3d.web.api.webapi import upload
from tidy3d.web.core.exceptions import WebError as CoreWebError

from ..utils import FULL_CHARGE, FULL_STEADY_HEAT


def _make_heat_simulation() -> HeatSimulation:
    """Build a deprecated heat-only simulation from the shared HeatCharge fixture."""
    return HeatSimulation(**FULL_STEADY_HEAT.model_dump(exclude={"type"}))


def _file_md5(path: Path) -> str:
    """Return md5 digest of a local file."""
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _add_distributed_generation_with_scalar_coords(
    simulation: td.HeatChargeSimulation,
) -> td.HeatChargeSimulation:
    """Add generated-carrier data shaped like mode-solver-derived notebook output."""
    x = np.linspace(-0.1, 0.1, 3)
    y = np.linspace(-0.1, 0.1, 4)
    z = np.linspace(-0.05, 0.05, 2)
    generation_rate = td.SpatialDataArray(
        np.ones((3, 4, 2)),
        coords={
            "x": x,
            "y": y,
            "z": z,
            "f": 193414489032258.06,
            "mode_index": 0,
        },
        dims=("x", "y", "z"),
    )
    distributed_generation = td.DistributedGeneration.from_rate_um3(generation_rate)

    updated_structures = []
    added_generation = False
    for structure in simulation.structures:
        medium = structure.medium
        if isinstance(medium, td.SemiconductorMedium):
            updated_medium = medium.updated_copy(R=(*tuple(medium.R), distributed_generation))
            updated_structures.append(structure.updated_copy(medium=updated_medium))
            added_generation = True
            continue

        charge = getattr(medium, "charge", None)
        if isinstance(charge, td.SemiconductorMedium):
            updated_charge = charge.updated_copy(R=(*tuple(charge.R), distributed_generation))
            updated_structures.append(
                structure.updated_copy(medium=medium.updated_copy(charge=updated_charge))
            )
            added_generation = True
            continue

        updated_structures.append(structure)

    assert added_generation
    return simulation.updated_copy(structures=tuple(updated_structures))


def test_heat_charge_workflow_structure():
    workflow = HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT)
    assert len(workflow.steps) == 2
    assert workflow.steps[0].name == "mesh"
    assert workflow.steps[1].name == "solve"
    assert workflow.steps[1].inputs[0].upstream_step == "mesh"
    assert workflow.steps[1].inputs[0].upstream_output == "volume_mesh"
    assert workflow.steps[0].get_output("volume_mesh").usage == "dependency"
    assert workflow.steps[0].default_load_output_name == "mesh_data"
    assert workflow.steps[1].default_load_output_name == "simulation_data"
    assert workflow.steps[1].default_load_output.kind == "HeatChargeSimulationData"


def test_heat_simulation_uses_default_two_step_workflow():
    heat_sim = _make_heat_simulation()

    workflow = resolve_workflow(heat_sim)

    assert len(workflow.steps) == 2
    assert workflow.steps[0].name == "mesh"
    assert workflow.steps[1].name == "solve"
    assert isinstance(workflow.steps[0].operation, VolumeMesher)
    assert isinstance(workflow.steps[1].operation, HeatSimulation)
    assert workflow.steps[1].default_load_output.kind == "HeatSimulationData"


def test_top_level_upload_rejects_multistep_simulation():
    with pytest.raises(DataError, match=r"web.run\(\).*web.Job"):
        upload(FULL_STEADY_HEAT, task_name="heat_charge", folder_name="default")


def test_top_level_upload_rejects_deprecated_heat_simulation():
    with pytest.raises(DataError, match=r"web.run\(\).*web.Job"):
        upload(_make_heat_simulation(), task_name="heat", folder_name="default")


def test_parent_task_upload_canonicalizes_heat_charge_simulation(tmp_path):
    simulation = _add_distributed_generation_with_scalar_coords(FULL_CHARGE)

    upload_path = tmp_path / "upload.hdf5"
    simulation.to_file(upload_path)

    mesher_path = tmp_path / "mesher.hdf5"
    child_path = tmp_path / "child.hdf5"
    VolumeMesher(simulation=simulation).to_file(mesher_path)
    child_simulation = VolumeMesher.from_file(mesher_path).simulation
    child_simulation.to_file(child_path)

    assert _file_md5(upload_path) == _file_md5(child_path)
    assert simulation._hash_self() == child_simulation._hash_self()


def test_resolve_workflow_override_is_used():
    custom = Workflow(steps=(Step(name="custom", operation=FULL_STEADY_HEAT),))
    resolved = resolve_workflow(FULL_STEADY_HEAT, workflow=custom)
    assert resolved is custom


def test_workflow_rejects_future_step_dependencies():
    with pytest.raises(ValidationError, match="unknown or future step") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="solve",
                    operation=FULL_STEADY_HEAT,
                    inputs=(StepInput(upstream_step="mesh", upstream_output="volume_mesh"),),
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 0, "inputs", 0, "upstream_step")


def test_workflow_rejects_duplicate_step_names():
    with pytest.raises(ValidationError, match="Duplicate workflow step name") as exc_info:
        Workflow(
            steps=(
                Step(name="dup", operation=FULL_STEADY_HEAT),
                Step(name="dup", operation=FULL_STEADY_HEAT),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 1, "name")


def test_workflow_rejects_empty_output_name():
    with pytest.raises(ValidationError, match="output names cannot be empty") as exc_info:
        Step(
            name="mesh",
            operation=FULL_STEADY_HEAT,
            outputs={"": StepOutput(kind="VolumeMesherData", usage="load")},
        )
    assert exc_info.value.errors()[0]["loc"] == ("outputs", "")


def test_workflow_rejects_unknown_output_dependencies():
    with pytest.raises(ValidationError, match="unknown output") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="mesh",
                    operation=FULL_STEADY_HEAT,
                    outputs={"mesh_data": StepOutput(kind="VolumeMesherData", usage="load")},
                ),
                Step(
                    name="solve",
                    operation=FULL_STEADY_HEAT,
                    inputs=(StepInput(upstream_step="mesh", upstream_output="volume_mesh"),),
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 1, "inputs", 0, "upstream_output")


def test_workflow_rejects_non_dependency_output_as_input():
    with pytest.raises(ValidationError, match="not available as a workflow dependency") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="mesh",
                    operation=FULL_STEADY_HEAT,
                    outputs={"mesh_data": StepOutput(kind="VolumeMesherData", usage="load")},
                ),
                Step(
                    name="solve",
                    operation=FULL_STEADY_HEAT,
                    inputs=(StepInput(upstream_step="mesh", upstream_output="mesh_data"),),
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 1, "inputs", 0, "upstream_output")


def test_web_execution_rejects_unsupported_parent_task_input_topology():
    workflow = Workflow(
        steps=(
            Step(
                name="mesh",
                operation=FULL_STEADY_HEAT,
                outputs={
                    "volume_mesh": StepOutput(kind="volume_mesh", usage="dependency"),
                    "mesh_data": StepOutput(kind="HeatChargeSimulationData", usage="load"),
                },
            ),
            Step(
                name="solve",
                operation=FULL_STEADY_HEAT,
                inputs=(StepInput(upstream_step="mesh", upstream_output="volume_mesh"),),
            ),
        )
    )
    with pytest.raises(
        ValidationError,
        match=r"Heat and HeatCharge workflow dependencies.*'volume_mesh' output",
    ) as exc_info:
        Job(simulation=FULL_STEADY_HEAT, task_name="workflow_job", workflow=workflow)
    assert exc_info.value.errors()[0]["loc"] == (
        "workflow",
        "steps",
        1,
        "inputs",
        0,
        "upstream_output",
    )


def test_workflow_rejects_multiple_loadable_outputs_even_with_default():
    with pytest.raises(ValidationError, match="multiple loadable outputs") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="mesh",
                    operation=FULL_STEADY_HEAT,
                    outputs={
                        "mesh_data": StepOutput(
                            kind="VolumeMesherData",
                            usage="load",
                            default_load=True,
                        ),
                        "mesh_preview": StepOutput(kind="VolumeMesherData", usage="load"),
                    },
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 0, "outputs")


def test_workflow_rejects_parent_task_only_step_outputs():
    with pytest.raises(ValidationError, match="only workflow dependency outputs") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="mesh",
                    operation=FULL_STEADY_HEAT,
                    outputs={"volume_mesh": StepOutput(kind="volume_mesh", usage="dependency")},
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 0, "outputs")


def test_workflow_rejects_multiple_dependency_outputs():
    with pytest.raises(ValidationError, match="multiple workflow dependency outputs") as exc_info:
        Workflow(
            steps=(
                Step(
                    name="mesh",
                    operation=FULL_STEADY_HEAT,
                    outputs={
                        "volume_mesh": StepOutput(kind="volume_mesh", usage="dependency"),
                        "mesh_index": StepOutput(kind="volume_mesh", usage="dependency"),
                        "mesh_data": StepOutput(kind="VolumeMesherData", usage="load"),
                    },
                ),
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("steps", 0, "outputs")


def test_workflow_rejects_unknown_output_kind():
    with pytest.raises(ValidationError):
        StepOutput(kind="mesh_index", usage="dependency")


def test_workflow_rejects_default_load_on_parent_task_output():
    with pytest.raises(ValidationError, match="default_load") as exc_info:
        StepOutput(
            kind="volume_mesh",
            usage="dependency",
            default_load=True,
        )
    assert exc_info.value.errors()[0]["loc"] == ("default_load",)


def test_job_state_roundtrip(tmp_path):
    state = JobState(
        task_ids={"mesh": "mesh-id", "solve": "solve-id"},
        step_statuses={"mesh": "completed", "solve": "running"},
        owned_task_ids={"mesh": False, "solve": True},
        cached_task_ids={"mesh": "mesh-id"},
        current_step_index=1,
    )
    path = tmp_path / "job_state.json"
    state.to_file(path)
    loaded = JobState.from_file(path)
    assert loaded.task_ids == state.task_ids
    assert loaded.step_statuses == state.step_statuses
    assert loaded.owned_task_ids == state.owned_task_ids
    assert loaded.cached_task_ids == state.cached_task_ids
    assert loaded.current_step_index == state.current_step_index


def test_job_state_property_returns_snapshot():
    job = Job(simulation=FULL_STEADY_HEAT, task_name="workflow_job", verbose=False)

    state = job.state
    state.task_ids["mesh"] = "external-mutation"
    state.step_statuses["mesh"] = "completed"

    assert job.task_ids["mesh"] is None
    assert job.state.step_statuses["mesh"] == "pending"


@pytest.fixture
def mock_multistep_task_api(monkeypatch):
    calls = {
        "upload": [],
        "start": [],
        "monitor": [],
        "download": [],
        "load": [],
        "delete": [],
    }

    monkeypatch.setattr(
        WebContainer,
        "_check_folder",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.restore_simulation_if_cached",
        lambda *args, **kwargs: (None, None),
    )

    def fake_upload_task(*args, **kwargs):
        task_id = f"task-{len(calls['upload']) + 1}"
        calls["upload"].append({"task_id": task_id, "kwargs": kwargs})
        return task_id

    def fake_start_task(task_id, **kwargs):
        calls["start"].append({"task_id": task_id, "kwargs": kwargs})

    def fake_monitor_task(task_id, **kwargs):
        calls["monitor"].append({"task_id": task_id, "kwargs": kwargs})

    def fake_load_task(task_id=None, path="simulation_data.hdf5", **kwargs):
        calls["load"].append({"task_id": task_id, "path": str(path), "kwargs": kwargs})
        return {"task_id": task_id, "path": str(path)}

    def fake_download_task(task_id, path, **kwargs):
        calls["download"].append({"task_id": task_id, "path": str(path), "kwargs": kwargs})
        Path(path).touch()

    def fake_delete_task(task_id, **kwargs):
        calls["delete"].append(task_id)

    monkeypatch.setattr("tidy3d.web.api.task_api.upload", fake_upload_task)
    monkeypatch.setattr("tidy3d.web.api.task_api.start", fake_start_task)
    monkeypatch.setattr("tidy3d.web.api.task_api.monitor", fake_monitor_task)
    monkeypatch.setattr("tidy3d.web.api.task_api.download", fake_download_task)
    monkeypatch.setattr("tidy3d.web.api.task_api.load", fake_load_task)
    monkeypatch.setattr("tidy3d.web.api.task_api.delete", fake_delete_task)
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.get_info",
        lambda task_id, **kwargs: SimpleNamespace(status="success"),
    )
    monkeypatch.setattr("tidy3d.web.api.task_api.estimate_cost", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.estimate_cost_info",
        lambda *a, **k: task_api.FlexCreditEstimate(maximum=0.0),
    )
    monkeypatch.setattr("tidy3d.web.api.task_api.real_cost", lambda *a, **k: 0.0)

    return calls


def test_multistep_job_run_chains_parent_tasks(mock_multistep_task_api, tmp_path):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    out_path = tmp_path / "solve.hdf5"
    data = job.run(path=out_path)

    calls = mock_multistep_task_api
    assert len(calls["upload"]) == 2
    assert len(calls["start"]) == 2
    assert len(calls["monitor"]) == 2
    assert len(calls["load"]) == 1
    assert calls["load"][0]["task_id"] == calls["upload"][1]["task_id"]
    assert calls["upload"][0]["kwargs"]["parent_tasks"] is None
    assert calls["upload"][1]["kwargs"]["parent_tasks"] == [calls["upload"][0]["task_id"]]
    assert data["task_id"] == calls["upload"][1]["task_id"]
    assert job.status == "success"


def test_multistep_job_run_prints_upload_estimates_when_verbose(mock_multistep_task_api, tmp_path):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=True,
    )

    job.run(path=tmp_path / "solve.hdf5")

    assert [
        upload_call["kwargs"]["verbose_estimate_cost"]
        for upload_call in mock_multistep_task_api["upload"]
    ] == [True, True]


def test_multistep_job_run_to_file_honors_verbose_estimate_override(
    mock_multistep_task_api, tmp_path
):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    job._run_to_file(path=tmp_path / "solve.hdf5", verbose_estimate_cost=True)

    assert [
        upload_call["kwargs"]["verbose_estimate_cost"]
        for upload_call in mock_multistep_task_api["upload"]
    ] == [True, True]


def test_single_step_cacheable_false_skips_local_cache(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fail_restore(*args, **kwargs):
        raise AssertionError("cacheable=False single-step workflows should not restore cache")

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fail_restore)

    workflow = Workflow(steps=(Step(name="execute", operation=FULL_STEADY_HEAT, cacheable=False),))
    job = Job(
        simulation=FULL_STEADY_HEAT,
        workflow=workflow,
        task_name="uncached_step",
        folder_name="default",
        verbose=False,
    )

    job.run(path=tmp_path / "uncached.hdf5")

    assert mock_multistep_task_api["load"][-1]["kwargs"]["store_in_cache"] is False


def test_multistep_job_download_does_not_load(mock_multistep_task_api, tmp_path):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job._runtime_state.task_ids["mesh"] = "mesh-id"
    job._runtime_state.task_ids["solve"] = "solve-id"
    job._runtime_state.step_statuses["mesh"] = "completed"
    job._runtime_state.step_statuses["solve"] = "completed"
    job._update_current_step_index()

    out_path = tmp_path / "solve.hdf5"
    job.download(out_path)

    assert mock_multistep_task_api["download"] == [
        {
            "task_id": "solve-id",
            "path": str(out_path),
            "kwargs": {"verbose": False, "progress_callback": None},
        }
    ]
    assert mock_multistep_task_api["load"] == []


def test_multistep_job_with_parent_mesh_skips_mesh_step(mock_multistep_task_api, tmp_path):
    mesh_task_id = "existing-mesh-task"
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        parent_tasks=(mesh_task_id,),
        verbose=False,
    )

    assert job.task_ids["mesh"] == mesh_task_id
    assert job._runtime_state.owned_task_ids["mesh"] is False
    assert job._runtime_state.step_statuses["mesh"] == "completed"
    assert job._runtime_state.current_step_index == 1

    solve_data = job.step(path=tmp_path / "solve.hdf5")

    calls = mock_multistep_task_api
    assert len(calls["upload"]) == 1
    assert calls["upload"][0]["kwargs"]["parent_tasks"] == [mesh_task_id]
    assert solve_data["task_id"] == calls["upload"][0]["task_id"]
    assert job.task_ids["solve"] == calls["upload"][0]["task_id"]
    assert job._runtime_state.owned_task_ids["solve"] is True
    assert job.status == "success"


def test_default_heat_charge_rejects_multiple_parent_tasks_before_mesh_upload(
    mock_multistep_task_api, tmp_path
):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        parent_tasks=("mesh-task", "extra-task"),
        verbose=False,
    )

    with pytest.raises(CoreWebError, match="single parent"):
        job.run(path=tmp_path / "solve.hdf5")

    assert mock_multistep_task_api["upload"] == []


def test_parent_mesh_shortcut_does_not_delete_or_bill_parent_mesh(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    billed_task_ids = []

    def _fake_real_cost(task_id, **kwargs):
        billed_task_ids.append(task_id)
        return 1.0

    monkeypatch.setattr("tidy3d.web.api.task_api.real_cost", _fake_real_cost)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        parent_tasks=("existing-mesh-task",),
        verbose=False,
    )

    job.run(path=tmp_path / "solve.hdf5")

    assert job.real_cost(verbose=False) == 1.0
    assert billed_task_ids == [job.task_ids["solve"]]

    job.delete()
    assert mock_multistep_task_api["delete"] == [job.task_ids["solve"]]


def test_multistep_job_real_cost_waits_for_all_owned_step_costs(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")

    def _fake_real_cost(task_id, **kwargs):
        if task_id == job.task_ids["solve"]:
            return None
        return 1.0

    monkeypatch.setattr("tidy3d.web.api.task_api.real_cost", _fake_real_cost)

    assert job.real_cost(verbose=False) is None


def test_multistep_batch_real_cost_waits_for_complete_job_costs(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    batch = Batch(
        simulations={"first": FULL_STEADY_HEAT, "second": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch.run(path_dir=tmp_path)

    solve_task_ids = {job.task_ids["solve"] for job in batch.jobs.values()}

    def _fake_real_cost(task_id, **kwargs):
        if task_id in solve_task_ids:
            return None
        return 1.0

    monkeypatch.setattr("tidy3d.web.api.task_api.real_cost", _fake_real_cost)

    assert batch.real_cost(verbose=False) is None


def test_multistep_estimate_cost_advances_past_cached_step(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    checked_folders = []

    def _fake_restore_if_cached(simulation, path=None, **kwargs):
        if simulation.__class__.__name__ == "VolumeMesher":
            Path(path).touch()
            return path, "cached-mesh-task"
        return None, None

    estimate_calls = []

    def _fake_estimate_cost_info(task_id, **kwargs):
        estimate_calls.append(task_id)
        return task_api.FlexCreditEstimate(maximum=12.0)

    monkeypatch.setattr(
        "tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore_if_cached
    )
    monkeypatch.setattr("tidy3d.web.api.task_api.estimate_cost_info", _fake_estimate_cost_info)
    monkeypatch.setattr(
        WebContainer,
        "_check_folder",
        staticmethod(lambda folder_name: checked_folders.append(folder_name)),
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    assert job.estimate_cost(verbose=False) == 12.0
    assert len(mock_multistep_task_api["upload"]) == 1
    assert mock_multistep_task_api["upload"][0]["kwargs"]["parent_tasks"] == ["cached-mesh-task"]
    assert checked_folders == ["default"]
    assert estimate_calls == [mock_multistep_task_api["upload"][0]["task_id"]]
    assert job._runtime_state.owned_task_ids["mesh"] is False


def test_multistep_estimate_cost_refreshes_uploaded_step_before_estimate(
    mock_multistep_task_api, monkeypatch
):
    def _fake_get_info(task_id, **kwargs):
        if task_id == "mesh-id":
            return SimpleNamespace(status="success")
        raise AssertionError(f"Unexpected status refresh for {task_id}.")

    estimate_calls = []

    def _fake_estimate_cost_info(task_id, **kwargs):
        estimate_calls.append(task_id)
        return task_api.FlexCreditEstimate(maximum=9.0)

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)
    monkeypatch.setattr("tidy3d.web.api.task_api.estimate_cost_info", _fake_estimate_cost_info)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    with job._state_lock:
        job._runtime_state.task_ids["mesh"] = "mesh-id"
        job._runtime_state.owned_task_ids["mesh"] = True
        job._runtime_state.step_statuses["mesh"] = "running"
        job._update_current_step_index()

    assert job.estimate_cost(verbose=False) == 9.0
    assert job._runtime_state.step_statuses["mesh"] == "completed"
    assert len(mock_multistep_task_api["upload"]) == 1
    assert mock_multistep_task_api["upload"][0]["kwargs"]["parent_tasks"] == ["mesh-id"]
    assert estimate_calls == [mock_multistep_task_api["upload"][0]["task_id"]]


def test_multistep_job_estimate_cost_reports_mesh_step_guidance(
    mock_multistep_task_api, monkeypatch
):
    log_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.get_logging_console",
        lambda: SimpleNamespace(log=log_messages.append),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.estimate_cost_info",
        lambda *a, **k: task_api.FlexCreditEstimate(maximum=1.0),
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    assert job.estimate_cost(verbose=True) == 1.0
    assert log_messages == [
        "The FlexCredit estimate shown above is for the next workflow step 'mesh' only.",
        "This is the mesh step. Run it first with 'Job.step()'; after it completes, call "
        "'Job.estimate_cost()' again for the solver estimate.",
    ]


def test_multistep_dependency_cache_without_task_id_is_bypassed(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    restored_paths = []

    def _fake_restore_if_cached(simulation, path=None, **kwargs):
        if simulation.__class__.__name__ == "VolumeMesher":
            Path(path).touch()
            restored_paths.append(Path(path))
            return path, None
        return None, None

    monkeypatch.setattr(
        "tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore_if_cached
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    data = job.run(path=tmp_path / "solve.hdf5")

    uploads = mock_multistep_task_api["upload"]
    assert len(uploads) == 2
    assert uploads[0]["kwargs"]["parent_tasks"] is None
    assert uploads[1]["kwargs"]["parent_tasks"] == [uploads[0]["task_id"]]
    assert data["task_id"] == uploads[1]["task_id"]
    assert job.task_ids["mesh"] == uploads[0]["task_id"]
    assert job._runtime_state.owned_task_ids["mesh"] is True
    assert restored_paths
    assert not restored_paths[0].exists()


@pytest.mark.parametrize(
    "missing_task_error",
    [CoreWebError("not found"), ValueError("Task not found.")],
    ids=["core-web-error", "task-not-found-value-error"],
)
def test_multistep_dependency_cache_with_unavailable_task_id_is_bypassed(
    mock_multistep_task_api, monkeypatch, tmp_path, missing_task_error
):
    restored_paths = []

    def _fake_restore_if_cached(simulation, path=None, **kwargs):
        if isinstance(simulation, VolumeMesher):
            Path(path).touch()
            restored_paths.append(Path(path))
            return path, "missing-cached-mesh"
        return None, None

    def _fake_get_info(task_id, **kwargs):
        if task_id == "missing-cached-mesh":
            raise missing_task_error
        return SimpleNamespace(status="success")

    monkeypatch.setattr(
        "tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore_if_cached
    )
    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    data = job.run(path=tmp_path / "solve.hdf5")

    uploads = mock_multistep_task_api["upload"]
    assert len(uploads) == 2
    assert uploads[0]["kwargs"]["parent_tasks"] is None
    assert uploads[1]["kwargs"]["parent_tasks"] == [uploads[0]["task_id"]]
    assert data["task_id"] == uploads[1]["task_id"]
    assert job.task_ids["mesh"] == uploads[0]["task_id"]
    assert job._runtime_state.owned_task_ids["mesh"] is True
    assert "mesh" not in job._runtime_state.cached_task_ids
    assert restored_paths
    assert not restored_paths[0].exists()


def test_multistep_resume_running_step_does_not_start_again(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    refreshed_statuses = iter(("running", "success"))

    def _fake_get_info(task_id, **kwargs):
        assert task_id == "mesh-id"
        return SimpleNamespace(status=next(refreshed_statuses))

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job._runtime_state.task_ids["mesh"] = "mesh-id"
    job._runtime_state.owned_task_ids["mesh"] = True
    job._runtime_state.step_statuses["mesh"] = "draft"
    job._update_current_step_index()

    data = job.step(path=tmp_path / "mesh.hdf5")

    assert mock_multistep_task_api["upload"] == []
    assert mock_multistep_task_api["start"] == []
    assert mock_multistep_task_api["monitor"] == [
        {"task_id": "mesh-id", "kwargs": {"verbose": False}}
    ]
    assert data["task_id"] == "mesh-id"
    assert job._runtime_state.step_statuses["mesh"] == "completed"


def test_multistep_nonfinal_diverged_step_does_not_run_downstream(
    mock_multistep_task_api, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.get_info",
        lambda task_id, **kwargs: SimpleNamespace(
            status="diverged" if task_id == "task-1" else "success"
        ),
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    with pytest.raises(DataError, match=r"Workflow step 'mesh'.*diverged"):
        job.run(path=tmp_path / "solve.hdf5")

    assert job._runtime_state.step_statuses["mesh"] == "diverged"
    assert job.task_ids["solve"] is None
    assert len(mock_multistep_task_api["upload"]) == 1
    assert len(mock_multistep_task_api["start"]) == 1
    assert len(mock_multistep_task_api["monitor"]) == 1


def test_multistep_status_reports_next_pending_after_nonfinal_success(monkeypatch):
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.get_info",
        lambda task_id, **kwargs: SimpleNamespace(status="success"),
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job._runtime_state.task_ids["mesh"] = "mesh-task"
    job._runtime_state.step_statuses["mesh"] = "running"

    assert job.status == "pending"
    assert job._runtime_state.step_statuses["mesh"] == "completed"
    assert job._runtime_state.current_step_index == 1


@pytest.mark.parametrize("status", ["completed", "visualize", "processed", "postprocess_success"])
def test_multistep_terminal_success_status_normalizes(monkeypatch, status):
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.get_info",
        lambda task_id, **kwargs: SimpleNamespace(status=status),
    )

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job._runtime_state.task_ids["mesh"] = "mesh-task"
    job._runtime_state.step_statuses["mesh"] = "completed"
    job._runtime_state.task_ids["solve"] = "solve-task"
    job._runtime_state.step_statuses["solve"] = "running"

    assert job.status == "success"
    assert job._runtime_state.step_statuses["solve"] == status


def test_explicit_multistep_workflow_rejects_parent_tasks(mock_multistep_task_api, tmp_path):
    custom = HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT)
    job = Job(
        simulation=FULL_STEADY_HEAT,
        workflow=custom,
        task_name="workflow_job",
        folder_name="default",
        parent_tasks=("existing-mesh-task",),
        verbose=False,
    )

    with pytest.raises(DataError, match="Custom workflow jobs"):
        job.step(path=tmp_path / "mesh.hdf5")

    assert mock_multistep_task_api["upload"] == []
    assert job.task_ids["mesh"] is None


def test_multistep_job_step_runs_in_order(mock_multistep_task_api, tmp_path):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    mesh_data = job.step(path=tmp_path / "mesh.hdf5")
    assert mesh_data["task_id"] == job.task_ids["mesh"]
    assert job._runtime_state.step_statuses["mesh"] == "completed"
    assert job._runtime_state.step_statuses["solve"] == "pending"

    solve_data = job.step(path=tmp_path / "solve.hdf5")
    assert solve_data["task_id"] == job.task_ids["solve"]
    assert job._runtime_state.step_statuses["solve"] == "success"
    assert job.status == "success"


def test_multistep_job_step_default_paths_are_step_specific(mock_multistep_task_api):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    job.step()
    job.step()

    paths = [Path(call["path"]).name for call in mock_multistep_task_api["load"]]
    assert paths[0].startswith("mesh_")
    assert paths[1].startswith("solve_")
    assert paths[0] != paths[1]
    assert not Path(job._default_output_path()).name.startswith("solve_")


def test_multistep_job_run_default_path_uses_final_output_path(mock_multistep_task_api):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )

    job.run()

    assert len(mock_multistep_task_api["load"]) == 1
    path = Path(mock_multistep_task_api["load"][0]["path"])
    assert path.name == Path(job._default_output_path()).name
    assert not path.name.startswith("solve_")


def test_multistep_job_delete_removes_all_step_tasks(mock_multistep_task_api, tmp_path):
    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")
    job.delete()

    calls = mock_multistep_task_api
    expected = {task_id for task_id in job.task_ids.values() if task_id is not None}
    assert set(calls["delete"]) == expected


def test_multistep_job_serialization_preserves_state(tmp_path):
    job = Job(simulation=FULL_STEADY_HEAT, task_name="workflow_job", verbose=False)
    job._runtime_state.task_ids["mesh"] = "mesh-id"
    job._runtime_state.task_ids["solve"] = "solve-id"
    job._runtime_state.owned_task_ids["mesh"] = False
    job._runtime_state.owned_task_ids["solve"] = True
    job._runtime_state.step_statuses["mesh"] = "completed"
    job._runtime_state.step_statuses["solve"] = "running"
    job._update_current_step_index()

    path = tmp_path / "job.json"
    job.to_file(path)
    loaded = Job.from_file(path)

    assert loaded.task_ids["mesh"] == "mesh-id"
    assert loaded.task_ids["solve"] == "solve-id"
    assert loaded.state.owned_task_ids["mesh"] is False
    assert loaded.state.owned_task_ids["solve"] is True
    assert loaded.state.step_statuses["mesh"] == "completed"
    assert loaded.state.step_statuses["solve"] == "running"
    assert loaded.state.current_step_index == 1


def test_multistep_job_status_preserves_diverged_final_step(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_get_info(task_id, **kwargs):
        return SimpleNamespace(status="diverged" if task_id == "task-2" else "success")

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")

    assert job._runtime_state.step_statuses["solve"] == "diverged"
    assert job.status == "diverged"


def test_multistep_batch_run_preserves_diverged_terminal_status(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_get_info(task_id, **kwargs):
        return SimpleNamespace(status="diverged" if task_id == "task-2" else "success")

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch.run(path_dir=tmp_path)

    assert batch._terminal_status_by_task["heat"] == "diverged"


def test_multistep_batch_run_tolerates_diverged_intermediate_step(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_get_info(task_id, **kwargs):
        return SimpleNamespace(status="diverged" if task_id == "task-1" else "success")

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    batch = Batch(
        simulations={"bad": FULL_STEADY_HEAT, "good": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    data = batch.run(path_dir=tmp_path)

    assert batch._terminal_status_by_task["bad"] == "diverged"
    assert batch._terminal_status_by_task["good"] == "success"
    assert "bad" not in data.task_ids
    assert data.task_ids["good"] == "task-3"


def test_multistep_batch_run_checkpoints_after_step_upload(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    batch_path = tmp_path / "batch.hdf5"
    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )

    def _fake_start(task_id, **kwargs):
        if task_id == "task-1":
            assert batch_path.exists()
            checkpointed = Batch.from_file(batch_path)
            assert checkpointed.jobs["heat"].task_ids["mesh"] == "task-1"
            assert checkpointed.jobs["heat"].state.step_statuses["mesh"] == "draft"
        mock_multistep_task_api["start"].append({"task_id": task_id, "kwargs": kwargs})

    monkeypatch.setattr("tidy3d.web.api.task_api.start", _fake_start)

    batch.run(path_dir=tmp_path)


def test_completed_multistep_get_run_info_uses_final_step(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    run_info_calls = []

    def _fake_get_run_info(task_id):
        run_info_calls.append(task_id)
        return (100, 0.0)

    monkeypatch.setattr("tidy3d.web.api.task_api.get_run_info", _fake_get_run_info)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")

    assert job.get_run_info() == (100, 0.0)
    assert run_info_calls == [job.task_ids["solve"]]

    batch_path = tmp_path / "batch_run_info"
    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch.run(path_dir=batch_path)

    assert batch.get_run_info() == {"heat": (100, 0.0)}
    assert run_info_calls[-1] == batch.jobs["heat"].task_ids["solve"]


def test_multistep_cache_only_final_step_survives_job_serialization(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    restore_calls = []

    def _fake_restore(simulation, path, **kwargs):
        restore_calls.append(simulation)
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")
    assert job.task_ids["solve"] is None

    job_path = tmp_path / "job.json"
    job.to_file(job_path)
    loaded = Job.from_file(job_path)
    loaded_path = tmp_path / "loaded.hdf5"
    loaded.load(path=loaded_path)

    assert loaded_path.read_text() == "cached final"
    assert loaded.task_ids["solve"] is None
    assert sum(not isinstance(call, VolumeMesher) for call in restore_calls) == 2


def test_multistep_cache_only_final_step_step_error_suggests_load_or_run(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")

    job_path = tmp_path / "job.json"
    job.to_file(job_path)
    loaded = Job.from_file(job_path)
    loaded.load(path=tmp_path / "loaded.hdf5")
    upload_count = len(mock_multistep_task_api["upload"])

    with pytest.raises(DataError) as exc_info:
        loaded.step(path=tmp_path / "unused.hdf5")

    message = str(exc_info.value)
    assert "Job.step() only advances an incomplete workflow one step" in message
    assert "Job.load()" in message
    assert "Job.run()" in message
    assert "local cache" in message
    assert len(mock_multistep_task_api["upload"]) == upload_count


def test_multistep_cache_only_final_step_cache_miss_reruns_after_job_serialization(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    cache_available = True

    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher) or not cache_available:
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")
    assert job.task_ids["solve"] is None

    job_path = tmp_path / "job.json"
    job.to_file(job_path)
    loaded = Job.from_file(job_path)
    upload_count = len(mock_multistep_task_api["upload"])
    cache_available = False

    data = loaded.run(path=tmp_path / "rerun.hdf5")

    assert len(mock_multistep_task_api["upload"]) == upload_count + 1
    assert mock_multistep_task_api["upload"][-1]["kwargs"]["parent_tasks"] == [job.task_ids["mesh"]]
    assert data["task_id"] == mock_multistep_task_api["upload"][-1]["task_id"]
    assert loaded.task_ids["solve"] == mock_multistep_task_api["upload"][-1]["task_id"]
    assert "solve" not in loaded.state.cached_task_ids


def test_multistep_cache_only_final_step_status_reopens_after_cache_eviction(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    cache_available = True

    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher) or not cache_available:
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")
    assert job.status == "success"

    job_path = tmp_path / "job.json"
    job.to_file(job_path)
    loaded = Job.from_file(job_path)
    cache_available = False

    assert loaded.status == "pending"
    assert loaded.state.step_statuses["solve"] == "pending"
    assert "solve" not in loaded.state.cached_task_ids


def test_multistep_cached_final_task_id_uses_local_cache_after_job_serialization(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    restore_calls = []

    def _fake_restore(simulation, path, **kwargs):
        restore_calls.append(simulation)
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final with id")
        return path, "cached-solve-task"

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    job = Job(
        simulation=FULL_STEADY_HEAT,
        task_name="workflow_job",
        folder_name="default",
        verbose=False,
    )
    job.run(path=tmp_path / "solve.hdf5")
    assert job.task_ids["solve"] == "cached-solve-task"
    assert job._runtime_state.cached_task_ids["solve"] == "cached-solve-task"

    job_path = tmp_path / "job.json"
    job.to_file(job_path)
    loaded = Job.from_file(job_path)
    loaded_path = tmp_path / "loaded.hdf5"
    loaded.load(path=loaded_path)

    assert loaded_path.read_text() == "cached final with id"
    assert loaded.task_ids["solve"] == "cached-solve-task"
    assert loaded.state.cached_task_ids["solve"] == "cached-solve-task"
    assert mock_multistep_task_api["load"][-1]["task_id"] is None
    assert sum(not isinstance(call, VolumeMesher) for call in restore_calls) == 2


def test_multistep_cache_only_final_step_survives_batch_serialization(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch_data = batch.run(path_dir=tmp_path)
    assert batch_data.cached_tasks == {"heat": True}
    assert batch_data.task_ids["heat"] is None
    final_path = Path(batch_data.task_paths["heat"])
    final_path.unlink()

    loaded = Batch.from_file(tmp_path / "batch.hdf5")
    loaded.download(path_dir=tmp_path)

    assert final_path.read_text() == "cached final"

    reloaded_data = loaded.load(path_dir=tmp_path, skip_download=True)
    assert reloaded_data.cached_tasks == {"heat": True}
    assert reloaded_data.task_ids["heat"] is None
    assert reloaded_data.load_sim_data("heat")["task_id"] is None


def test_multistep_batch_load_uses_existing_fallback_after_cache_eviction(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    cache_available = True

    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher) or not cache_available:
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch_data = batch.run(path_dir=tmp_path)
    final_path = Path(batch_data.task_paths["heat"])
    assert final_path.exists()

    loaded = Batch.from_file(tmp_path / "batch.hdf5")
    upload_count = len(mock_multistep_task_api["upload"])
    cache_available = False

    reloaded_data = loaded.load(path_dir=tmp_path, skip_download=True)

    assert len(mock_multistep_task_api["upload"]) == upload_count
    assert reloaded_data.cached_tasks == {"heat": True}
    assert reloaded_data.task_ids["heat"] is None
    assert Path(reloaded_data.task_paths["heat"]) == final_path
    assert loaded.jobs["heat"].state.step_statuses["solve"] == "completed"
    assert reloaded_data.load_sim_data("heat")["task_id"] is None


def test_multistep_batch_result_id_reopens_missing_cache_only_final_step(
    monkeypatch, mock_multistep_task_api
):
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.restore_simulation_if_cached",
        lambda *args, **kwargs: (None, None),
    )

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    job = batch.jobs["heat"]
    with job._state_lock:
        job._runtime_state.task_ids["mesh"] = "mesh-id"
        job._runtime_state.owned_task_ids["mesh"] = True
        job._runtime_state.step_statuses["mesh"] = "completed"
        job._runtime_state.task_ids["solve"] = None
        job._runtime_state.owned_task_ids["solve"] = False
        job._runtime_state.cached_task_ids["solve"] = None
        job._runtime_state.step_statuses["solve"] = "completed"
        job._update_current_step_index()

    assert batch._known_multi_step_result_task_id("heat", job) is None
    assert job.state.step_statuses["solve"] == "pending"
    assert "solve" not in job.state.cached_task_ids


def test_multistep_cache_only_final_step_cache_miss_reruns_after_batch_serialization(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    cache_available = True

    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher) or not cache_available:
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch_data = batch.run(path_dir=tmp_path)
    assert batch_data.cached_tasks == {"heat": True}
    assert batch_data.task_ids["heat"] is None

    loaded = Batch.from_file(tmp_path / "batch.hdf5")
    upload_count = len(mock_multistep_task_api["upload"])
    cache_available = False

    rerun_data = loaded.run(path_dir=tmp_path)

    assert len(mock_multistep_task_api["upload"]) == upload_count + 1
    assert mock_multistep_task_api["upload"][-1]["kwargs"]["parent_tasks"] == [
        batch.jobs["heat"].task_ids["mesh"]
    ]
    assert rerun_data.cached_tasks == {"heat": False}
    assert rerun_data.task_ids["heat"] == mock_multistep_task_api["upload"][-1]["task_id"]


def test_multistep_batch_download_respects_replace_existing_for_cached_stash(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch_data = batch.run(path_dir=tmp_path)
    final_path = Path(batch_data.task_paths["heat"])
    final_path.write_text("existing")

    batch.download(path_dir=tmp_path, replace_existing=False)
    assert final_path.read_text() == "existing"

    batch.download(path_dir=tmp_path, replace_existing=True)
    assert final_path.read_text() == "cached final"


def test_multistep_batch_run_uses_download_only_path(mock_multistep_task_api, tmp_path):
    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch_data = batch.run(path_dir=tmp_path)
    assert "heat" in batch_data.task_paths
    assert batch_data.task_ids["heat"] == batch.jobs["heat"].task_ids["solve"]
    assert batch_data.cached_tasks == {"heat": False}
    assert Path(batch_data.task_paths["heat"]).name == f"{batch_data.task_ids['heat']}.hdf5"
    assert mock_multistep_task_api["load"] == []
    assert len(mock_multistep_task_api["download"]) == 1
    assert mock_multistep_task_api["download"][0]["task_id"] == batch.jobs["heat"].task_ids["solve"]


def test_uniform_multistep_batch_run_orchestrates_by_workflow_steps(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _unexpected_run_to_file(self, *args, **kwargs):
        raise AssertionError("uniform workflow batches should use rolling workflow orchestration")

    def _unexpected_upload_jobs(self, *args, **kwargs):
        raise AssertionError("rolling workflow scheduler should not nest batch upload progress")

    monkeypatch.setattr(Job, "_run_to_file", _unexpected_run_to_file)
    monkeypatch.setattr(Batch, "_upload_jobs", _unexpected_upload_jobs)

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": _make_heat_simulation()},
        folder_name="default",
        verbose=True,
        num_workers=2,
    )

    data = batch.run(path_dir=tmp_path, priority=4)

    uploads = mock_multistep_task_api["upload"]
    assert len(uploads) == 4
    assert {call["kwargs"]["task_name"] for call in uploads[:2]} == {"a_mesh", "b_mesh"}
    assert {call["kwargs"]["task_name"] for call in uploads[2:]} == {"a_solve", "b_solve"}
    assert all(call["kwargs"]["verbose"] is False for call in uploads)
    assert all(call["kwargs"]["verbose_estimate_cost"] is True for call in uploads)
    assert all(call["kwargs"]["parent_tasks"] is None for call in uploads[:2])
    assert {
        call["kwargs"]["task_name"]: call["kwargs"]["parent_tasks"] for call in uploads[2:]
    } == {
        "a_solve": [batch.jobs["a"].task_ids["mesh"]],
        "b_solve": [batch.jobs["b"].task_ids["mesh"]],
    }

    mesh_task_ids = {batch.jobs[name].task_ids["mesh"] for name in ("a", "b")}
    solve_task_ids = {batch.jobs[name].task_ids["solve"] for name in ("a", "b")}
    starts = mock_multistep_task_api["start"]
    assert {call["task_id"] for call in starts[:2]} == mesh_task_ids
    assert {call["task_id"] for call in starts[2:]} == solve_task_ids
    assert all(call["kwargs"]["priority"] == 4 for call in starts)
    assert mock_multistep_task_api["monitor"] == []

    assert data.task_ids == {
        "a": batch.jobs["a"].task_ids["solve"],
        "b": batch.jobs["b"].task_ids["solve"],
    }
    assert mock_multistep_task_api["load"] == []
    assert {call["task_id"] for call in mock_multistep_task_api["download"]} == solve_task_ids


def test_uniform_multistep_batch_run_supports_explicit_default_workflows(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _unexpected_run_to_file(self, *args, **kwargs):
        raise AssertionError("explicit default workflow batches should use rolling orchestration")

    monkeypatch.setattr(Job, "_run_to_file", _unexpected_run_to_file)

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=2,
    )
    batch._cached_properties = {
        "jobs": {
            "a": Job(
                simulation=FULL_STEADY_HEAT,
                workflow=HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT),
                task_name="a",
                folder_name="default",
                verbose=False,
            ),
            "b": Job(
                simulation=FULL_STEADY_HEAT,
                workflow=HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT),
                task_name="b",
                folder_name="default",
                verbose=False,
            ),
        }
    }

    batch.run(path_dir=tmp_path)

    assert [call["kwargs"]["task_name"] for call in mock_multistep_task_api["upload"]] == [
        "a_mesh",
        "b_mesh",
        "a_solve",
        "b_solve",
    ]


def test_uniform_multistep_batch_run_respects_num_workers(mock_multistep_task_api, tmp_path):
    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )

    batch.run(path_dir=tmp_path)

    assert [call["kwargs"]["task_name"] for call in mock_multistep_task_api["upload"]] == [
        "a_mesh",
        "a_solve",
        "b_mesh",
        "b_solve",
    ]


def test_uniform_multistep_batch_run_rolls_solver_after_each_mesh(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    b_mesh_success_seen = False
    b_mesh_polls = 0

    def _fake_get_info(task_id, **kwargs):
        nonlocal b_mesh_polls, b_mesh_success_seen
        if task_id == "task-2":
            b_mesh_polls += 1
            if b_mesh_polls < 4:
                return SimpleNamespace(status="running")
            b_mesh_success_seen = True
            return SimpleNamespace(status="success")
        if task_id == "task-3":
            assert not b_mesh_success_seen
        return SimpleNamespace(status="success")

    monkeypatch.setattr("tidy3d.web.api.task_api.get_info", _fake_get_info)

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=2,
    )

    batch.run(path_dir=tmp_path)

    assert [call["kwargs"]["task_name"] for call in mock_multistep_task_api["upload"]] == [
        "a_mesh",
        "b_mesh",
        "a_solve",
        "b_solve",
    ]
    assert b_mesh_polls >= 4


def test_uniform_multistep_batch_step_checkpoints_and_resumes(mock_multistep_task_api, tmp_path):
    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=True,
        num_workers=1,
    )

    first_step_result = batch.step(path_dir=tmp_path)

    assert first_step_result is None
    assert (tmp_path / "batch.hdf5").exists()
    assert {batch.jobs[name].state.step_statuses["mesh"] for name in ("a", "b")} == {"completed"}
    assert {batch.jobs[name].task_ids["solve"] for name in ("a", "b")} == {None}
    assert mock_multistep_task_api["download"] == []

    loaded = Batch.from_file(tmp_path / "batch.hdf5")
    second_step_result = loaded.step(path_dir=tmp_path)

    assert second_step_result.task_ids == {
        "a": loaded.jobs["a"].task_ids["solve"],
        "b": loaded.jobs["b"].task_ids["solve"],
    }
    assert {
        call["kwargs"]["task_name"]: call["kwargs"]["parent_tasks"]
        for call in mock_multistep_task_api["upload"][2:]
    } == {
        "a_solve": [loaded.jobs["a"].task_ids["mesh"]],
        "b_solve": [loaded.jobs["b"].task_ids["mesh"]],
    }
    assert {call["task_id"] for call in mock_multistep_task_api["download"]} == {
        loaded.jobs["a"].task_ids["solve"],
        loaded.jobs["b"].task_ids["solve"],
    }
    assert all(
        upload_call["kwargs"]["verbose_estimate_cost"] is True
        for upload_call in mock_multistep_task_api["upload"]
    )


def test_uniform_multistep_batch_step_complete_error_suggests_load_or_run(
    mock_multistep_task_api, tmp_path
):
    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch.run(path_dir=tmp_path)
    upload_count = len(mock_multistep_task_api["upload"])

    with pytest.raises(DataError) as exc_info:
        batch.step(path_dir=tmp_path)

    message = str(exc_info.value)
    assert "Batch.step() only advances an incomplete batch workflow one step" in message
    assert "Batch.load()" in message
    assert "Batch.run()" in message
    assert "local cache" in message
    assert len(mock_multistep_task_api["upload"]) == upload_count


def test_uniform_multistep_batch_step_cache_restored_complete_error_suggests_load_or_run(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    def _fake_restore(simulation, path, **kwargs):
        if isinstance(simulation, VolumeMesher):
            return None, None
        Path(path).write_text("cached final")
        return path, None

    monkeypatch.setattr("tidy3d.web.api.task_api.restore_simulation_if_cached", _fake_restore)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch.run(path_dir=tmp_path)
    loaded = Batch.from_file(tmp_path / "batch.hdf5")
    upload_count = len(mock_multistep_task_api["upload"])

    with pytest.raises(DataError) as exc_info:
        loaded.step(path_dir=tmp_path)

    message = str(exc_info.value)
    assert "Batch.step() only advances an incomplete batch workflow one step" in message
    assert "Batch.load()" in message
    assert "Batch.run()" in message
    assert "local cache" in message
    assert len(mock_multistep_task_api["upload"]) == upload_count


def test_uniform_multistep_batch_estimate_cost_reports_mesh_frontier(
    monkeypatch, mock_multistep_task_api
):
    log_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.workflow_batch.get_logging_console",
        lambda: SimpleNamespace(log=log_messages.append),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.estimate_cost_info",
        lambda *a, **k: task_api.FlexCreditEstimate(maximum=2.5),
    )

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )

    assert batch.estimate_cost(verbose=True) == 5.0
    assert [call["kwargs"]["task_name"] for call in mock_multistep_task_api["upload"]] == [
        "a_mesh",
        "b_mesh",
    ]
    assert mock_multistep_task_api["start"] == []
    assert log_messages == [
        "Maximum FlexCredit cost: 5.000 for the next workflow step 'mesh' across the batch.",
        "This estimates the mesh step only. Run the mesh step first with 'Batch.step()'; "
        "after it completes, call 'Batch.estimate_cost()' again for the solver estimate.",
    ]


def test_uniform_multistep_batch_estimate_cost_reports_solver_frontier(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    log_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.workflow_batch.get_logging_console",
        lambda: SimpleNamespace(log=log_messages.append),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.estimate_cost_info",
        lambda *a, **k: task_api.FlexCreditEstimate(maximum=7.0),
    )

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch.step(path_dir=tmp_path)

    assert batch.estimate_cost(verbose=True) == 14.0
    assert [call["kwargs"]["task_name"] for call in mock_multistep_task_api["upload"][2:]] == [
        "a_solve",
        "b_solve",
    ]
    assert log_messages == [
        "Maximum FlexCredit cost: 14.000 for the next workflow step 'solve' across the batch.",
        "All jobs are at the final solver step, so this is the estimated solver cost for the batch.",
    ]


def test_uniform_multistep_batch_estimate_cost_reports_typical_solver_frontier(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    log_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.workflow_batch.get_logging_console",
        lambda: SimpleNamespace(log=log_messages.append),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.estimate_cost_info",
        lambda *a, **k: task_api.FlexCreditEstimate(maximum=7.0, typical=2.0),
    )

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch.step(path_dir=tmp_path)

    assert batch.estimate_cost(verbose=True) == 14.0
    assert log_messages == [
        "Estimated typical FlexCredit cost: 4.000 for the next workflow step 'solve' across the batch.",
        "Maximum FlexCredit cost: 14.000 for the next workflow step 'solve' across the batch.",
        "All jobs are at the final solver step, so this is the estimated solver cost for the batch.",
    ]


def test_uniform_multistep_batch_estimate_cost_includes_heat_only_solver_in_typical_total(
    monkeypatch, mock_multistep_task_api, tmp_path
):
    log_messages = []
    estimate_calls = []
    monkeypatch.setattr(
        "tidy3d.web.api.workflow_batch.get_logging_console",
        lambda: SimpleNamespace(log=log_messages.append),
    )

    def estimate_cost_info(task_id, **kwargs):
        estimate_calls.append({"task_id": task_id, "kwargs": kwargs})
        if kwargs["is_final_billed_cost"]:
            return task_api.FlexCreditEstimate(
                maximum=5.0,
                task_type="HEAT_CHARGE",
                is_final_billed_cost=True,
            )
        return task_api.FlexCreditEstimate(
            maximum=7.0,
            typical=2.0,
            task_type="HEAT_CHARGE",
            is_final_billed_cost=False,
            typical_cost_kind=task_api._TYPICAL_COST_KIND_CHARGE_SOLVER_ITERATIONS,
        )

    monkeypatch.setattr("tidy3d.web.api.task_api.estimate_cost_info", estimate_cost_info)

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT, "charge": FULL_CHARGE},
        folder_name="default",
        verbose=False,
    )
    batch.step(path_dir=tmp_path)

    assert batch.estimate_cost(verbose=True) == 12.0
    assert [call["kwargs"]["is_final_billed_cost"] for call in estimate_calls[-2:]] == [
        True,
        False,
    ]
    assert log_messages == [
        "Estimated typical FlexCredit cost: 7.000 for the next workflow step 'solve' across the batch.",
        "Maximum FlexCredit cost: 12.000 for the next workflow step 'solve' across the batch.",
        "For charge simulations, the billed cost depends on the number of solver iterations "
        "required for convergence.",
        "All jobs are at the final solver step, so this is the estimated solver cost for the batch.",
    ]


def test_uniform_multistep_batch_estimate_cost_rejects_partial_frontier(
    mock_multistep_task_api,
):
    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    job = batch.jobs["a"]
    with job._state_lock:
        job._runtime_state.task_ids["mesh"] = "mesh-a"
        job._runtime_state.owned_task_ids["mesh"] = True
        job._runtime_state.step_statuses["mesh"] = "completed"
        job._update_current_step_index()

    with pytest.raises(DataError, match="same next workflow step"):
        batch.estimate_cost(verbose=False)


def test_mixed_batch_run_single_step_uses_download_only_path(mock_multistep_task_api, tmp_path):
    single_step_workflow = Workflow(
        steps=(Step(name="execute", operation=FULL_STEADY_HEAT),),
    )
    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT, "single": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
        num_workers=1,
    )
    batch._cached_properties = {
        "jobs": {
            "heat": Job(
                simulation=FULL_STEADY_HEAT,
                task_name="heat",
                folder_name="default",
                verbose=False,
            ),
            "single": Job(
                simulation=FULL_STEADY_HEAT,
                workflow=single_step_workflow,
                task_name="single",
                folder_name="default",
                verbose=False,
            ),
        }
    }

    batch_data = batch.run(path_dir=tmp_path)

    assert set(batch_data.task_paths) == {"heat", "single"}
    assert mock_multistep_task_api["load"] == []
    assert len(mock_multistep_task_api["download"]) == 2
    assert {call["task_id"] for call in mock_multistep_task_api["download"]} == {
        batch.jobs["heat"].task_ids["solve"],
        batch.jobs["single"].task_ids["execute"],
    }


def test_mixed_batch_run_keeps_single_step_batch_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(Batch, "to_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(WebContainer, "_check_folder", staticmethod(lambda *args, **kwargs: None))
    warning_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg, **kwargs: warning_messages.append(msg)
    )
    events = []

    class SingleStepJob:
        is_multi_step = False
        load_if_cached = False
        task_id_cached = None
        simulation = FULL_STEADY_HEAT
        task_name = "single"

        def __init__(self):
            self._task_id = None

        def upload(self):
            events.append("single_upload")
            self._task_id = "single-task-id"

        def start(self, **kwargs):
            events.append(("single_start", kwargs))

        def get_info(self):
            return SimpleNamespace(status="success")

        def estimate_cost(self, **kwargs):
            events.append(("single_estimate", kwargs))
            return 1.0

        @property
        def task_id(self):
            return self._task_id

        def download(self, path):
            events.append(("single_download", str(path)))
            Path(path).write_text("single")

        def _run_to_file(self, *args, **kwargs):
            raise AssertionError("single-step jobs should use the batch pipeline")

    class MultiStepJob:
        is_multi_step = True
        load_if_cached = False
        simulation = FULL_STEADY_HEAT
        steps = HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT).steps
        task_ids = {}
        _step_cached_task_ids = {}
        _step_stash_paths = {}

        @property
        def status(self):
            return "success"

        def _run_to_file(self, path, **kwargs):
            events.append(("multi_run", kwargs))
            self.task_ids["solve"] = "multi-solve-id"
            Path(path).write_text("multi")

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT, "single": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=True,
    )
    batch._cached_properties = {"jobs": {"heat": MultiStepJob(), "single": SingleStepJob()}}

    data = batch.run(path_dir=tmp_path, priority=7)

    assert "single_upload" in events
    assert (
        "single_start",
        {"priority": 7, "vgpu_allocation": None, "ignore_memory_limit": None},
    ) in events
    assert ("single_estimate", {"verbose": False}) in events
    assert ("single_download", str(tmp_path / "single-task-id.hdf5")) in events
    multi_run_kwargs = next(event[1] for event in events if event[0] == "multi_run")
    assert multi_run_kwargs["verbose_estimate_cost"] is True
    assert multi_run_kwargs["priority"] == 7
    assert data.task_ids == {"heat": "multi-solve-id", "single": "single-task-id"}
    assert warning_messages == [
        "Batches containing both regular jobs and workflow jobs run those groups separately. "
        "For maximum parallelism, split them into separate batches."
    ]


def test_multistep_batch_temp_path_hashes_task_name(tmp_path):
    temp_path = Batch._multi_step_temp_path("../escape/subdir", tmp_path)

    assert temp_path.parent == tmp_path
    assert temp_path.name.startswith(".multi_step_")
    assert temp_path.name.endswith(".tmp.hdf5")
    assert ".." not in temp_path.name
    assert "/" not in temp_path.name


def test_multistep_batch_run_removes_temp_file_on_tolerable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(Batch, "to_file", lambda *args, **kwargs: None)
    warning_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning",
        lambda msg, **kwargs: warning_messages.append(msg),
    )

    class FailingMultiStepJob:
        is_multi_step = True
        simulation = FULL_STEADY_HEAT
        load_if_cached = False
        steps = HeatChargeWorkflow.from_simulation(FULL_STEADY_HEAT).steps
        task_ids = {}
        _step_cached_task_ids = {}
        _step_stash_paths = {}

        def __init__(self):
            self.status_calls = 0

        @property
        def status(self):
            self.status_calls += 1
            return "blocked"

        def _run_to_file(self, path, **kwargs):
            del kwargs
            Path(path).write_text("partial")
            raise WebError("blocked")

    failing_job = FailingMultiStepJob()
    batch = Batch(simulations={"heat": FULL_STEADY_HEAT}, folder_name="default", verbose=False)
    batch._cached_properties = {"jobs": {"heat": failing_job}}
    original_load = Batch.load

    def _assert_warning_before_load(self, *args, **kwargs):
        assert warning_messages == ["Not loading 'heat' as the task errored."]
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(Batch, "load", _assert_warning_before_load)

    result = batch.run(path_dir=tmp_path)

    assert result.task_paths == {}
    assert result.cached_tasks == {}
    assert failing_job.status_calls == 1
    assert batch._terminal_status_by_task["heat"] == "blocked"
    assert not list(tmp_path.glob(".multi_step_*.tmp.hdf5"))
    assert warning_messages == ["Not loading 'heat' as the task errored."]


def test_multistep_batch_run_logs_additional_fatal_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(Batch, "to_file", lambda *args, **kwargs: None)
    error_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.error",
        lambda msg, **kwargs: error_messages.append(msg),
    )

    class FatalMultiStepJob:
        is_multi_step = True
        simulation = FULL_STEADY_HEAT

        def __init__(self, task_name):
            self.task_name = task_name

        def _run_to_file(self, path, **kwargs):
            del kwargs
            Path(path).write_text("partial")
            raise RuntimeError(f"{self.task_name} failed")

    batch = Batch(
        simulations={"a": FULL_STEADY_HEAT, "b": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    batch._cached_properties = {"jobs": {"a": FatalMultiStepJob("a"), "b": FatalMultiStepJob("b")}}

    with pytest.raises(RuntimeError, match="failed"):
        batch.run(path_dir=tmp_path)

    assert not list(tmp_path.glob(".multi_step_*.tmp.hdf5"))
    assert len(error_messages) == 1
    assert error_messages[0].startswith("Additional multi-step batch job failures:")


def test_multistep_batch_load_skips_pending_jobs(monkeypatch, tmp_path):
    warning_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg, **kwargs: warning_messages.append(msg)
    )

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )

    data = batch.load(path_dir=tmp_path, skip_download=True)

    assert data.task_ids == {}
    assert data.task_paths == {}
    assert data.cached_tasks == {}
    assert warning_messages == ["Not loading 'heat' as the final workflow step hasn't completed."]


def test_multistep_batch_download_skips_pending_jobs(monkeypatch, tmp_path):
    warning_messages = []
    monkeypatch.setattr(Batch, "to_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg, **kwargs: warning_messages.append(msg)
    )

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )

    batch.download(path_dir=tmp_path)

    assert warning_messages == [
        "Not downloading 'heat' as the final workflow step hasn't completed."
    ]


def test_multistep_batch_skips_uploaded_incomplete_final_step(monkeypatch, tmp_path):
    warning_messages = []
    download_calls = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg, **kwargs: warning_messages.append(msg)
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.get_info",
        lambda task_id, **kwargs: SimpleNamespace(status="running"),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.task_api.download",
        lambda *args, **kwargs: download_calls.append((args, kwargs)),
    )

    batch = Batch(
        simulations={"heat": FULL_STEADY_HEAT},
        folder_name="default",
        verbose=False,
    )
    job = batch.jobs["heat"]
    with job._state_lock:
        job._runtime_state.task_ids["mesh"] = "mesh-id"
        job._runtime_state.owned_task_ids["mesh"] = True
        job._runtime_state.step_statuses["mesh"] = "completed"
        job._runtime_state.task_ids["solve"] = "solve-id"
        job._runtime_state.owned_task_ids["solve"] = True
        job._runtime_state.step_statuses["solve"] = "running"
        job._update_current_step_index()

    data = batch.load(path_dir=tmp_path, skip_download=True)
    assert data.task_ids == {}
    assert data.task_paths == {}

    batch.download(path_dir=tmp_path)

    assert download_calls == []
    assert warning_messages == [
        "Not loading 'heat' as the final workflow step hasn't completed.",
        "Not downloading 'heat' as the final workflow step hasn't completed.",
    ]


def test_multistep_batch_guard_methods():
    batch = Batch(simulations={"heat": FULL_STEADY_HEAT}, folder_name="default", verbose=False)
    with pytest.raises(DataError, match=r"Batch\.upload"):
        batch.upload()
    with pytest.raises(DataError, match=r"Batch\.start"):
        batch.start()
    with pytest.raises(DataError, match=r"Batch\.monitor"):
        batch.monitor()


def test_multistep_job_task_id_raises():
    job = Job(simulation=FULL_STEADY_HEAT, task_name="workflow_job", verbose=False)
    with pytest.raises(DataError, match="task_ids"):
        _ = job.task_id
