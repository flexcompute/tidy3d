"""Batch orchestration helpers for supported multi-step workflow jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from tidy3d.exceptions import DataError
from tidy3d.log import get_logging_console, log
from tidy3d.web.api import task_api
from tidy3d.web.api.states import (
    COMPLETED_PERCENT,
    DIVERGED_STATES,
    DRAFT_STATES,
    END_STATES,
    ERROR_STATES,
    STATE_PROGRESS_PERCENTAGE,
)
from tidy3d.web.api.workflow_dependencies import has_supported_parent_task_dependency

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from os import PathLike

    from rich.progress import TaskID

    from tidy3d.components.types.workflow import WorkflowOperationType
    from tidy3d.components.workflow import Step
    from tidy3d.web.api.container import Batch, BatchData, Job
    from tidy3d.web.core.constants import TaskId, TaskName
    from tidy3d.web.core.task_info import TaskInfo


WORKFLOW_BATCH_PROGRESS_REFRESH_TIME = task_api.REFRESH_TIME


class WorkflowStepJobAdapter:
    """Expose one workflow step through the single-step batch hooks."""

    is_multi_step = False
    load_if_cached = False
    task_id_cached: None = None

    def __init__(
        self,
        *,
        task_name: TaskName,
        job: Job,
        step_index: int,
        should_start: bool = True,
    ) -> None:
        self.task_name = task_name
        self.job = job
        self.step_index = step_index
        self.should_start = should_start

    @property
    def step(self) -> Step:
        return self.job._workflow_step(self.step_index)

    @property
    def simulation(self) -> WorkflowOperationType:
        return self.step.operation

    @property
    def task_id(self) -> TaskId:
        return self.job._workflow_required_step_task_id(self.step.name)

    def upload(
        self,
        *,
        verbose: bool | None = False,
        verbose_estimate_cost: bool | None = None,
    ) -> None:
        self.job._workflow_upload_step(
            self.step,
            verbose=verbose,
            verbose_estimate_cost=verbose_estimate_cost,
        )

    def start(
        self,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
    ) -> None:
        if not self.should_start:
            return

        self.job._workflow_set_step_status(self.step.name, "queued")

        task_api.start(
            task_id=self.task_id,
            solver_version=self.job.solver_version,
            pay_type=self.job.pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

        self.job._workflow_set_step_status(self.step.name, "running")

    def get_info(self) -> TaskInfo:
        return task_api.get_info(task_id=self.task_id, verbose=self.job.verbose)

    def _estimate_cost_info(self, verbose: bool = True) -> task_api.FlexCreditEstimate:
        """Estimate FlexCredit charge details for this workflow step task."""
        return task_api.estimate_cost_info(
            task_id=self.task_id,
            verbose=verbose,
            solver_version=self.job.solver_version,
            is_final_billed_cost=task_api._operation_estimate_is_final_billed_cost(
                self.step.operation
            ),
        )

    def estimate_cost(self, verbose: bool = True) -> float:
        return self._estimate_cost_info(verbose=verbose).maximum

    def download(self, path: PathLike) -> None:
        self.job._workflow_download_step(self.step.name, path=path)


@dataclass(frozen=True)
class WorkflowStepSchedule:
    """Scheduling decision for one workflow step."""

    adapter: WorkflowStepJobAdapter | None = None
    needs_upload: bool = False
    advanced: bool = False


class UniformMultiStepBatchRunner:
    """Orchestrate supported uniform multi-step workflow batches."""

    def __init__(self, batch: Batch) -> None:
        self.batch = batch

    @staticmethod
    def _workflow_shape(job: Job) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Return the orchestration-relevant workflow shape for a job."""
        return tuple(
            (
                step.name,
                tuple(
                    (step_input.upstream_step, step_input.upstream_output)
                    for step_input in step.inputs
                ),
            )
            for step in job.steps
        )

    @staticmethod
    def supports_job(job: Job) -> bool:
        """Whether a job can participate in uniform workflow batch execution."""
        if not job.is_multi_step:
            return False
        if not all(
            hasattr(job, attr)
            for attr in (
                "_workflow_advance_cache_frontier",
                "_workflow_next_pending_step_index",
                "_workflow_upload_step",
                "_workflow_sync_step_status",
            )
        ):
            return False
        return has_supported_parent_task_dependency(job.steps)

    def uniform_jobs(
        self, jobs: Mapping[TaskName, Job] | None = None
    ) -> dict[TaskName, Job] | None:
        """Return jobs when they form one supported uniform workflow batch."""
        candidates = dict(jobs or self.batch.jobs)
        if not candidates:
            return None
        if not all(self.supports_job(job) for job in candidates.values()):
            return None

        shapes = {self._workflow_shape(job) for job in candidates.values()}
        if len(shapes) != 1:
            return None
        return candidates

    @staticmethod
    def blocking_status(job: Job) -> str | None:
        """Return a terminal status that prevents this workflow from advancing."""
        for step in job.steps[:-1]:
            status = job._workflow_step_status(step.name)
            if status in ERROR_STATES or status in DIVERGED_STATES:
                return status

        final_status = job._workflow_step_status(job.steps[-1].name)
        if final_status in ERROR_STATES:
            return final_status
        return None

    def runnable_jobs(self, jobs: Mapping[TaskName, Job]) -> dict[TaskName, Job]:
        """Filter out workflow jobs that already reached a non-runnable terminal status."""
        runnable_jobs = {}
        for task_name, job in jobs.items():
            blocking_status = self.blocking_status(job)
            if blocking_status is not None:
                self.batch._terminal_status_by_task[task_name] = blocking_status
                self.batch._warn_tolerable_job_run_error(task_name, blocking_status)
                continue
            runnable_jobs[task_name] = job
        return runnable_jobs

    @staticmethod
    def next_step_index(jobs: Mapping[TaskName, Job]) -> int:
        """Return the earliest incomplete workflow step across runnable jobs."""
        if not jobs:
            return 0

        next_indices = []
        for job in jobs.values():
            job._workflow_advance_cache_frontier()
            next_indices.append(job._workflow_next_pending_step_index())
        return min(next_indices, default=len(next(iter(jobs.values())).steps))

    def prepare_step(
        self,
        task_name: TaskName,
        job: Job,
        *,
        step_index: int,
        checkpoint_batch: Callable[[], None],
    ) -> WorkflowStepSchedule:
        """Prepare the scheduling action for one workflow job at one step."""
        if step_index >= len(job.steps):
            return WorkflowStepSchedule()

        step = job.steps[step_index]
        if job._workflow_step_is_complete(step.name):
            return WorkflowStepSchedule(advanced=True)

        status = job._workflow_step_status(step.name)
        if status in ERROR_STATES or (
            step.name != job.steps[-1].name and status in DIVERGED_STATES
        ):
            self.batch._terminal_status_by_task[task_name] = status
            self.batch._warn_tolerable_job_run_error(task_name, status)
            return WorkflowStepSchedule()

        if job._workflow_restore_step_if_cached(step):
            checkpoint_batch()
            return WorkflowStepSchedule(advanced=True)

        task_id = job._workflow_step_task_id(step.name)
        if task_id is None:
            return WorkflowStepSchedule(
                adapter=WorkflowStepJobAdapter(task_name=task_name, job=job, step_index=step_index),
                needs_upload=True,
            )

        status = job._workflow_refresh_uploaded_step_status(step, task_id)
        if job._workflow_step_is_complete(step.name):
            checkpoint_batch()
            return WorkflowStepSchedule(advanced=True)

        if status in ERROR_STATES or (
            step.name != job.steps[-1].name and status in DIVERGED_STATES
        ):
            self.batch._terminal_status_by_task[task_name] = status
            self.batch._warn_tolerable_job_run_error(task_name, status)
            checkpoint_batch()
            return WorkflowStepSchedule()

        return WorkflowStepSchedule(
            adapter=WorkflowStepJobAdapter(
                task_name=task_name,
                job=job,
                step_index=step_index,
                should_start=status in DRAFT_STATES,
            )
        )

    def prepare_step_batch(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        step_index: int,
        checkpoint_batch: Callable[[], None],
    ) -> tuple[
        list[WorkflowStepJobAdapter],
        list[WorkflowStepJobAdapter],
        list[WorkflowStepJobAdapter],
    ]:
        """Prepare upload/start/monitor adapter groups for one shared workflow step."""
        jobs_to_upload: list[WorkflowStepJobAdapter] = []
        jobs_to_start: list[WorkflowStepJobAdapter] = []
        jobs_to_monitor: list[WorkflowStepJobAdapter] = []

        for task_name, job in jobs.items():
            schedule = self.prepare_step(
                task_name,
                job,
                step_index=step_index,
                checkpoint_batch=checkpoint_batch,
            )
            adapter = schedule.adapter
            if adapter is None:
                continue
            if schedule.needs_upload:
                jobs_to_upload.append(adapter)
                jobs_to_start.append(adapter)
                jobs_to_monitor.append(adapter)
                continue
            if adapter.should_start:
                jobs_to_start.append(adapter)
            jobs_to_monitor.append(adapter)

        return jobs_to_upload, jobs_to_start, jobs_to_monitor

    def sync_step_statuses(
        self,
        jobs_to_monitor: list[WorkflowStepJobAdapter],
        *,
        is_final_step: bool,
    ) -> None:
        """Copy monitored step statuses back onto the owning workflow jobs."""
        for adapter in jobs_to_monitor:
            task_name = adapter.task_name
            job = adapter.job
            step = adapter.step
            status = task_api.get_info(task_id=adapter.task_id, verbose=False).status

            job._workflow_sync_step_status(step, status)

            if is_final_step:
                if status in END_STATES:
                    self.batch._terminal_status_by_task[task_name] = (
                        job._workflow_batch_terminal_status()
                    )
                    final_server_task_id = self.batch._known_multi_step_server_task_id(job)
                    if final_server_task_id is not None:
                        self.batch._terminal_task_id_by_task[task_name] = final_server_task_id
                if status in ERROR_STATES:
                    self.batch._warn_tolerable_job_run_error(task_name, status)
                continue

            if status in ERROR_STATES or status in DIVERGED_STATES:
                self.batch._terminal_status_by_task[task_name] = status
                self.batch._warn_tolerable_job_run_error(task_name, status)
            else:
                self.batch._terminal_status_by_task.pop(task_name, None)
                self.batch._terminal_task_id_by_task.pop(task_name, None)

    def complete_step_batch(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        step_index: int,
        batch_path: PathLike,
        priority: int | None,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
        path_dir: PathLike,
        replace_existing: bool,
    ) -> None:
        """Complete one workflow step across all runnable jobs in the batch."""

        def checkpoint_batch() -> None:
            self.batch.to_file(batch_path)

        jobs_to_upload, jobs_to_start, jobs_to_monitor = self.prepare_step_batch(
            jobs,
            step_index=step_index,
            checkpoint_batch=checkpoint_batch,
        )
        if jobs_to_upload:
            self.batch._upload_jobs(jobs_to_upload)
            checkpoint_batch()

        for adapter in jobs_to_monitor:
            self.batch._terminal_status_by_task.pop(adapter.task_name, None)
            self.batch._terminal_task_id_by_task.pop(adapter.task_name, None)

        if jobs_to_start:
            self.batch._start_jobs(
                jobs_to_start,
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
            )

        is_final_step = step_index == len(next(iter(jobs.values())).steps) - 1
        if jobs_to_monitor:
            self.batch._monitor_jobs(
                {adapter.task_name: adapter for adapter in jobs_to_monitor},
                download_on_success=is_final_step,
                path_dir=path_dir,
                replace_existing=replace_existing,
            )
            self.sync_step_statuses(
                jobs_to_monitor,
                is_final_step=is_final_step,
            )

        checkpoint_batch()

    def download_results(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        path_dir: PathLike,
        replace_existing: bool,
    ) -> None:
        """Ensure completed final workflow artifacts are materialized for batch loading."""
        for task_name, job in jobs.items():
            status = self.batch._terminal_status_by_task.get(task_name)
            if status is None:
                status = job.status
            if status in END_STATES:
                self.batch._terminal_status_by_task[task_name] = status
            if status in ERROR_STATES:
                continue

            final_task_id = self.batch._known_multi_step_result_task_id(task_name, job)
            if final_task_id is None:
                continue

            final_path = self.batch._multi_step_result_path(task_name, job, path_dir)
            if final_path.exists() and not replace_existing:
                final_server_task_id = self.batch._known_multi_step_server_task_id(job)
                if final_server_task_id is not None:
                    self.batch._terminal_task_id_by_task[task_name] = final_server_task_id
                continue

            job._workflow_download_step(job.steps[-1].name, path=final_path)
            final_server_task_id = self.batch._known_multi_step_server_task_id(job)
            if final_server_task_id is not None:
                self.batch._terminal_task_id_by_task[task_name] = final_server_task_id

    @staticmethod
    def advance_cache_frontier(job: Job) -> None:
        """Restore consecutive cached workflow steps before scheduling or estimating."""
        job._workflow_advance_cache_frontier()

    def mark_job_terminal(self, task_name: TaskName, job: Job) -> None:
        """Remember terminal status for a completed multi-step job."""
        status = job._workflow_batch_terminal_status()
        self.batch._terminal_status_by_task[task_name] = status
        final_server_task_id = self.batch._known_multi_step_server_task_id(job)
        if final_server_task_id is not None:
            self.batch._terminal_task_id_by_task[task_name] = final_server_task_id

    def start_next_task(
        self,
        task_name: TaskName,
        job: Job,
        *,
        checkpoint_batch: Callable[[], None],
        priority: int | None,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
    ) -> WorkflowStepJobAdapter | None:
        """Upload/start the next schedulable workflow step for one job."""
        while True:
            self.advance_cache_frontier(job)
            step_idx = job._workflow_next_pending_step_index()
            if step_idx >= len(job.steps):
                self.mark_job_terminal(task_name, job)
                checkpoint_batch()
                return None

            schedule = self.prepare_step(
                task_name,
                job,
                step_index=step_idx,
                checkpoint_batch=checkpoint_batch,
            )
            adapter = schedule.adapter
            if adapter is None:
                if schedule.advanced:
                    continue
                checkpoint_batch()
                return None

            if schedule.needs_upload:
                try:
                    adapter.upload(verbose_estimate_cost=self.batch.verbose)
                except Exception as exc:
                    log.error(
                        f"Failed to upload workflow step '{adapter.step.name}' for task "
                        f"'{task_name}': {exc.__class__.__name__}: {exc}"
                    )
                    raise
                checkpoint_batch()

            if adapter.should_start:
                self.batch._start_jobs(
                    [adapter],
                    priority=priority,
                    vgpu_allocation=vgpu_allocation,
                    ignore_memory_limit=ignore_memory_limit,
                )
            return adapter

    @staticmethod
    def job_complete(job: Job) -> bool:
        """Whether a workflow job has no more runnable steps."""
        return job._workflow_next_pending_step_index() >= len(job.steps)

    def run_scheduler(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        batch_path: PathLike,
        priority: int | None,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
    ) -> None:
        """Run uniform workflows with a central rolling scheduler."""

        def checkpoint_batch() -> None:
            self.batch.to_file(batch_path)

        active_jobs: dict[TaskName, WorkflowStepJobAdapter] = {}
        max_active_steps = self.batch.num_workers or len(jobs)
        max_task_name = max(len(task_name) for task_name in jobs)
        max_name_length = min(30, max(max_task_name, 15))
        pbar_tasks: dict[TaskName, TaskID] = {}

        def pbar_description(task_name: TaskName, step_name: str, status: str) -> str:
            display_task_name = task_name
            if len(display_task_name) > max_name_length - 3:
                display_task_name = display_task_name[: (max_name_length - 3)] + "..."
            task_part = f"{display_task_name:<{max_name_length}}"
            return f"{task_part} {step_name} -> {status}"

        def update_progress(
            progress: Progress | None,
            task_name: TaskName,
            step_name: str,
            status: str,
        ) -> None:
            if progress is None:
                return
            if task_name not in pbar_tasks:
                pbar_tasks[task_name] = progress.add_task(
                    pbar_description(task_name, step_name, status),
                    total=COMPLETED_PERCENT,
                    completed=STATE_PROGRESS_PERCENTAGE.get(status, 0),
                )
                return
            progress.update(
                pbar_tasks[task_name],
                description=pbar_description(task_name, step_name, status),
                completed=STATE_PROGRESS_PERCENTAGE.get(status, COMPLETED_PERCENT),
            )

        def schedule_available_work(progress: Progress | None) -> bool:
            scheduled = False
            for task_name, job in jobs.items():
                if task_name in active_jobs:
                    continue
                self.advance_cache_frontier(job)
                blocking_status = self.blocking_status(job)
                if blocking_status is not None:
                    self.batch._terminal_status_by_task[task_name] = blocking_status
                    self.batch._warn_tolerable_job_run_error(task_name, blocking_status)
                    update_progress(progress, task_name, job.steps[-1].name, blocking_status)
                    continue
                if self.job_complete(job):
                    self.mark_job_terminal(task_name, job)
                    update_progress(
                        progress,
                        task_name,
                        job.steps[-1].name,
                        job._workflow_batch_terminal_status(),
                    )
                    continue

                if len(active_jobs) >= max_active_steps:
                    continue

                adapter = self.start_next_task(
                    task_name,
                    job,
                    checkpoint_batch=checkpoint_batch,
                    priority=priority,
                    vgpu_allocation=vgpu_allocation,
                    ignore_memory_limit=ignore_memory_limit,
                )
                if adapter is None:
                    continue
                active_jobs[task_name] = adapter
                scheduled = True
                update_progress(
                    progress,
                    task_name,
                    adapter.step.name,
                    job._workflow_step_status(adapter.step.name),
                )
            return scheduled

        def poll_active_work(progress: Progress | None) -> bool:
            completed_adapters: list[WorkflowStepJobAdapter] = []
            for task_name, adapter in list(active_jobs.items()):
                status = adapter.get_info().status
                update_progress(progress, task_name, adapter.step.name, status)
                if status in END_STATES:
                    completed_adapters.append(adapter)

            if not completed_adapters:
                return False

            for adapter in completed_adapters:
                task_name = adapter.task_name
                self.sync_step_statuses(
                    [adapter],
                    is_final_step=adapter.step_index == len(adapter.job.steps) - 1,
                )
                active_jobs.pop(task_name, None)
                status = adapter.job._workflow_step_status(adapter.step.name)
                update_progress(progress, task_name, adapter.step.name, status)
            checkpoint_batch()
            return True

        console = get_logging_console() if self.batch.verbose else None
        progress_columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        progress_context = (
            Progress(*progress_columns, console=console, transient=False)
            if self.batch.verbose
            else None
        )

        if progress_context is None:
            while True:
                scheduled = schedule_available_work(None)
                completed = poll_active_work(None)
                if active_jobs:
                    if not completed:
                        time.sleep(task_api.REFRESH_TIME)
                    continue
                if not scheduled and not completed:
                    break
            return

        with progress_context as progress:
            while True:
                scheduled = schedule_available_work(progress)
                completed = poll_active_work(progress)
                progress.refresh()
                if active_jobs:
                    if not completed:
                        time.sleep(WORKFLOW_BATCH_PROGRESS_REFRESH_TIME)
                    continue
                if not scheduled and not completed:
                    break
            console.log("Batch complete.")

    def run_batch(
        self,
        jobs: Mapping[TaskName, Job],
        *,
        path_dir: PathLike,
        priority: int | None,
        replace_existing: bool,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
    ) -> BatchData:
        """Run a supported uniform multi-step batch with rolling workflow scheduling."""
        self.batch._check_path_dir(path_dir)
        batch_path = self.batch._batch_path(path_dir=path_dir)

        self.run_scheduler(
            jobs,
            batch_path=batch_path,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

        self.download_results(
            jobs,
            path_dir=path_dir,
            replace_existing=replace_existing,
        )
        self.batch.to_file(batch_path)
        return self.batch.load(path_dir=path_dir, skip_download=True)

    def step(
        self,
        *,
        path_dir: PathLike,
        priority: int | None,
        replace_existing: bool,
        vgpu_allocation: int | None,
        ignore_memory_limit: bool | None,
    ) -> BatchData | None:
        """Complete the next shared workflow step for a supported uniform workflow batch."""
        jobs = self.uniform_jobs()
        if jobs is None:
            raise DataError(
                "Batch.step() supports only uniform Heat and HeatCharge multi-step batches."
            )

        self.batch._check_path_dir(path_dir)
        batch_path = self.batch._batch_path(path_dir=path_dir)
        runnable_jobs = self.runnable_jobs(jobs)
        if not runnable_jobs:
            raise DataError("No runnable workflow steps remain in this batch.")

        step_index = self.next_step_index(runnable_jobs)
        if step_index >= len(next(iter(runnable_jobs.values())).steps):
            raise DataError(
                "All workflow steps are already complete. Batch.step() only advances "
                "an incomplete batch workflow one step. Use 'Batch.load()' to load "
                "completed results, or 'Batch.run()' to return the final results, "
                "including results restored from the local cache."
            )

        is_final_step = step_index == len(next(iter(runnable_jobs.values())).steps) - 1
        self.complete_step_batch(
            runnable_jobs,
            step_index=step_index,
            batch_path=batch_path,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            path_dir=path_dir,
            replace_existing=replace_existing,
        )

        if not is_final_step:
            self.batch.to_file(batch_path)
            return None

        self.download_results(
            jobs,
            path_dir=path_dir,
            replace_existing=replace_existing,
        )
        self.batch.to_file(batch_path)
        return self.batch.load(path_dir=path_dir, skip_download=True)

    def estimate_frontier_cost(
        self,
        jobs: Mapping[TaskName, Job],
        verbose: bool = True,
    ) -> float:
        """Estimate the shared next workflow step for a uniform multi-step batch."""
        next_step_indices: dict[TaskName, int] = {}
        for task_name, job in jobs.items():
            self.advance_cache_frontier(job)
            blocking_status = self.blocking_status(job)
            if blocking_status is not None:
                raise DataError(
                    "Batch.estimate_cost() does not support workflow batches with failed "
                    f"or diverged jobs. Job '{task_name}' is at status '{blocking_status}'."
                )

            step_idx = job._workflow_next_pending_step_index()
            if step_idx < len(job.steps):
                next_step_indices[task_name] = step_idx

        if not next_step_indices:
            if verbose:
                console = get_logging_console()
                console.log("No FlexCredit cost for the batch as all workflow steps are complete.")
            return 0.0

        if len(next_step_indices) != len(jobs) or len(set(next_step_indices.values())) != 1:
            raise DataError(
                "Batch.estimate_cost() for workflow batches requires all jobs to be at the "
                "same next workflow step. Use Job.estimate_cost() per job for partially "
                "resumed workflow batches."
            )

        step_idx = next(iter(next_step_indices.values()))
        step_name = next(iter(jobs.values())).steps[step_idx].name
        job_estimates = [job._estimate_cost_info(verbose=False) for job in jobs.values()]
        batch_cost = sum(estimate.maximum for estimate in job_estimates)
        batch_typical_cost = task_api._batch_typical_flex_credit_cost(job_estimates)

        if verbose:
            console = get_logging_console()
            if batch_typical_cost is not None:
                console.log(
                    f"Estimated typical FlexCredit cost: {batch_typical_cost:1.3f} "
                    f"for the next workflow step '{step_name}' across the batch."
                )
                console.log(
                    f"Maximum FlexCredit cost: {batch_cost:1.3f} for the next workflow "
                    f"step '{step_name}' across the batch."
                )
                if any(
                    task_api._estimate_has_charge_solver_iteration_scaling(estimate)
                    for estimate in job_estimates
                ):
                    console.log(
                        "For charge simulations, the billed cost depends on the number of "
                        "solver iterations required for convergence."
                    )
            else:
                console.log(
                    f"Maximum FlexCredit cost: {batch_cost:1.3f} for the next workflow "
                    f"step '{step_name}' across the batch."
                )
            if step_idx == 0:
                console.log(
                    "This estimates the mesh step only. Run the mesh step first with "
                    "'Batch.step()'; after it completes, call 'Batch.estimate_cost()' again "
                    "for the solver estimate."
                )
            elif step_idx == len(next(iter(jobs.values())).steps) - 1:
                console.log(
                    "All jobs are at the final solver step, so this is the estimated solver "
                    "cost for the batch."
                )
            else:
                console.log(
                    "This estimates only the current workflow step. Downstream step costs "
                    "may require this step to complete first."
                )

        return batch_cost
