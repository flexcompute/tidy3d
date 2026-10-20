"""Task operations in private and company shared workspaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from requests import RequestException

from tidy3d.log import log
from tidy3d.web.core.exceptions import WebError, WebNotFoundError
from tidy3d.web.core.http_util import http
from tidy3d.web.core.task_core import Folder, SimulationTask
from tidy3d.web.core.workspace import (
    ActivityEntry,
    UserInfo,
    resolve_task_context,
    resolve_workspace,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tidy3d.web.core.task_info import TaskInfo
    from tidy3d.web.core.workspace import Workspace


def _require_task_workspace(info: TaskInfo) -> None:
    if info.workspace is None:
        raise WebError(
            f"The service does not expose the workspace for task '{info.taskId}'.", log_error=False
        )


def _workspace_users(project_id: str = "") -> list[UserInfo]:
    return TypeAdapter(list[UserInfo]).validate_python(
        http.get("tidy3d/workspaces/users", params={"id": project_id}) or []
    )


def list_users(task_id: str) -> list[UserInfo]:
    """List task members and their edit-lock state when reported by the service.

    Parameters
    ----------
    task_id : str
        Task whose accessible members should be listed.

    Returns
    -------
    list[UserInfo]
        Members with access. ``isLockHolder=None`` means the service does not
        expose the member's lock state.
    """
    info, group = resolve_task_context(
        SimulationTask(taskId=task_id).detail(), include_private_owner=True
    )
    if not info.groupId:
        raise WebError(f"Task '{task_id}' has no project identifier for listing users.")
    _require_task_workspace(info)
    if info.workspace == "private":
        owner = group.get("userId")
        if not owner:
            raise WebError(f"The service does not expose the owner of private task '{task_id}'.")
        return [UserInfo(userId=owner, isLockHolder=False)]
    # The unfiltered endpoint lists all members; a project filter returns only
    # currently active viewers and the editor, rather than every member with access.
    members = _workspace_users()
    active_by_id = {item.userId: item for item in _workspace_users(info.groupId)}
    users = []
    for item in members:
        member = active_by_id.get(item.userId, item)
        if member.isLockHolder is None:
            member.isLockHolder = False
        users.append(member)
    return users


def _prepare_tasks(
    task_ids: str | list[str], *, duplicate: bool
) -> tuple[list[TaskInfo], list[str]]:
    """Check all source tasks before creating a destination or mutating tasks."""
    task_ids = [task_ids] if isinstance(task_ids, str) else task_ids
    if not isinstance(task_ids, list) or any(not isinstance(i, str) or not i for i in task_ids):
        raise ValueError("task_ids must be a task ID or a list of nonempty task IDs.")
    tasks, failures = [], []
    for task_id in task_ids:
        try:
            info, _ = resolve_task_context(
                SimulationTask(taskId=task_id).detail(),
                include_folder=not duplicate,
                include_operation=not duplicate,
            )
            if not duplicate and info.operationStatus in {"moving", "copying"}:
                raise WebError(
                    f"Project operation '{info.operationStatus}' is still in progress for task "
                    f"'{task_id}'; try again when it finishes.",
                    log_error=False,
                )
            if not info.groupId or (duplicate and not info.version):
                raise WebError(
                    "The service does not expose the project identifier or version.",
                    log_error=False,
                )
            _require_task_workspace(info)
            if not duplicate and not info.folderId:
                raise WebError("The service does not expose this task's folder.", log_error=False)
            tasks.append(info)
        except WebError as exc:
            failures.append(f"{task_id}: {exc}")
    return tasks, failures


def _check_move_locks(projects: Iterable[list[TaskInfo]]) -> list[str]:
    """Check active editors for every shared source without ever acquiring a lock."""
    failures = []
    for tasks in projects:
        info = tasks[0]
        if info.workspace != "shared":
            continue
        try:
            holders = [
                item.personName or item.userId
                for item in _workspace_users(info.groupId)
                if item.role == "EDITOR" or item.isLockHolder
            ]
            if holders:
                raise WebError("Edit lock held by " + ", ".join(holders), log_error=False)
        except WebError as exc:
            failures.extend(f"{task.taskId}: {exc}" for task in tasks)
    return failures


def move(task_ids: str | list[str], folder_name: str, workspace: Workspace | None = None) -> None:
    """Move the projects containing the supplied tasks, including all revisions.

    Parameters
    ----------
    task_ids : str | list[str]
        Source task IDs. Each containing project is moved once, even when
        multiple supplied IDs belong to that project.
    folder_name : str
        Destination folder, created if absent.
    workspace : Literal['private', 'shared'] | None
        Destination workspace. Defaults to ``td.config.workspace.default``.

    Notes
    -----
    All source tasks are checked before moving. Each move affects every revision
    of the containing project. Crossing workspaces changes visibility for all
    revisions and removes existing share links. The client processes one project
    per request, so a later failure can leave earlier tasks moved. The error
    reports completed task IDs. Edit locks are enforced by the service.

    Cross-workspace moves run in the background: returning means the requests
    were accepted, not that files have arrived. Transfers may take several
    minutes and time out after 30 minutes. Poll ``web.get_info(task_id).operationStatus``
    for ``move_completed`` or ``move_timeout``. A reported ``moving`` or ``copying``
    source is rejected before sending another move request.
    """
    workspace = resolve_workspace(workspace)
    tasks, failures = _prepare_tasks(task_ids, duplicate=False)
    folder = Folder.get(folder_name, workspace=workspace) if tasks else None
    if folder is not None:
        tasks = [info for info in tasks if info.folderId != folder.folder_id]
    projects = {}
    for info in tasks:
        projects.setdefault(info.groupId, []).append(info)
    failures.extend(_check_move_locks(projects.values()))
    if failures:
        raise WebError("Tasks could not be moved: " + "; ".join(failures))
    if not tasks:
        return
    folder = folder or Folder.create(folder_name, workspace=workspace)
    completed = []
    for project_tasks in projects.values():
        info = project_tasks[0]
        affected_task_ids = [task.taskId for task in project_tasks]
        if info.workspace != workspace:
            log.warning(
                f"Moving project '{info.groupId}' from {info.workspace} to {workspace} workspace "
                "changes who can see all of its revisions and removes existing share links."
            )
        try:
            if info.workspace == workspace:
                http.put(
                    "tidy3d/group/movement",
                    {
                        "parentFolderId": folder.folder_id,
                        "items": [{"id": info.groupId, "type": "Tidy3D"}],
                    },
                )
            else:
                http.put(
                    "tidy3d/workspaces/projectsMovement",
                    {"sourceProjectId": info.groupId, "destFolderId": folder.folder_id},
                )
        except (WebError, RequestException) as exc:
            uncertain = (
                " The current move may have completed; verify its outcome before retrying."
                if isinstance(exc, RequestException)
                else ""
            )
            raise WebError(
                f"Could not move tasks {affected_task_ids}: {exc}. Already moved: {completed}."
                + uncertain
            ) from exc
        completed.extend(affected_task_ids)


def duplicate(
    task_ids: str | list[str], folder_name: str, workspace: Workspace | None = None
) -> list[str]:
    """Copy selected project revisions, returning new task IDs in input order.

    Parameters
    ----------
    task_ids : str | list[str]
        Source task IDs, each selecting its containing project revision.
        Repeated IDs create separate copies. The client never takes an edit lock.
    folder_name : str
        Destination folder, created if absent. May be the source folder.
    workspace : Literal['private', 'shared'] | None
        Destination workspace. Defaults to ``td.config.workspace.default``.

    Returns
    -------
    list[str]
        New task IDs in input order. Copy names are assigned by the service.

    Notes
    -----
    Each input copies one project revision, rather than every revision in the
    project. Requests run sequentially. An error reports copies already created.
    The service may reject a source with an active editor and temporarily
    report a copying status on the source while copying its data.
    """
    workspace = resolve_workspace(workspace)
    tasks, failures = _prepare_tasks(task_ids, duplicate=True)
    if failures:
        raise WebError("Source tasks could not be accessed: " + "; ".join(failures))
    if not tasks:
        return []
    folder = Folder.create(folder_name, workspace=workspace)
    copies = []
    for info in tasks:
        if info.workspace != workspace:
            log.warning(
                f"Duplicating revision '{info.version}' of project '{info.groupId}' from "
                f"{info.workspace} to {workspace} workspace creates a copy with the "
                "destination workspace's visibility. The source stays in its workspace."
            )
        project_id = None
        try:
            response = http.put(
                "tidy3d/workspaces/projectsDuplication",
                {
                    "sourceProjectId": info.groupId,
                    "sourceVersion": info.version,
                    "destFolderId": folder.folder_id,
                },
            )
            project_id = response.get("projectId") if isinstance(response, dict) else None
            if not isinstance(project_id, str) or not project_id:
                raise WebError(
                    "The copy response does not expose a project ID; verify before retrying."
                )
            copied = http.get(f"tidy3d/group/{project_id}")
            task_id = copied.get("taskId") if isinstance(copied, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise WebError(f"Copy project '{project_id}' does not yet expose a task ID.")
            copies.append(task_id)
        except (WebError, RequestException) as exc:
            raise WebError(
                f"Could not duplicate task '{info.taskId}': {exc}. Created task IDs: {copies}; "
                f"current copy project: {project_id}. The current copy may have been created; "
                "verify its outcome before retrying."
            ) from exc
    return copies


def get_activity(
    folder_name: str | None = None,
    task_id: str | None = None,
    workspace: Workspace = "shared",
) -> list[ActivityEntry]:
    """Return activity for a folder or the project containing a task.

    Parameters
    ----------
    folder_name : str | None
        Folder whose activity should be listed.
    task_id : str | None
        Task identifying the project whose activity should be listed, across
        all revisions; mutually exclusive with folder_name.
    workspace : Literal['private', 'shared']
        Workspace containing folder_name. Ignored for task_id.

    Returns
    -------
    list[ActivityEntry]
        Activity records, including time, user, action, and target.

    Raises
    ------
    ValueError
        Both or neither target is supplied.
    NotImplementedError
        The service does not support activity for the requested target.
    """
    if (folder_name is None) == (task_id is None):
        raise ValueError("Pass exactly one of folder_name or task_id.")
    if task_id is not None:
        info, _ = resolve_task_context(SimulationTask(taskId=task_id).detail())
        if not info.groupId:
            raise NotImplementedError("Task activity is not yet supported by this service.")
        _require_task_workspace(info)
        if info.workspace == "private":
            raise NotImplementedError("Private task activity is not yet supported by this service.")
        params = {"projectIds": [info.groupId]}
    else:
        if resolve_workspace(workspace) == "private":
            raise NotImplementedError(
                "Private folder activity is not yet supported by this service."
            )
        folder = Folder.get(folder_name, workspace=workspace)
        if folder is None:
            raise WebNotFoundError(f"Folder '{folder_name}' was not found in '{workspace}'.")
        params = {"folderId": folder.folder_id}
    entries = []
    params.update(pageIndex=1, pageSize=100)
    while True:
        try:
            page = http.get("tidy3d/workspaces/log", params=params) or {}
        except WebError as exc:
            if not isinstance(exc, WebNotFoundError) and exc.status_code != 501:
                raise
            raise NotImplementedError(
                f"Activity is not yet supported by this service: {exc}"
            ) from exc
        records = page.get("records", [])
        entries.extend(ActivityEntry(**item) for item in records)
        cursor = page.get("lastCursorKey")
        if not records or len(entries) >= page.get("total", float("inf")):
            return entries
        if cursor:
            if params.get("lastCursorKey") == cursor:
                raise WebError("The service returned a repeated activity cursor.")
            params["lastCursorKey"] = cursor
        elif "lastCursorKey" in params:
            return entries
        elif page.get("total") is None and len(records) < page.get("pageSize", params["pageSize"]):
            return entries
        params["pageIndex"] += 1
