"""Workspace selection and service response models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from tidy3d.config import config

from .exceptions import WebError
from .http_util import http

if TYPE_CHECKING:
    from .task_info import TaskInfo

Workspace = Literal["private", "shared"]


class WorkspaceInfo(BaseModel):
    """A workspace returned by the service."""

    id: str
    name: str
    type: str
    status: str | None = None
    rootFolderId: str | None = None

    @property
    def workspace(self) -> Workspace:
        """Return the configuration value for this workspace."""
        return resolve_workspace(self.type.lower())


class UserInfo(BaseModel):
    """A member who can access a task."""

    userId: str
    personName: str | None = None
    email: str | None = None
    role: str | None = None
    isLockHolder: bool | None = None
    """Whether this member holds the lock; ``None`` when not reported."""

    @model_validator(mode="before")
    @classmethod
    def _lock_role(cls, data: Any) -> Any:
        if isinstance(data, dict) and "isLockHolder" not in data and data.get("role"):
            data = {**data, "isLockHolder": data["role"] == "EDITOR"}
        return data


class ActivityEntry(BaseModel):
    """An activity's time, actor, action, and target."""

    time: datetime = Field(alias="createdAt")
    user: str | None = Field(default=None, alias="userId")
    action: str = Field(alias="type")
    target: str | None = Field(default=None, alias="projectId")
    folderId: str | None = None
    userName: str | None = None
    projectName: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _activity_target(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {**data, "projectId": data.get("projectId") or data.get("folderId")}
        return data


def resolve_workspace(workspace: Workspace | None = None) -> Workspace:
    """Resolve and validate one workspace selection."""
    workspace = config.workspace.default if workspace is None else workspace
    if workspace not in ("private", "shared"):
        raise ValueError("Workspace must be 'private' or 'shared'.")
    return workspace


def list_workspaces() -> list[WorkspaceInfo]:
    """List the private workspace and any enabled company shared workspace."""
    try:
        response = http.get("tidy3d/workspaces")
    except WebError as exc:
        raise WebError(
            f"Could not list workspaces. The service may not support workspaces: {exc}"
        ) from exc
    workspaces = TypeAdapter(list[WorkspaceInfo]).validate_python(response or [])
    available = [
        item
        for item in workspaces
        if item.workspace == "private" or (item.status or "").upper() == "ENABLED"
    ]
    return sorted(available, key=lambda item: item.workspace != "private")


def require_workspace(workspace: Workspace) -> None:
    """Reject unavailable shared destinations before looking up or creating folders."""
    if workspace == "shared" and not any(item.workspace == "shared" for item in list_workspaces()):
        raise WebError(
            "Shared workspace is unavailable. You must belong to a company with "
            "Shared Workspace enabled, and the service must support shared workspaces."
        )


def resolve_task_context(
    info: TaskInfo,
    *,
    include_private_owner: bool = False,
    include_folder: bool = False,
    include_operation: bool = False,
) -> tuple[TaskInfo, dict[str, Any]]:
    """Resolve required task metadata and retain any fetched project details."""
    group = {}
    if info.groupId and (
        info.workspace is None
        or (include_private_owner and info.workspace == "private")
        or (include_folder and not info.folderId)
        or (include_operation and info.operationStatus is None)
    ):
        context = f"project details for task '{info.taskId}' (project '{info.groupId}')"
        try:
            group = http.get(
                f"tidy3d/group/{info.groupId}",
                params={"version": info.version},
                suppress_404=True,
                log_error=False,
            )
        except WebError as exc:
            raise WebError(
                f"Could not retrieve {context}: {exc}",
                status_code=getattr(exc, "status_code", None),
                log_error=False,
            ) from exc
        if not isinstance(group, dict):
            raise WebError(f"The service returned no {context}.", log_error=False)
        item = group.get("workspaceItem")
        if item:
            info.workspace = resolve_workspace(item["type"].lower())
            info.workspaceId = item.get("id")
        info.folderId = group.get("folderId") or info.folderId
        info.creator = group.get("createdBy") or info.creator
        info.operationStatus = group.get("operationStatus") or info.operationStatus
    return info, group
