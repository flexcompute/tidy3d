"""FastMCP tool adapters for Tidy3D bridge operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from pydantic import Field

from . import python_env, screenshots, viewer

if TYPE_CHECKING:
    from typing import NoReturn

ViewerId = Annotated[str, Field(description="Identifier returned by validate_simulation")]
ViewerDirection = Annotated[
    str,
    Field(description="One of TOP, BOTTOM, LEFT, RIGHT, FRONT, BACK (case-insensitive)"),
]
VisibilityInput = Annotated[
    list[object],
    Field(description="Structure visibility flags; truthiness normalized per entry"),
]


def _raise_tool_error(exc: ValueError | RuntimeError) -> NoReturn:
    """Translate pure bridge helper failures into FastMCP's tool-error boundary."""

    raise ToolError(str(exc)) from exc


async def detect_python_environment(resource: str | None = None) -> ToolResult:
    """Resolve the active Python interpreter, environment manager, and project manager."""

    try:
        structured = python_env.detect_python_environment_payload(resource)
    except RuntimeError as exc:
        _raise_tool_error(exc)
    return ToolResult(
        content=[TextContent(type="text", text=python_env.python_environment_summary(structured))],
        structured_content=structured,
    )


async def rotate_viewer(viewer_id: ViewerId, direction: ViewerDirection) -> ToolResult:
    """Align the viewer camera to the requested orientation."""

    try:
        payload = viewer.rotate_viewer_payload(viewer_id, direction)
    except (ValueError, RuntimeError) as exc:
        _raise_tool_error(exc)
    return ToolResult(
        content=[TextContent(type="text", text=f"Viewer aligned to {payload['direction']}")],
        structured_content=payload,
    )


async def show_structures(viewer_id: ViewerId, visibility: VisibilityInput) -> ToolResult:
    """Toggle structure visibility with normalized boolean flags."""

    try:
        payload = viewer.show_structures_payload(viewer_id, visibility)
    except (ValueError, RuntimeError) as exc:
        _raise_tool_error(exc)
    return ToolResult(
        content=[TextContent(type="text", text=f"Updated visibility for {viewer_id}")],
        structured_content=payload,
    )


async def validate_simulation(
    file: Annotated[
        str | None, Field(description="Absolute path or workspace URI to evaluate", default=None)
    ] = None,
    symbol: Annotated[
        str | None,
        Field(description="Optional variable name selecting a tidy3d.Simulation", default=None),
    ] = None,
    index: Annotated[
        int | None,
        Field(description="Zero-based simulation index when multiple exist", ge=0, default=None),
    ] = None,
    viewer_id: Annotated[
        str | None, Field(description="Existing viewer identifier to revalidate", default=None)
    ] = None,
) -> ToolResult:
    """Validate a simulation file or refresh an existing viewer."""

    try:
        payload = viewer.validate_simulation_payload(
            file=file,
            symbol=symbol,
            index=index,
            viewer_id=viewer_id,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_tool_error(exc)
    summary_status = payload.get("status") or payload.get("error") or "ok"
    resolved_viewer_id = payload.get("viewer_id") or viewer_id or "<unknown>"
    return ToolResult(
        content=[
            TextContent(type="text", text=f"Validation for {resolved_viewer_id}: {summary_status}")
        ],
        structured_content=payload,
    )


async def capture(viewer_id: ViewerId) -> ToolResult:
    """Capture a single frame from the viewer."""

    try:
        frame = screenshots.capture_frame_payload(viewer_id)
    except (ValueError, RuntimeError) as exc:
        _raise_tool_error(exc)
    image = frame["image"]
    image_content = Image(data=image["data"], format=image["format"]).to_image_content(
        mime_type=image["mime"]
    )
    structured = {"viewer_id": frame["viewer_id"], "images": [image_content]}
    return ToolResult(
        content=[
            TextContent(type="text", text=f"Captured viewer frame for {viewer_id}"),
            image_content,
        ],
        structured_content=structured,
    )
