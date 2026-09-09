"""Shared grid specification constants and errors."""

from __future__ import annotations

from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D
from tidy3d.exceptions import SetupError

# Scaling factor applied to internally generated lower bound of grid size that is computed from
# estimated minimal grid size
MIN_STEP_BOUND_SCALE = 0.5

# Minimum physically meaningful grid spacing in micrometers. Smaller values usually indicate
# that input quantities were specified with the wrong unit scale.
MIN_GRID_SPACING = 1e-6
UNITS_HELP_URL = (
    "https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/"
    "What-are-the-units-used-in-the-simulation.html"
)

# Default refinement factor in GridRefinement when both dl and refinement_factor are not defined
DEFAULT_REFINEMENT_FACTOR = 2

# Max passes when unioning same-grid-size in-plane overrides. Collapsing connected components to
# bounding boxes can spawn new overlaps, so the union iterates to a fixpoint; this caps that loop
# as a safety bound. Real geometries converge in a couple of passes; a leftover overlap after the
# cap is harmless (same-``dl`` ``shadow=False`` boxes still mesh correctly), just not fully merged.
INPLANE_OVERRIDE_UNION_MAX_ITERS = 5

# Tolerance for distinguishing pec/grid intersections. Also the minimum in-plane extent a
# geometry must have for small-geometry refinement to resolve it; below it the axis is skipped
# as a near-zero sliver.
GAP_MESHING_TOL = 1e-3

CornersAndConvexity = tuple[list[ArrayFloat2D], list[ArrayFloat1D]]

# Fraction of detected gap width used to set dl_min_from_gaps
DL_MIN_FROM_GAPS_FRACTION = 0.45

# Threshold for warning when dl_min_from_gaps is very small relative to lateral grid size
GAP_REFINEMENT_WARNING_THRESH = 0.1


class _GeneratedGridSizeError(SetupError):
    """Raised when a generated grid spacing is below the supported minimum."""

    def __init__(self, message: str, grid_name: str, axis_name: str, min_size: float) -> None:
        self.grid_name = grid_name
        self.axis_name = axis_name
        self.min_size = min_size
        super().__init__(message)
