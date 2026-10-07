"""=================================================================================================
Descartes modified from https://pypi.org/project/descartes/ for Shapely >= 1.8.0

Copyright Flexcompute 2022

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    from numpy.typing import ArrayLike
    from shapely.geometry.base import BaseGeometry

import shapely
from numpy import cumsum, diff, flatnonzero, full, split


class Polygon:
    """Adapt Shapely polygons to a common interface"""

    def __init__(self, context: dict[str, Any]) -> None:
        if isinstance(context, dict):
            self.context = context["coordinates"]
        else:
            self.context = context

    @property
    def exterior(self) -> Any:
        """Get polygon exterior."""
        value = getattr(self.context, "exterior", None)
        if value is None:
            value = self.context[0]
        return value

    @property
    def interiors(self) -> Any:
        """Get polygon interiors."""
        value = getattr(self.context, "interiors", None)
        if value is None:
            value = self.context[1:]
        return value


def polygon_path(polygon: BaseGeometry | ArrayLike | dict[str, Any]) -> Path:
    """Constructs a compound matplotlib path from a Shapely (multi)polygon, an array of them, or a
    GeoJSON-like (multi)polygon"""
    from matplotlib.path import Path

    if isinstance(polygon, dict) or (
        hasattr(polygon, "__geo_interface__") and not isinstance(polygon, shapely.Geometry)
    ):
        polygon = shapely.geometry.shape(polygon)
    rings = shapely.get_rings(shapely.get_parts(polygon))
    vertices, ring_index = shapely.get_coordinates(rings, return_index=True)
    # every ring starts with a MOVETO, all other vertices are LINETOs
    codes = full(len(vertices), Path.LINETO, dtype=Path.code_type)
    codes[flatnonzero(diff(ring_index, prepend=-1))] = Path.MOVETO
    return Path(vertices, codes)


def polygon_paths(polygons: ArrayLike) -> list[Path]:
    """Constructs one matplotlib path per (non-empty) Shapely polygon in ``polygons``."""
    from matplotlib.path import Path

    if not shapely.get_num_interior_rings(polygons).any():
        # no holes: every path is one ring, which is what a path without codes means; reading the
        # coordinates straight from the polygons avoids copying rings and per-vertex indices
        vertices = shapely.get_coordinates(polygons)
        counts = shapely.get_num_coordinates(polygons)
        if len(counts) and (counts == counts[0]).all():
            return [Path(v) for v in vertices.reshape(len(counts), counts[0], 2)]
        return [Path(v) for v in split(vertices, cumsum(counts)[:-1])]
    rings, ring_polygon = shapely.get_rings(polygons, return_index=True)
    vertices, vertex_ring = shapely.get_coordinates(rings, return_index=True)
    codes = full(len(vertices), Path.LINETO, dtype=Path.code_type)
    codes[flatnonzero(diff(vertex_ring, prepend=-1))] = Path.MOVETO
    cuts = flatnonzero(diff(ring_polygon[vertex_ring])) + 1
    return [Path(v, c) for v, c in zip(split(vertices, cuts), split(codes, cuts))]


def polygon_patch(polygon: BaseGeometry | ArrayLike | dict[str, Any], **kwargs: Any) -> PathPatch:
    """Constructs a matplotlib patch from a geometric object

    The ``polygon`` may be a Shapely (multi)polygon with or without holes, an array of them, or a
    GeoJSON-like (multi)polygon.
    The ``kwargs`` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example
    -------
    >>> b = Point(0, 0).buffer(1.0) # doctest: +SKIP
    >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5) # doctest: +SKIP
    >>> axis.add_patch(patch) # doctest: +SKIP

    """
    from matplotlib.patches import PathPatch

    return PathPatch(polygon_path(polygon), **kwargs)


"""End descartes modification
================================================================================================="""
