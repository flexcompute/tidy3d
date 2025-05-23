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

try:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
except ImportError:
    pass
from numpy import array, concatenate, ones


class Polygon:
    """Adapt Shapely polygons to a common interface"""

    def __init__(self, context):
        if isinstance(context, dict):
            self.context = context["coordinates"]
        else:
            self.context = context

    @property
    def exterior(self):
        """Get polygon exterior."""
        return getattr(self.context, "exterior", None) or self.context[0]

    @property
    def interiors(self):
        """Get polygon interiors."""
        value = getattr(self.context, "interiors", None)
        if value is None:
            value = self.context[1:]
        return value


def polygon_path(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(obj):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(obj, "coords", None) or obj)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    ptype = polygon.geom_type
    if ptype == "Polygon":
        polygon = [Polygon(polygon)]
    elif ptype == "MultiPolygon":
        polygon = [Polygon(p) for p in polygon.geoms]

    vertices = concatenate(
        [
            concatenate(
                [array(t.exterior.coords)[:, :2]] + [array(r.coords)[:, :2] for r in t.interiors]
            )
            for t in polygon
        ]
    )
    codes = concatenate(
        [concatenate([coding(t.exterior)] + [coding(r) for r in t.interiors]) for t in polygon]
    )

    return Path(vertices, codes)


def polygon_patch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The ``polygon`` may be a Shapely or GeoJSON-like object with or without holes.
    The ``kwargs`` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example
    -------
    >>> b = Point(0, 0).buffer(1.0) # doctest: +SKIP
    >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5) # doctest: +SKIP
    >>> axis.add_patch(patch) # doctest: +SKIP

    """
    return PathPatch(polygon_path(polygon), **kwargs)


"""End descartes modification
================================================================================================="""
