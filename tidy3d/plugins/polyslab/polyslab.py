"""Divide a complex polyslab where self-intersecting polygon can occur during extrusion."""

from ...components.geometry.polyslab import ComplexPolySlabBase
from ...components.medium import MediumType
from ...components.structure import Structure


class ComplexPolySlab(ComplexPolySlabBase):
    """Interface for dividing a complex polyslab where self-intersecting polygon can
    occur during extrusion.

    Example
    -------
    >>> import tidy3d as td
    >>> from tidy3d.plugins.polyslab import ComplexPolySlab
    >>> vertices = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0.9), (0, 0.11))
    >>> p = ComplexPolySlab(vertices=vertices, axis=2, slab_bounds=(0, 1), sidewall_angle=0.785)
    >>> # To obtain the divided polyslabs, there are two approaches:
    >>> # 1) a list of divided polyslabs
    >>> geo_list = p.sub_polyslabs
    >>> # 2) geometry group containing the divided polyslabs
    >>> geo_group = p.geometry_group
    >>> # Or directly obtain the structure with a user-specified medium
    >>> mat = td.Medium(permittivity=2)
    >>> structure = p.to_structure(mat)
    ...

    Note
    ----
    This version is limited to neighboring vertex-vertex crossing type of
    self-intersecting events. Extension to cover all types of self-intersecting
    events is expected in the future.

    The algorithm is as follows (for the convenience of illustration,
    let's consider the reference plane to lie at the bottom of the polyslab),

    1. Starting from the reference plane, find out the critical
    extrusion distance for the first vertices degeneracy
    event when marching towards the top of the polyslab;

    2. Construct a sub-polyslab whose base is the polygon at
    the reference plane and height to be the critical
    extrusion distance;

    3. At the critical extrusion distance, constructing a new polygon
    that keeps only one of the degenerate vertices;

    4. Set the reference plane to the position of the new polygon,
    and  repeating 1-3 to construct sub-polyslabs until reaching
    the top of the polyslab, or all vertices collapsed into a 1D curve
    or a 0D point.
    """

    def to_structure(self, medium: MediumType) -> Structure:
        """Construct a structure containing a user-specified medium
        and a GeometryGroup made of all the divided PolySlabs from this object.

        Parameters
        ----------
        medium : :class:`.MediumType`
            Medium for the complex polyslab.

        Returns
        -------
        :class:`.Structure`
            The structure containing all divided polyslabs made of a user-specified
            medium.
        """
        return Structure(geometry=self.geometry_group, medium=medium)
