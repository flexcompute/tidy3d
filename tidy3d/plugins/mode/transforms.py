"""Coordinate transformations.

The Jacobian of a transformation from coordinates r = (x, y, z) into coordinates
r' = (u, v, w) is defined as J_ij = dr'_i/dr_j. Here, z and w are the propagation axes in the
original and transformed planes, respectively, and the coords are only provided in (x, y) and
transformed to (u, v). The Yee grid positions also have to be taken into account. The Jacobian
for the transformation of eps and E is evaluated at the r' positions of E-field components.
Similarly, the jacobian for mu and H is evaluated at the r' positions of H-field components.
Currently, the half-step offset in w is ignored, which should be a pretty good approximation."""

from ...components.mode.transforms import angled_transform, radial_transform

_ = [radial_transform, angled_transform]
