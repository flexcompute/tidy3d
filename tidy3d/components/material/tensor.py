"""Symmetric material tensors given by principal values and a rotation."""

from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.transformation import RotationAroundAxis, RotationType


class AbstractPrincipalAxesTensor(Tidy3dBaseModel):
    """Symmetric tensor given by principal values ``xx``, ``yy``, ``zz`` along local axes,
    plus an optional ``rotation`` that orients those axes in the global frame.

    The global tensor is :math:`R \\, \\mathrm{diag}(xx, yy, zz) \\, R^T`, with :math:`R` the
    active rotation of ``rotation``, so local axis :math:`i` points along :math:`R e_i`. Subclasses
    declare the principal values with their units and bounds.
    """

    xx: float
    yy: float
    zz: float
    rotation: RotationType | None = Field(
        default=None,
        title="Rotation",
        description="Optional rotation orienting the principal axes ``(xx, yy, zz)`` in the "
        "global frame. When ``None`` the principal axes coincide with the global axes "
        "(diagonal tensor).",
    )

    @cached_property
    def matrix(self) -> np.ndarray:
        """The 3x3 tensor in the global frame."""
        diag = np.diag([self.xx, self.yy, self.zz])
        return self.rotation.rotate_tensor(diag) if self.rotation is not None else diag

    def to_tensor(self) -> tuple[float, float, float, float, float, float]:
        """Resolve to the six packed symmetric tensor components ``[xx, yy, zz, xy, xz, yz]``
        in the global frame.

        This order is neither Voigt (``[xx, yy, zz, yz, xz, xy]``) nor the solvers' packed order.
        """
        t = self.matrix
        return (
            float(t[0, 0]),
            float(t[1, 1]),
            float(t[2, 2]),
            float(t[0, 1]),
            float(t[0, 2]),
            float(t[1, 2]),
        )

    @staticmethod
    def _diagonalize(
        xx: float, yy: float, zz: float, xy: float, xz: float, yz: float
    ) -> tuple[np.ndarray, RotationAroundAxis | None]:
        """Eigenvalues (ascending) of the symmetric tensor and the rotation mapping the global
        axes onto its principal axes (``None`` when they already coincide)."""
        # scipy is banned at module scope; import lazily for the rotation-matrix decomposition.
        from scipy.spatial.transform import Rotation

        matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        # 'eigh' columns are the principal axes but may form a reflection; flip one to obtain a
        # proper rotation (det +1) that maps onto a 'RotationAroundAxis'.
        if np.linalg.det(eigvecs) < 0:
            eigvecs = eigvecs.copy()
            eigvecs[:, 0] = -eigvecs[:, 0]
        rotvec = Rotation.from_matrix(eigvecs).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        rotation = (
            None
            if np.isclose(angle, 0.0)
            else RotationAroundAxis(axis=tuple(float(c) for c in rotvec / angle), angle=angle)
        )
        return eigvals, rotation
