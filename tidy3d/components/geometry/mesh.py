"""Mesh-defined geometry."""
from __future__ import annotations

from abc import ABC
from typing import List, Tuple, Union, Optional

import pydantic as pydantic
import numpy as np

from ..base import cached_property
from ..types import Ax, Bound, Shapely
from ..viz import add_ax_if_none, equal_aspect
from ...log import log
from ...exceptions import ValidationError, DataError
from ...constants import fp_eps, inf
from ..data.dataset import TriangleMeshDataset
from ..data.data_array import TriangleMeshDataArray, DATA_ARRAY_MAP

from . import base

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except Exception:
    TRIMESH_AVAILABLE = False

try:
    NETWORKX_RTREE_AVAILABLE = True
except Exception:
    NETWORKX_RTREE_AVAILABLE = False


class TriangleMesh(base.Geometry, ABC):
    """Custom surface geometry given by a triangle mesh, as in the STL file format.

    Example
    -------
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    >>> stl_geom = TriangleMesh.from_vertices_faces(vertices, faces)
    """

    mesh_dataset: Optional[TriangleMeshDataset] = pydantic.Field(
        ...,
        title="Surface mesh data",
        description="Surface mesh data.",
    )

    @classmethod
    def _check_trimesh_library(cls):
        """Check if the trimesh package is imported."""
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "The package 'trimesh' was not found. Please install the 'trimesh' "
                "dependencies to use 'TriangleMesh'. For example: "
                "pip install -r requirements/trimesh.txt."
            )

    @pydantic.root_validator(pre=True)
    def _validate_trimesh_library(cls, values):
        """Check if the trimesh package is imported as a validator."""
        cls._check_trimesh_library()
        return values

    @pydantic.validator("mesh_dataset", pre=True, always=True)
    def _warn_if_none(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Warn if the Dataset fails to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning("Loading 'mesh_dataset' without data.")
                return None
        return val

    @pydantic.validator("mesh_dataset", always=True)
    def _check_mesh(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Check that the mesh is valid."""
        if val is None:
            return None
        mesh = cls._triangles_to_trimesh(val.surface_mesh)
        if not all(np.array(mesh.area_faces) > fp_eps):
            raise ValidationError(
                "The provided mesh has triangles with zero area. "
                "Consider using 'trimesh.Trimesh.remove_degenerate_faces' to fix this."
            )
        if not mesh.is_winding_consistent:
            log.warning(
                "The provided mesh does not have consistent winding (face orientations). "
                "This can lead to incorrect permittivity distributions. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You could repair the mesh manually, or try 'trimesh.repair.fix_winding'."
            )
        if not mesh.is_watertight:
            log.warning(
                "The provided mesh is not watertight and may require manual repair. "
                "Non-watertight meshes can generate incorrect permittivity distributions. "
                "They also cause problems with plotting. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You can see which faces are broken using 'trimesh.repair.broken_faces'."
            )

        return val

    @classmethod
    def from_stl(
        cls,
        filename: str,
        scale: float = 1.0,
        origin: Tuple[float, float, float] = (0, 0, 0),
        solid_index: int = None,
        **kwargs,
    ) -> Union[TriangleMesh, base.GeometryGroup]:
        """Load a :class:`.TriangleMesh` directly from an STL file.
        The ``solid_index`` parameter can be used to select a single solid from the file.
        Otherwise, if the file contains a single solid, it will be loaded as a
        :class:`.TriangleMesh`; if the file contains multiple solids,
        they will all be loaded as a :class:`.GeometryGroup`.

        Parameters
        ----------
        filename : str
            The name of the STL file containing the surface geometry mesh data.
        scale : float = 1.0
            The length scale for the loaded geometry (um).
            For example, a scale of 10.0 means that a vertex (1, 0, 0) will be placed at
            x = 10 um.
        origin : Tuple[float, float, float] = (0, 0, 0)
            The origin of the loaded geometry, in units of ``scale``.
            Translates from (0, 0, 0) to this point after applying the scaling.
        solid_index : int = None
            If set, read a single solid with this index from the file.

        Returns
        -------
        Union[:class:`.TriangleMesh`, :class:`.GeometryGroup`]
            The geometry or geometry group from the file.
        """

        def process_single(mesh: trimesh.Trimesh) -> TriangleMesh:
            """Process a single 'trimesh.Trimesh' using scale and origin."""
            mesh.apply_scale(scale)
            mesh.apply_translation(origin)
            return cls.from_trimesh(mesh)

        cls._check_trimesh_library()

        scene = trimesh.load(filename, **kwargs)
        meshes = []
        if isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        elif isinstance(scene, trimesh.Scene):
            meshes = scene.dump()
        else:
            raise ValidationError(
                "Invalid trimesh type in file. Supported types are 'trimesh.Trimesh' "
                "and 'trimesh.Scene'."
            )

        if solid_index is None:
            if isinstance(scene, trimesh.Trimesh):
                return process_single(scene)
            if isinstance(scene, trimesh.Scene):
                geoms = [process_single(mesh) for mesh in meshes]
                return base.GeometryGroup(geometries=geoms)

        if solid_index < len(meshes):
            return process_single(meshes[solid_index])
        raise ValidationError("No solid found at 'solid_index' in the stl file.")

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a ``trimesh.Trimesh`` object.

        Parameters
        ----------
        trimesh : ``trimesh.Trimesh``
            The Trimesh object containing the surface geometry mesh data.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the ``trimesh.Trimesh`` provided.
        """
        return cls.from_vertices_faces(mesh.vertices, mesh.faces)

    @classmethod
    def from_triangles(cls, triangles: np.ndarray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a numpy array
        containing the triangles of a surface mesh.

        Parameters
        ----------
        triangles : ``np.ndarray``
            A numpy array of shape (N, 3, 3) storing the triangles of the surface mesh.
            The first index labels the triangle, the second index labels the vertex
            within a given triangle, and the third index is the coordinate (x, y, or z).

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the triangles provided.

        """
        triangles = np.array(triangles)
        if len(triangles.shape) != 3 or triangles.shape[1] != 3 or triangles.shape[2] != 3:
            raise ValidationError(
                f"Provided 'triangles' must be an N x 3 x 3 array, given {triangles.shape}."
            )
        num_faces = len(triangles)
        coords = dict(
            face_index=np.arange(num_faces),
            vertex_index=np.arange(3),
            axis=np.arange(3),
        )
        vertices = TriangleMeshDataArray(triangles, coords=coords)
        mesh_dataset = TriangleMeshDataset(surface_mesh=vertices)
        return TriangleMesh(mesh_dataset=mesh_dataset)

    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: np.ndarray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from numpy arrays containing the data
        of a surface mesh. The first array contains the vertices, and the second array contains
        faces formed from triples of the vertices.

        Parameters
        ----------
        vertices: ``np.ndarray``
            A numpy array of shape (N, 3) storing the vertices of the surface mesh.
            The first index labels the vertex, and the second index is the coordinate
            (x, y, or z).
        faces : ``np.ndarray``
            A numpy array of shape (M, 3) storing the indices of the vertices of each face
            in the surface mesh. The first index labels the face, and the second index
            labels the vertex index within the ``vertices`` array.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the vertices and faces provided.

        """

        cls._check_trimesh_library()

        vertices = np.array(vertices)
        faces = np.array(faces)
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValidationError(
                f"Provided 'vertices' must be an N x 3 array, given {vertices.shape}."
            )
        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValidationError(f"Provided 'faces' must be an M x 3 array, given {faces.shape}.")
        return cls.from_triangles(trimesh.Trimesh(vertices, faces).triangles)

    @classmethod
    def _triangles_to_trimesh(cls, triangles: np.ndarray) -> trimesh.Trimesh:
        """Convert an (N, 3, 3) numpy array of triangles to a ``trimesh.Trimesh``."""

        cls._check_trimesh_library()
        return trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles))

    @cached_property
    def trimesh(self) -> trimesh.Trimesh:
        """A ``trimesh.Trimesh`` object representing the custom surface mesh geometry."""
        return self._triangles_to_trimesh(self.triangles)

    @cached_property
    def triangles(self) -> np.ndarray:
        """The triangles of the surface mesh as an ``np.ndarray``."""
        if self.mesh_dataset is None:
            raise DataError("Can't get triangles as 'mesh_dataset' is None.")
        return self.mesh_dataset.surface_mesh.to_numpy()

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        # currently ignores bounds
        return self.trimesh.area

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        # currently ignores bounds
        return self.trimesh.volume

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        if self.mesh_dataset is None:
            return ((-inf, -inf, -inf), (inf, inf, inf))
        return self.trimesh.bounds

    def intersections_plane(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Shapely]:
        """Returns list of shapely geomtries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        if self.mesh_dataset is None:
            return []

        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)

        origin = self.unpop_axis(position, (0, 0), axis=axis)
        normal = self.unpop_axis(1, (0, 0), axis=axis)

        mesh = self.trimesh

        if not NETWORKX_RTREE_AVAILABLE:
            raise ImportError(
                "'TriangleMesh.intersections_plane' requires 'networkx' and 'rtree'. "
                "Please install the 'trimesh' dependencies. For example: "
                "pip install -r requirements/trimesh.txt."
            )

        section = mesh.section(plane_origin=origin, plane_normal=normal)

        if section is None:
            return []

        # homogeneous transformation matrix to map to xy plane
        mapping = np.eye(4)

        # translate to origin
        mapping[3, :3] = -np.array(origin)

        # permute so normal is aligned with z axis
        # and (y, z), (x, z), resp. (x, y) are aligned with (x, y)
        identity = np.eye(3)
        permutation = self.unpop_axis(identity[2], identity[0:2], axis=axis)
        mapping[:3, :3] = np.array(permutation).T

        section2d, _ = section.to_planar(to_2D=mapping)
        return list(section2d.polygons_full)

    def inside(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """

        arrays = tuple(map(np.array, (x, y, z)))
        self._ensure_equal_shape(*arrays)
        inside = np.zeros((arrays[0].size,), dtype=bool)
        arrays_flat = map(np.ravel, arrays)
        arrays_stacked = np.stack(tuple(arrays_flat), axis=-1)
        inside = self.trimesh.contains(arrays_stacked)
        return inside.reshape(arrays[0].shape)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:
        """Plot geometry cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        log.warning(
            "Plotting a 'TriangleMesh' may give inconsistent results "
            "if the mesh is not unionized. We recommend unionizing all meshes before import. "
            "A 'PermittivityMonitor' can be used to check that the mesh is loaded correctly."
        )

        return base.Geometry.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)
