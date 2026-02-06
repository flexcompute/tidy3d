"""Mesh-defined geometry."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from autograd import numpy as anp
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, field_validator, model_validator

from tidy3d.components.autograd import get_static
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DATA_ARRAY_MAP, TriangleMeshDataArray
from tidy3d.components.data.dataset import TriangleMeshDataset
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.config import config
from tidy3d.constants import fp_eps, inf
from tidy3d.exceptions import DataError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import verify_packages_import

from . import base

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any, Callable, Literal, Optional, Union

    from trimesh import Trimesh

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.types import Ax, Bound, Coordinate, MatrixReal4x4, Shapely

AREA_SIZE_THRESHOLD = 1e-36


class TriangleMesh(base.Geometry, ABC):
    """Custom surface geometry given by a triangle mesh, as in the STL file format.

    Example
    -------
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    >>> stl_geom = TriangleMesh.from_vertices_faces(vertices, faces)
    """

    mesh_dataset: Optional[TriangleMeshDataset] = Field(
        None,
        title="Surface mesh data",
        description="Surface mesh data.",
    )

    _no_nans_mesh = validate_no_nans("mesh_dataset")
    _barycentric_samples: dict[int, NDArray] = PrivateAttr(default_factory=dict)

    @verify_packages_import(["trimesh"])
    @model_validator(mode="before")
    @classmethod
    def _validate_trimesh_library(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Check if the trimesh package is imported as a validator."""
        return data

    @field_validator("mesh_dataset", mode="before")
    @classmethod
    def _warn_if_none(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Warn if the Dataset fails to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning("Loading 'mesh_dataset' without data.")
                return None
        return val

    @field_validator("mesh_dataset")
    @classmethod
    def _check_mesh(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Check that the mesh is valid."""
        if val is None:
            return None

        import trimesh

        surface_mesh = val.surface_mesh
        triangles = get_static(surface_mesh.data)
        mesh = cls._triangles_to_trimesh(triangles)
        if not all(np.array(mesh.area_faces) > AREA_SIZE_THRESHOLD):
            old_tol = trimesh.tol.merge
            trimesh.tol.merge = np.sqrt(2 * AREA_SIZE_THRESHOLD)
            new_mesh = mesh.process(validate=True)
            trimesh.tol.merge = old_tol
            val = TriangleMesh.from_trimesh(new_mesh).mesh_dataset
            log.warning(
                f"The provided mesh has triangles with near zero area < {AREA_SIZE_THRESHOLD}. "
                "Triangles which have one edge of their 2D oriented bounding box shorter than "
                f"'sqrt(2*{AREA_SIZE_THRESHOLD}) are being automatically removed.'"
            )
            if not all(np.array(new_mesh.area_faces) > AREA_SIZE_THRESHOLD):
                raise ValidationError(
                    f"The provided mesh has triangles with near zero area < {AREA_SIZE_THRESHOLD}. "
                    "The automatic removal of these triangles has failed. You can try "
                    "using numpy-stl's 'from_file' import with 'remove_empty_areas' set "
                    "to True and a suitable 'AREA_SIZE_THRESHOLD' to remove them."
                )
        if not mesh.is_watertight:
            log.warning(
                "The provided mesh is not watertight. "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fill_holes', which attempts to repair the mesh. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You can see which faces are broken using 'trimesh.repair.broken_faces'."
            )
        if not mesh.is_winding_consistent:
            log.warning(
                "The provided mesh does not have consistent winding (face orientations). "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fix_winding', which attempts to repair the mesh. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
            )
        if not mesh.is_volume:
            log.warning(
                "The provided mesh does not represent a valid volume, possibly due to "
                "incorrect normal vector orientation. "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fix_normals', "
                "which attempts to fix the normals to be consistent and outward-facing. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct."
            )

        return val

    @verify_packages_import(["trimesh"])
    def fix_winding(self) -> TriangleMesh:
        """Try to fix winding in the mesh."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fix_winding(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @verify_packages_import(["trimesh"])
    def fill_holes(self) -> TriangleMesh:
        """Try to fill holes in the mesh. Can be used to repair non-watertight meshes."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fill_holes(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @verify_packages_import(["trimesh"])
    def fix_normals(self) -> TriangleMesh:
        """Try to fix normals to be consistent and outward-facing."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fix_normals(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_stl(
        cls,
        filename: str,
        scale: float = 1.0,
        origin: tuple[float, float, float] = (0, 0, 0),
        solid_index: Optional[int] = None,
        **kwargs: Any,
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
        origin : tuple[float, float, float] = (0, 0, 0)
            The origin of the loaded geometry, in units of ``scale``.
            Translates from (0, 0, 0) to this point after applying the scaling.
        solid_index : int = None
            If set, read a single solid with this index from the file.

        Returns
        -------
        Union[:class:`.TriangleMesh`, :class:`.GeometryGroup`]
            The geometry or geometry group from the file.
        """
        import trimesh

        from tidy3d.components.types.third_party import TrimeshType

        def process_single(mesh: TrimeshType) -> TriangleMesh:
            """Process a single 'trimesh.Trimesh' using scale and origin."""
            mesh.apply_scale(scale)
            mesh.apply_translation(origin)
            return cls.from_trimesh(mesh)

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

    @verify_packages_import(["trimesh"])
    def to_stl(
        self,
        filename: PathLike,
        *,
        binary: bool = True,
    ) -> None:
        """Export this TriangleMesh to an STL file.

        Parameters
        ----------
        filename : str
            Output STL filename.
        binary : bool = True
            Whether to write binary STL. Set False for ASCII STL.
        """
        triangles = get_static(self.mesh_dataset.surface_mesh.data)
        mesh = self._triangles_to_trimesh(triangles)

        file_type = "stl" if binary else "stl_ascii"
        mesh.export(file_obj=filename, file_type=file_type)

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_trimesh(cls, mesh: Trimesh) -> TriangleMesh:
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
    def from_triangles(cls, triangles: NDArray) -> TriangleMesh:
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
        triangles = anp.array(triangles)
        if len(triangles.shape) != 3 or triangles.shape[1] != 3 or triangles.shape[2] != 3:
            raise ValidationError(
                f"Provided 'triangles' must be an N x 3 x 3 array, given {triangles.shape}."
            )
        num_faces = len(triangles)
        coords = {
            "face_index": np.arange(num_faces),
            "vertex_index": np.arange(3),
            "axis": np.arange(3),
        }
        vertices = TriangleMeshDataArray(triangles, coords=coords)
        mesh_dataset = TriangleMeshDataset(surface_mesh=vertices)
        return TriangleMesh(mesh_dataset=mesh_dataset)

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_vertices_faces(cls, vertices: NDArray, faces: NDArray) -> TriangleMesh:
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
        import trimesh

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
    @verify_packages_import(["trimesh"])
    def _triangles_to_trimesh(
        cls, triangles: NDArray
    ) -> Trimesh:  # -> We need to get this out of the classes and into functional methods operating on a class (maybe still referenced to the class)
        """Convert an (N, 3, 3) numpy array of triangles to a ``trimesh.Trimesh``."""
        import trimesh

        # ``triangles`` may contain autograd ``ArrayBox`` entries when differentiating
        # geometry parameters. ``trimesh`` expects plain ``float`` values, so strip any
        # tracing information before constructing the mesh.
        triangles = get_static(anp.array(triangles))
        return trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles))

    @classmethod
    def from_height_grid(
        cls,
        axis: Ax,
        direction: Literal["-", "+"],
        base: float,
        grid: tuple[np.ndarray, np.ndarray],
        height: NDArray,
    ) -> TriangleMesh:
        """Construct a TriangleMesh object from grid based height information.

        Parameters
        ----------
        axis : Ax
            Axis of extrusion.
        direction : Literal["-", "+"]
            Direction of extrusion.
        base : float
            Coordinate of the base surface along the geometry's axis.
        grid : Tuple[np.ndarray, np.ndarray]
            Tuple of two one-dimensional arrays representing the sampling grid (XY, YZ, or ZX
            corresponding to values of axis)
        height : np.ndarray
            Height values sampled on the given grid. Can be 1D (raveled) or 2D (matching grid mesh).

        Returns
        -------
        TriangleMesh
            The resulting TriangleMesh geometry object.
        """

        x_coords = grid[0]
        y_coords = grid[1]

        nx = len(x_coords)
        ny = len(y_coords)
        nt = nx * ny

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")

        sign = 1
        if direction == "-":
            sign = -1

        flat_height = np.ravel(height)
        if flat_height.shape[0] != nt:
            raise ValueError(
                f"Shape of flattened height array {flat_height.shape} does not match "
                f"the number of grid points {nt}."
            )

        if np.any(flat_height < 0):
            raise ValueError("All height values must be non-negative.")

        max_h = np.max(flat_height)
        min_h_clip = fp_eps * max_h
        flat_height = np.clip(flat_height, min_h_clip, inf)

        vertices_raw_list = [
            [np.ravel(x_mesh), np.ravel(y_mesh), base + sign * flat_height],  # Alpha surface
            [np.ravel(x_mesh), np.ravel(y_mesh), base * np.ones(nt)],
        ]

        if direction == "-":
            vertices_raw_list = vertices_raw_list[::-1]

        vertices = np.hstack(vertices_raw_list).T
        vertices = np.roll(vertices, shift=axis - 2, axis=1)

        q0 = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        q1 = (np.arange(1, nx)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        q2 = (np.arange(1, nx)[:, None] * ny + np.arange(1, ny)[None, :]).ravel()
        q3 = (np.arange(nx - 1)[:, None] * ny + np.arange(1, ny)[None, :]).ravel()

        q0_b = nt + q0
        q1_b = nt + q1
        q2_b = nt + q2
        q3_b = nt + q3

        top_quads = np.stack((q0, q1, q2, q3), axis=-1)
        bottom_quads = np.stack((q0_b, q3_b, q2_b, q1_b), axis=-1)

        s1_q0 = (0 * ny + np.arange(ny - 1)).ravel()
        s1_q1 = (0 * ny + np.arange(1, ny)).ravel()
        s1_q2 = (nt + 0 * ny + np.arange(1, ny)).ravel()
        s1_q3 = (nt + 0 * ny + np.arange(ny - 1)).ravel()
        side1_quads = np.stack((s1_q0, s1_q1, s1_q2, s1_q3), axis=-1)

        s2_q0 = ((nx - 1) * ny + np.arange(ny - 1)).ravel()
        s2_q1 = (nt + (nx - 1) * ny + np.arange(ny - 1)).ravel()
        s2_q2 = (nt + (nx - 1) * ny + np.arange(1, ny)).ravel()
        s2_q3 = ((nx - 1) * ny + np.arange(1, ny)).ravel()
        side2_quads = np.stack((s2_q0, s2_q1, s2_q2, s2_q3), axis=-1)

        s3_q0 = (np.arange(nx - 1) * ny + 0).ravel()
        s3_q1 = (nt + np.arange(nx - 1) * ny + 0).ravel()
        s3_q2 = (nt + np.arange(1, nx) * ny + 0).ravel()
        s3_q3 = (np.arange(1, nx) * ny + 0).ravel()
        side3_quads = np.stack((s3_q0, s3_q1, s3_q2, s3_q3), axis=-1)

        s4_q0 = (np.arange(nx - 1) * ny + ny - 1).ravel()
        s4_q1 = (np.arange(1, nx) * ny + ny - 1).ravel()
        s4_q2 = (nt + np.arange(1, nx) * ny + ny - 1).ravel()
        s4_q3 = (nt + np.arange(nx - 1) * ny + ny - 1).ravel()
        side4_quads = np.stack((s4_q0, s4_q1, s4_q2, s4_q3), axis=-1)

        all_quads = np.vstack(
            (top_quads, bottom_quads, side1_quads, side2_quads, side3_quads, side4_quads)
        )

        triangles_list = [
            np.stack((all_quads[:, 0], all_quads[:, 1], all_quads[:, 3]), axis=-1),
            np.stack((all_quads[:, 3], all_quads[:, 1], all_quads[:, 2]), axis=-1),
        ]
        tri_faces = np.vstack(triangles_list)

        return cls.from_vertices_faces(vertices=vertices, faces=tri_faces)

    @classmethod
    def from_height_function(
        cls,
        axis: Ax,
        direction: Literal["-", "+"],
        base: float,
        center: tuple[float, float],
        size: tuple[float, float],
        grid_size: tuple[int, int],
        height_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> TriangleMesh:
        """Construct a TriangleMesh object from analytical expression of height function.
        The height function should be vectorized to accept 2D meshgrid arrays.

        Parameters
        ----------
        axis : Ax
            Axis of extrusion.
        direction : Literal["-", "+"]
            Direction of extrusion.
        base : float
            Coordinate of the base rectangle along the geometry's axis.
        center : Tuple[float, float]
            Center of the base rectangle in the plane perpendicular to the extrusion axis
            (XY, YZ, or ZX corresponding to values of axis).
        size : Tuple[float, float]
            Size of the base rectangle in the plane perpendicular to the extrusion axis
            (XY, YZ, or ZX corresponding to values of axis).
        grid_size : Tuple[int, int]
            Number of grid points for discretization of the base rectangle
            (XY, YZ, or ZX corresponding to values of axis).
        height_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Vectorized function to compute height values from 2D meshgrid coordinate arrays.
            It should take two ndarrays (x_mesh, y_mesh) and return an ndarray of heights.

        Returns
        -------
        TriangleMesh
            The resulting TriangleMesh geometry object.
        """
        x_lin = np.linspace(center[0] - 0.5 * size[0], center[0] + 0.5 * size[0], grid_size[0])
        y_lin = np.linspace(center[1] - 0.5 * size[1], center[1] + 0.5 * size[1], grid_size[1])

        x_mesh, y_mesh = np.meshgrid(x_lin, y_lin, indexing="ij")

        height_values = height_func(x_mesh, y_mesh)

        if not (isinstance(height_values, np.ndarray) and height_values.shape == x_mesh.shape):
            raise ValueError(
                f"The 'height_func' must return a NumPy array with shape {x_mesh.shape}, "
                f"but got shape {getattr(height_values, 'shape', type(height_values))}."
            )

        return cls.from_height_grid(
            axis=axis,
            direction=direction,
            base=base,
            grid=(x_lin, y_lin),
            height=height_values,
        )

    @cached_property
    @verify_packages_import(["trimesh"])
    def trimesh(
        self,
    ) -> Trimesh:  # -> We need to get this out of the classes and into functional methods operating on a class (maybe still referenced to the class)
        """A ``trimesh.Trimesh`` object representing the custom surface mesh geometry."""
        return self._triangles_to_trimesh(self.triangles)

    @cached_property
    def triangles(self) -> np.ndarray:
        """The triangles of the surface mesh as an ``np.ndarray``."""
        if self.mesh_dataset is None:
            raise DataError("Can't get triangles as 'mesh_dataset' is None.")
        return np.asarray(get_static(self.mesh_dataset.surface_mesh.data))

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
        tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        if self.mesh_dataset is None:
            return ((-inf, -inf, -inf), (inf, inf, inf))
        return self.trimesh.bounds

    def intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        cleanup: bool = True,
        quad_segs: Optional[int] = None,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for TriangleMesh.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        section = self.trimesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return []
        path, _ = section.to_2D(to_2D=to_2D)
        return path.polygons_full

    def intersections_plane(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        cleanup: bool = True,
        quad_segs: Optional[int] = None,
    ) -> list[Shapely]:
        """Returns list of shapely geometries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for TriangleMesh.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
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

        try:
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

            section2d, _ = section.to_2D(to_2D=mapping)
            return list(section2d.polygons_full)

        except ValueError as e:
            if not mesh.is_watertight:
                log.warning(
                    "Unable to compute 'TriangleMesh.intersections_plane' "
                    "because the mesh was not watertight. Using bounding box instead. "
                    "This may be overly strict; consider using 'TriangleMesh.fill_holes' "
                    "to repair the non-watertight mesh."
                )
            else:
                log.warning(
                    "Unable to compute 'TriangleMesh.intersections_plane'. "
                    "Using bounding box instead."
                )
            log.warning(f"Error encountered: {e}")
            return self.bounding_box.intersections_plane(x=x, y=y, z=z, cleanup=cleanup)

    def inside(self, x: NDArray, y: NDArray, z: NDArray) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`~tidy3d.Geometry`, and ``False`` otherwise.

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
        arrays_flat = map(np.ravel, arrays)
        arrays_stacked = np.stack(tuple(arrays_flat), axis=-1)
        inside = self.trimesh.contains(arrays_stacked)
        return inside.reshape(arrays[0].shape)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **patch_kwargs: Any,
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

    def _compute_derivatives_via_mesh(
        self,
        derivative_info: DerivativeInfo,
        clip_operation: Optional[base.ClipOperationContext] = None,
    ) -> AutogradFieldMap:
        return self._compute_derivatives(derivative_info, clip_operation=clip_operation)

    def _compute_derivatives(
        self,
        derivative_info: DerivativeInfo,
        clip_operation: Optional[base.ClipOperationContext] = None,
    ) -> AutogradFieldMap:
        """Compute adjoint derivatives for a ``TriangleMesh`` geometry."""
        vjps: AutogradFieldMap = {}

        if not self.mesh_dataset:
            raise DataError("Can't compute derivatives without mesh data.")

        valid_paths = {("mesh_dataset", "surface_mesh")}
        for path in derivative_info.paths:
            if path not in valid_paths:
                raise ValueError(f"No derivative defined w.r.t. 'TriangleMesh' field '{path}'.")

        if ("mesh_dataset", "surface_mesh") not in derivative_info.paths:
            return vjps

        triangles = np.asarray(self.triangles, dtype=config.adjoint.gradient_dtype_float)

        # early exit if geometry is completely outside simulation bounds
        sim_min, sim_max = map(np.asarray, derivative_info.simulation_bounds)
        mesh_min, mesh_max = map(np.asarray, self.bounds)
        if np.any(mesh_max < sim_min) or np.any(mesh_min > sim_max):
            log.warning(
                "'TriangleMesh' lies completely outside the simulation domain.",
                log_once=True,
            )
            zeros = np.zeros_like(triangles)
            vjps[("mesh_dataset", "surface_mesh")] = zeros
            return vjps

        # gather surface samples within the simulation bounds
        dx = derivative_info.adaptive_vjp_spacing()
        samples = self._collect_surface_samples(
            triangles=triangles,
            spacing=dx,
            sim_min=sim_min,
            sim_max=sim_max,
        )
        if clip_operation is not None:
            samples = self._apply_clip_filters(samples, clip_operation)

        if samples["points"].shape[0] == 0:
            zeros = np.zeros_like(triangles)
            vjps[("mesh_dataset", "surface_mesh")] = zeros
            return vjps

        interpolators = derivative_info.interpolators
        if interpolators is None:
            interpolators = derivative_info.create_interpolators(
                dtype=config.adjoint.gradient_dtype_float
            )

        g = derivative_info.evaluate_gradient_at_points(
            samples["points"],
            samples["normals"],
            samples["perps1"],
            samples["perps2"],
            interpolators,
        )

        # accumulate per-vertex contributions using barycentric weights
        weights = (samples["weights"] * g).real
        normals = samples["normals"]
        faces = samples["faces"]
        bary = samples["barycentric"]

        contrib_vec = weights[:, None] * normals

        triangle_grads = np.zeros_like(triangles, dtype=config.adjoint.gradient_dtype_float)
        for vertex_idx in range(3):
            scaled = contrib_vec * bary[:, vertex_idx][:, None]
            np.add.at(triangle_grads[:, vertex_idx, :], faces, scaled)

        vjps[("mesh_dataset", "surface_mesh")] = triangle_grads
        return vjps

    def _apply_clip_filters(
        self,
        samples: dict[str, np.ndarray],
        clip_operation: base.ClipOperationContext,
    ) -> dict[str, np.ndarray]:
        """Filter and adjust samples according to clip operation rules."""

        clip_contexts = self._normalize_clip_contexts(clip_operation)
        filtered = samples
        for context in clip_contexts:
            filtered = self._apply_clip_filters_single(filtered, context)
            if filtered["points"].shape[0] == 0:
                return filtered
        return filtered

    @staticmethod
    def _normalize_clip_contexts(
        clip_operation: base.ClipOperationContext,
    ) -> list[base.ClipOperationContext]:
        """Normalize clip contexts into a flat list."""

        if (
            isinstance(clip_operation, tuple)
            and len(clip_operation) == 2
            and isinstance(clip_operation[0], base.ClipOperation)
        ):
            return [clip_operation]

        contexts: list[base.ClipOperationContext] = []
        for entry in clip_operation:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[0], base.ClipOperation)
            ):
                contexts.append(entry)
        if not contexts:
            raise ValueError("Invalid ClipOperation context provided.")
        return contexts

    def _apply_clip_filters_single(
        self,
        samples: dict[str, np.ndarray],
        clip_operation: base.ClipOperationContext,
    ) -> dict[str, np.ndarray]:
        """Filter samples for a single clip operation context."""

        clip_obj, which = clip_operation
        other_geometry = clip_obj._other_geometry(which)
        if not isinstance(other_geometry, TriangleMesh):
            return self._apply_basic_clip_filters(samples, clip_operation)

        clip_geometry = self._prepare_clip_geometry(other_geometry)

        points = np.asarray(samples["points"], dtype=config.adjoint.gradient_dtype_float)
        normals = np.asarray(samples["normals"], dtype=config.adjoint.gradient_dtype_float)
        total_points = points.shape[0]
        if total_points == 0:
            return samples

        shift = max(float(config.adjoint.edge_clip_tolerance), 1e-9)
        probe_points = points - normals * shift
        inside_mask = np.asarray(
            clip_geometry.inside(
                probe_points[:, 0],
                probe_points[:, 1],
                probe_points[:, 2],
            ),
            dtype=bool,
        ).reshape(-1)

        use_mask, flip_mask = clip_obj._clip_masks_from_inside(which, inside_mask)
        if use_mask.size != total_points:
            raise ValueError("ClipOperation sample mask has incorrect shape.")
        if not np.any(use_mask):
            return {key: np.asarray(value[:0]).copy() for key, value in samples.items()}

        filtered = {key: np.asarray(value[use_mask]).copy() for key, value in samples.items()}

        if flip_mask.size != total_points:
            raise ValueError("ClipOperation normal flip mask has incorrect shape.")
        flip_mask = flip_mask[use_mask]
        if np.any(flip_mask):
            flip_signs = np.where(flip_mask[:, None], -1.0, 1.0)
            filtered["normals"] = filtered["normals"] * flip_signs

        return filtered

    def _apply_basic_clip_filters(
        self,
        samples: dict[str, np.ndarray],
        clip_operation: base.ClipOperationContext,
    ) -> dict[str, np.ndarray]:
        """Fallback clip filtering that mirrors the historical behavior."""

        clip_obj, which = clip_operation
        points = samples["points"]
        total_points = points.shape[0]
        if total_points == 0:
            return {key: np.asarray(value[:0]).copy() for key, value in samples.items()}

        use_mask, flip_mask, _ = clip_obj._clip_masks_for_points(which, points)
        if not np.any(use_mask):
            return {key: np.asarray(value[:0]).copy() for key, value in samples.items()}

        filtered = {key: np.asarray(value[use_mask]).copy() for key, value in samples.items()}

        flip_mask = flip_mask[use_mask]
        if np.any(flip_mask):
            flip_signs = np.where(flip_mask[:, None], -1.0, 1.0)
            filtered["normals"] = filtered["normals"] * flip_signs

        return filtered

    @staticmethod
    def _prepare_clip_geometry(other: base.Geometry) -> base.Geometry:
        """Return a TriangleMesh suitable for geometric clipping operations."""

        if not isinstance(other, TriangleMesh):
            return other

        try:
            tri_mesh = other.trimesh
        except Exception:
            return other

        if not tri_mesh.is_volume:
            raise ValueError(
                "ClipOperation requires volume TriangleMesh geometry for clip filtering."
            )

        return other

    def _collect_surface_samples(
        self,
        triangles: NDArray,
        spacing: float,
        sim_min: NDArray,
        sim_max: NDArray,
    ) -> dict[str, np.ndarray]:
        """Deterministic per-triangle sampling used historically."""

        dtype = config.adjoint.gradient_dtype_float
        tol = config.adjoint.edge_clip_tolerance

        sim_min = np.asarray(sim_min, dtype=dtype)
        sim_max = np.asarray(sim_max, dtype=dtype)

        samples = self._SampleAccumulators.empty()

        spacing = max(float(spacing), np.finfo(float).eps)
        triangles_arr = np.asarray(triangles, dtype=dtype)
        if triangles_arr.size == 0:
            return self._empty_sample_result(dtype)

        edges01 = triangles_arr[:, 1, :] - triangles_arr[:, 0, :]
        edges02 = triangles_arr[:, 2, :] - triangles_arr[:, 0, :]
        edges12 = triangles_arr[:, 2, :] - triangles_arr[:, 1, :]
        cross = np.cross(edges01, edges02)
        norm = np.linalg.norm(cross, axis=1)
        areas = 0.5 * norm

        normals = np.zeros((triangles_arr.shape[0], 3), dtype=dtype)
        nonzero_norm = norm > 0.0
        normals[nonzero_norm] = cross[nonzero_norm] / norm[nonzero_norm][:, None]

        edge_tol = np.finfo(dtype).eps
        edge_choice = np.where(
            (np.linalg.norm(edges01, axis=1) > edge_tol)[:, None],
            edges01,
            edges12,
        )
        edge_choice_norm = np.linalg.norm(edge_choice, axis=1)
        has_edge = edge_choice_norm > edge_tol
        perps1 = np.zeros_like(normals)
        perps1[has_edge] = edge_choice[has_edge] / edge_choice_norm[has_edge][:, None]
        perps2_tmp = np.cross(normals, perps1)
        perps2_norm = np.linalg.norm(perps2_tmp, axis=1)
        has_basis = perps2_norm > edge_tol
        perps2 = np.zeros_like(normals)
        perps2[has_basis] = perps2_tmp[has_basis] / perps2_norm[has_basis][:, None]

        usable_mask = (areas > AREA_SIZE_THRESHOLD) & has_edge & has_basis

        sim_extents = sim_max - sim_min
        valid_axes = np.abs(sim_extents) > tol
        collapsed_indices = np.flatnonzero(np.isclose(sim_extents, 0.0, atol=tol))
        collapsed_axis: Optional[int] = None
        plane_value: Optional[float] = None
        if collapsed_indices.size == 1:
            collapsed_axis = int(collapsed_indices[0])
            plane_value = float(sim_min[collapsed_axis])

        if collapsed_axis is not None and plane_value is not None:
            return self._collect_plane_samples(
                triangles_arr=triangles_arr,
                normals=normals,
                perps1=perps1,
                perps2=perps2,
                usable_mask=usable_mask,
                spacing=spacing,
                sim_min=sim_min,
                sim_max=sim_max,
                valid_axes=valid_axes,
                collapsed_axis=collapsed_axis,
                plane_value=plane_value,
                tol=tol,
                dtype=dtype,
            )

        tri_min = np.min(triangles_arr, axis=1)
        tri_max = np.max(triangles_arr, axis=1)
        overlaps = np.all(tri_max >= (sim_min - tol), axis=1) & np.all(
            tri_min <= (sim_max + tol), axis=1
        )
        usable_mask &= overlaps

        fully_inside = (
            usable_mask
            & np.all(tri_min >= (sim_min - tol), axis=1)
            & np.all(tri_max <= (sim_max + tol), axis=1)
        )
        needs_clip = usable_mask & ~fully_inside

        warned = False

        if np.any(fully_inside):
            face_indices = np.flatnonzero(fully_inside)
            tri_inside = triangles_arr[face_indices]
            normals_inside = normals[face_indices]
            perp1_inside = perps1[face_indices]
            perp2_inside = perps2[face_indices]
            areas_inside = areas[face_indices]
            edge_lengths = np.linalg.norm(
                np.stack(
                    (
                        edges01[face_indices],
                        edges02[face_indices],
                        edges12[face_indices],
                    ),
                    axis=1,
                ),
                axis=2,
            )
            subdivisions = self._vectorized_subdivisions(areas_inside, spacing, edge_lengths)
            unique_subdiv, inverse = np.unique(subdivisions, return_inverse=True)

            for group_idx, group_subdiv in enumerate(unique_subdiv):
                group_faces = np.flatnonzero(inverse == group_idx)
                if group_faces.size == 0:
                    continue
                barycentric = self._get_barycentric_samples(int(group_subdiv), dtype)
                self._append_barycentric_group(
                    samples=samples,
                    barycentric=barycentric,
                    triangles=tri_inside[group_faces],
                    normals=normals_inside[group_faces],
                    perps1=perp1_inside[group_faces],
                    perps2=perp2_inside[group_faces],
                    areas=areas_inside[group_faces],
                    face_ids=face_indices[group_faces],
                    dtype=dtype,
                )

        if np.any(needs_clip):
            for face_index in np.flatnonzero(needs_clip):
                tri = triangles_arr[face_index]
                normal = normals[face_index]
                perp1 = perps1[face_index]
                perp2 = perps2[face_index]
                clipped, was_clipped = self._clip_triangle_to_bounds(tri, sim_min, sim_max, tol)
                if was_clipped and not warned:
                    log.warning(
                        "Some triangles from the mesh lie outside the simulation bounds - this may lead to inaccurate gradients."
                    )
                    warned = True
                if not clipped:
                    continue

                for tri_clip in clipped:
                    area_clip, _ = self._triangle_area_and_normal(tri_clip)
                    if area_clip <= AREA_SIZE_THRESHOLD:
                        continue

                    edge_lengths = (
                        np.linalg.norm(tri_clip[1] - tri_clip[0]),
                        np.linalg.norm(tri_clip[2] - tri_clip[1]),
                        np.linalg.norm(tri_clip[0] - tri_clip[2]),
                    )
                    subdivisions = self._subdivision_count(area_clip, spacing, edge_lengths)
                    barycentric_clip = self._get_barycentric_samples(subdivisions, dtype)
                    num_samples = barycentric_clip.shape[0]
                    base_weight = area_clip / num_samples

                    bary_basis = np.stack(
                        [
                            self._barycentric_coordinates(tri, vertex[None, :], tol)[0]
                            for vertex in tri_clip
                        ],
                        axis=0,
                    )
                    bary_orig = barycentric_clip @ bary_basis
                    sample_points = bary_orig @ tri

                    weights_tile = np.full(num_samples, base_weight, dtype=dtype)
                    faces_tile = np.full(num_samples, face_index, dtype=int)
                    samples.append(
                        sample=self._SampleBlock(
                            points=sample_points,
                            normals=np.repeat(normal[None, :], num_samples, axis=0),
                            perps1=np.repeat(perp1[None, :], num_samples, axis=0),
                            perps2=np.repeat(perp2[None, :], num_samples, axis=0),
                            weights=weights_tile,
                            faces=faces_tile,
                            barycentric=bary_orig,
                        )
                    )

        return self._finalize_sample_lists(dtype, samples)

    def _collect_plane_samples(
        self,
        triangles_arr: np.ndarray,
        normals: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
        usable_mask: np.ndarray,
        spacing: float,
        sim_min: np.ndarray,
        sim_max: np.ndarray,
        valid_axes: np.ndarray,
        collapsed_axis: int,
        plane_value: float,
        tol: float,
        dtype: np.dtype,
    ) -> dict[str, np.ndarray]:
        """Sample intersection of triangles with a collapsed-axis plane."""

        samples = self._SampleAccumulators.empty()

        warned = False
        face_indices = np.flatnonzero(usable_mask)
        for face_index in face_indices:
            tri = triangles_arr[face_index]
            normal = normals[face_index]
            perp1 = perps1[face_index]
            perp2 = perps2[face_index]

            segments = self._triangle_plane_segments(
                triangle=tri, axis=collapsed_axis, plane_value=plane_value, tol=tol
            )
            for start, end in segments:
                vec = end - start
                length = float(np.linalg.norm(vec))
                if length <= tol:
                    continue

                subdivisions = max(1, int(np.ceil(length / spacing)))
                t_vals = (np.arange(subdivisions, dtype=dtype) + 0.5) / subdivisions
                sample_points = start[None, :] + t_vals[:, None] * vec[None, :]
                barycentric = self._barycentric_coordinates(tri, sample_points, tol)

                inside_mask = np.ones(sample_points.shape[0], dtype=bool)
                if np.any(valid_axes):
                    min_bound = (sim_min - tol)[valid_axes]
                    max_bound = (sim_max + tol)[valid_axes]
                    coords = sample_points[:, valid_axes]
                    inside_mask = np.all(coords >= min_bound, axis=1) & np.all(
                        coords <= max_bound, axis=1
                    )

                if not np.all(inside_mask) and not warned:
                    log.warning(
                        "Some triangles from the mesh lie outside the simulation bounds - this may lead to inaccurate gradients."
                    )
                    warned = True

                if not np.any(inside_mask):
                    continue

                sample_points = sample_points[inside_mask]
                bary_inside = barycentric[inside_mask]
                n_inside = sample_points.shape[0]

                weights_tile = np.full(n_inside, length / subdivisions, dtype=dtype)
                faces_tile = np.full(n_inside, face_index, dtype=int)
                samples.append(
                    sample=self._SampleBlock(
                        points=sample_points,
                        normals=np.repeat(normal[None, :], n_inside, axis=0),
                        perps1=np.repeat(perp1[None, :], n_inside, axis=0),
                        perps2=np.repeat(perp2[None, :], n_inside, axis=0),
                        weights=weights_tile,
                        faces=faces_tile,
                        barycentric=bary_inside,
                    )
                )

        return self._finalize_sample_lists(dtype, samples)

    @staticmethod
    def _vectorized_subdivisions(
        areas: np.ndarray, spacing: float, edge_lengths: np.ndarray
    ) -> np.ndarray:
        """Compute subdivision counts for many triangles at once."""

        spacing = max(float(spacing), np.finfo(float).eps)
        target = np.sqrt(np.maximum(areas, 0.0))
        area_based = np.ceil(np.sqrt(2.0) * target / spacing)

        max_edge = np.max(edge_lengths, axis=1)
        edge_based = np.ceil(max_edge / spacing)

        subdivisions = np.maximum(area_based, edge_based)
        subdivisions = np.maximum(subdivisions, 1.0)
        return subdivisions.astype(int)

    @staticmethod
    def _empty_sample_result(dtype: np.dtype) -> dict[str, np.ndarray]:
        """Return the default empty sampling dictionary."""

        zeros_vec = np.zeros((0, 3), dtype=dtype)
        zeros_scalar = np.zeros((0,), dtype=dtype)
        zeros_faces = np.zeros((0,), dtype=int)
        return {
            "points": zeros_vec,
            "normals": zeros_vec.copy(),
            "perps1": zeros_vec.copy(),
            "perps2": zeros_vec.copy(),
            "weights": zeros_scalar,
            "faces": zeros_faces,
            "barycentric": zeros_vec.copy(),
        }

    @dataclass
    class _SampleAccumulators:
        """Holds sample arrays before concatenation."""

        points: list[np.ndarray]
        normals: list[np.ndarray]
        perps1: list[np.ndarray]
        perps2: list[np.ndarray]
        weights: list[np.ndarray]
        faces: list[np.ndarray]
        barycentric: list[np.ndarray]

        @classmethod
        def empty(cls) -> TriangleMesh._SampleAccumulators:
            return cls(
                points=[],
                normals=[],
                perps1=[],
                perps2=[],
                weights=[],
                faces=[],
                barycentric=[],
            )

        def append(self, *, sample: TriangleMesh._SampleBlock) -> None:
            self.points.append(sample.points)
            self.normals.append(sample.normals)
            self.perps1.append(sample.perps1)
            self.perps2.append(sample.perps2)
            self.weights.append(sample.weights)
            self.faces.append(sample.faces)
            self.barycentric.append(sample.barycentric)

    @dataclass
    class _SampleBlock:
        """Single sample batch to append."""

        points: np.ndarray
        normals: np.ndarray
        perps1: np.ndarray
        perps2: np.ndarray
        weights: np.ndarray
        faces: np.ndarray
        barycentric: np.ndarray

    @staticmethod
    def _finalize_sample_lists(
        dtype: np.dtype,
        samples: TriangleMesh._SampleAccumulators,
    ) -> dict[str, np.ndarray]:
        """Concatenate accumulated sampling data or return an empty structure."""

        if not samples.points:
            return TriangleMesh._empty_sample_result(dtype)

        return {
            "points": np.concatenate(samples.points, axis=0),
            "normals": np.concatenate(samples.normals, axis=0),
            "perps1": np.concatenate(samples.perps1, axis=0),
            "perps2": np.concatenate(samples.perps2, axis=0),
            "weights": np.concatenate(samples.weights, axis=0),
            "faces": np.concatenate(samples.faces, axis=0),
            "barycentric": np.concatenate(samples.barycentric, axis=0),
        }

    def _append_barycentric_group(
        self,
        *,
        samples: TriangleMesh._SampleAccumulators,
        barycentric: np.ndarray,
        triangles: np.ndarray,
        normals: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
        areas: np.ndarray,
        face_ids: np.ndarray,
        dtype: np.dtype,
    ) -> None:
        """Append barycentric samples for a group of triangles sharing subdivision count."""

        num_samples = barycentric.shape[0]
        sample_points = np.einsum("sb,fbc->fsc", barycentric, triangles).reshape(-1, 3)
        bary_tile = np.broadcast_to(barycentric, (triangles.shape[0], num_samples, 3)).reshape(
            -1, 3
        )
        weights = np.repeat((areas / num_samples).astype(dtype), num_samples)
        faces = np.repeat(face_ids, num_samples)

        samples.append(
            sample=self._SampleBlock(
                points=sample_points,
                normals=np.repeat(normals, num_samples, axis=0),
                perps1=np.repeat(perps1, num_samples, axis=0),
                perps2=np.repeat(perps2, num_samples, axis=0),
                weights=weights,
                faces=faces,
                barycentric=bary_tile,
            )
        )

    @staticmethod
    def _triangle_area_and_normal(triangle: NDArray) -> tuple[float, np.ndarray]:
        """Return area and outward normal of the provided triangle."""

        edge01 = triangle[1] - triangle[0]
        edge02 = triangle[2] - triangle[0]
        cross = np.cross(edge01, edge02)
        norm = np.linalg.norm(cross)
        if norm <= 0.0:
            return 0.0, np.zeros(3, dtype=triangle.dtype)
        normal = (cross / norm).astype(triangle.dtype, copy=False)
        area = 0.5 * norm
        return area, normal

    @staticmethod
    def _triangle_plane_segments(
        triangle: NDArray, axis: int, plane_value: float, tol: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return intersection segments between a triangle and an axis-aligned plane."""

        vertices = np.asarray(triangle)
        distances = vertices[:, axis] - plane_value
        edges = ((0, 1), (1, 2), (2, 0))

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        points: list[np.ndarray] = []

        def add_point(pt: np.ndarray) -> None:
            for existing in points:
                if np.linalg.norm(existing - pt) <= tol:
                    return
            points.append(pt.copy())

        for i, j in edges:
            di = distances[i]
            dj = distances[j]
            vi = vertices[i]
            vj = vertices[j]

            if abs(di) <= tol and abs(dj) <= tol:
                segments.append((vi.copy(), vj.copy()))
                continue

            if di * dj > 0.0:
                continue

            if abs(di) <= tol:
                add_point(vi)
                continue

            if abs(dj) <= tol:
                add_point(vj)
                continue

            denom = di - dj
            if abs(denom) <= tol:
                continue
            t = di / denom
            if t < 0.0 or t > 1.0:
                continue
            point = vi + t * (vj - vi)
            add_point(point)

        if segments:
            return segments

        if len(points) >= 2:
            return [(points[0], points[1])]

        return []

    @staticmethod
    def _clip_polygon_with_plane(
        polygon: list[np.ndarray], axis: int, bound: float, keep_below: bool, tol: float
    ) -> list[np.ndarray]:
        """Clip a polygon with an axis-aligned plane."""

        if not polygon:
            return []

        result: list[np.ndarray] = []
        prev = polygon[-1]
        prev_val = prev[axis]
        prev_inside = (prev_val <= bound + tol) if keep_below else (prev_val >= bound - tol)

        for current in polygon:
            curr_val = current[axis]
            curr_inside = (curr_val <= bound + tol) if keep_below else (curr_val >= bound - tol)

            if curr_inside:
                if not prev_inside:
                    result.append(
                        TriangleMesh._segment_plane_intersection(prev, current, axis, bound, tol)
                    )
                result.append(current)
            elif prev_inside:
                result.append(
                    TriangleMesh._segment_plane_intersection(prev, current, axis, bound, tol)
                )

            prev = current
            prev_inside = curr_inside

        return result

    @staticmethod
    def _segment_plane_intersection(
        p0: np.ndarray, p1: np.ndarray, axis: int, bound: float, tol: float
    ) -> np.ndarray:
        """Return intersection point between segment (p0,p1) and axis-aligned plane."""

        v0 = float(p0[axis]) - bound
        v1 = float(p1[axis]) - bound
        denom = v1 - v0
        if abs(denom) <= tol:
            return p0.copy()
        t = -v0 / denom
        t = float(np.clip(t, 0.0, 1.0))
        return p0 + t * (p1 - p0)

    @classmethod
    def _clip_triangle_to_bounds(
        cls, triangle: NDArray, sim_min: NDArray, sim_max: NDArray, tol: float
    ) -> tuple[list[NDArray], bool]:
        """Clip triangle against axis-aligned bounds, return list of sub-triangles and flag."""

        vertices = np.asarray(triangle)
        inside = np.all(vertices >= (sim_min - tol), axis=1) & np.all(
            vertices <= (sim_max + tol), axis=1
        )
        if np.all(inside):
            return [triangle], False

        polygon = [triangle[0].copy(), triangle[1].copy(), triangle[2].copy()]
        clipped_flag = True
        for axis in range(3):
            polygon = cls._clip_polygon_with_plane(polygon, axis, sim_min[axis], False, tol)
            if not polygon:
                return [], True
            polygon = cls._clip_polygon_with_plane(polygon, axis, sim_max[axis], True, tol)
            if not polygon:
                return [], True

        if len(polygon) < 3:
            return [], True

        triangles: list[NDArray] = []
        anchor = polygon[0]
        for idx in range(1, len(polygon) - 1):
            tri_clip = np.array([anchor, polygon[idx], polygon[idx + 1]], dtype=triangle.dtype)
            triangles.append(tri_clip)
        return triangles, clipped_flag

    @staticmethod
    def _barycentric_coordinates(triangle: NDArray, points: np.ndarray, tol: float) -> np.ndarray:
        """Compute barycentric coordinates of ``points`` with respect to ``triangle``."""

        pts = np.asarray(points, dtype=triangle.dtype)
        v0 = triangle[0]
        v1 = triangle[1]
        v2 = triangle[2]
        v0v1 = v1 - v0
        v0v2 = v2 - v0

        d00 = float(np.dot(v0v1, v0v1))
        d01 = float(np.dot(v0v1, v0v2))
        d11 = float(np.dot(v0v2, v0v2))
        denom = d00 * d11 - d01 * d01
        if abs(denom) <= tol:
            return np.tile(
                np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=triangle.dtype), (pts.shape[0], 1)
            )

        v0p = pts - v0
        d20 = v0p @ v0v1
        d21 = v0p @ v0v2
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary = np.stack((u, v, w), axis=1)
        return bary.astype(triangle.dtype, copy=False)

    @classmethod
    def _subdivision_count(
        cls,
        area: float,
        spacing: float,
        edge_lengths: Optional[tuple[float, float, float]] = None,
    ) -> int:
        """Determine the number of subdivisions needed for the given area and spacing."""

        spacing = max(float(spacing), np.finfo(float).eps)

        target = np.sqrt(max(area, 0.0))
        area_based = np.ceil(np.sqrt(2.0) * target / spacing)

        edge_based = 0.0
        if edge_lengths:
            max_edge = max(edge_lengths)
            if max_edge > 0.0:
                edge_based = np.ceil(max_edge / spacing)

        subdivisions = max(1, int(max(area_based, edge_based)))
        return subdivisions

    def _get_barycentric_samples(self, subdivisions: int, dtype: np.dtype) -> np.ndarray:
        """Return barycentric sample coordinates for a subdivision level."""

        cache = self._barycentric_samples
        if subdivisions not in cache:
            cache[subdivisions] = self._build_barycentric_samples(subdivisions)
        return cache[subdivisions].astype(dtype, copy=False)

    @staticmethod
    def _build_barycentric_samples(subdivisions: int) -> np.ndarray:
        """Construct barycentric sampling points for a given subdivision level."""

        if subdivisions <= 1:
            return np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])

        bary = []
        for i in range(subdivisions):
            for j in range(subdivisions - i):
                l1 = (i + 1.0 / 3.0) / subdivisions
                l2 = (j + 1.0 / 3.0) / subdivisions
                l0 = 1.0 - l1 - l2
                bary.append((l0, l1, l2))
        return np.asarray(bary, dtype=float)

    @staticmethod
    def subdivide_faces(vertices: NDArray, faces: NDArray) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly subdivide each triangular face by inserting edge midpoints."""

        midpoint_cache: dict[tuple[int, int], int] = {}
        verts_list = [np.asarray(v, dtype=float) for v in vertices]

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in midpoint_cache:
                return midpoint_cache[key]
            vm = 0.5 * (verts_list[i] + verts_list[j])
            verts_list.append(vm)
            idx = len(verts_list) - 1
            midpoint_cache[key] = idx
            return idx

        new_faces: list[tuple[int, int, int]] = []
        for tri in faces:
            a = midpoint(tri[0], tri[1])
            b = midpoint(tri[1], tri[2])
            c = midpoint(tri[2], tri[0])
            new_faces.extend(((tri[0], a, c), (tri[1], b, a), (tri[2], c, b), (a, b, c)))

        verts_arr = np.asarray(verts_list, dtype=float)
        return verts_arr, np.asarray(new_faces, dtype=int)

    @staticmethod
    def _triangle_tangent_basis(
        triangle: NDArray, normal: NDArray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Compute orthonormal tangential vectors for a triangle."""

        tol = np.finfo(triangle.dtype).eps
        edges = [triangle[1] - triangle[0], triangle[2] - triangle[0], triangle[2] - triangle[1]]

        edge = None
        for candidate in edges:
            length = np.linalg.norm(candidate)
            if length > tol:
                edge = (candidate / length).astype(triangle.dtype, copy=False)
                break

        if edge is None:
            return None

        perp1 = edge
        perp2 = np.cross(normal, perp1)
        perp2_norm = np.linalg.norm(perp2)
        if perp2_norm <= tol:
            return None
        perp2 = (perp2 / perp2_norm).astype(triangle.dtype, copy=False)
        return perp1, perp2
