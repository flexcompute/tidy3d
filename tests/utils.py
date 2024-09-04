from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import tidy3d as td
import trimesh
import xarray as xr
from autograd.core import VJPNode
from autograd.tracer import new_box
from tidy3d import ModeIndexDataArray
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.log import _get_level_int
from tidy3d.web import BatchData

""" utilities shared between all tests """
np.random.seed(4)

# function used to generate the data for emulated runs
DATA_GEN_FN = np.random.random

FREQS = np.array([1.90, 2.01, 2.2]) * 1e12
SIM_MONITORS = td.Simulation(
    size=(10.0, 10.0, 10.0),
    grid_spec=td.GridSpec(wavelength=1.0),
    run_time=1e-13,
    monitors=[
        td.FieldMonitor(size=(1, 1, 1), center=(0, 1, 0), freqs=FREQS, name="field_freq"),
        td.FieldTimeMonitor(size=(1, 1, 0), center=(1, 0, 0), interval=10, name="field_time"),
        td.FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=FREQS, name="flux_freq"),
        td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), start=1e-12, name="flux_time"),
        td.ModeMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=FREQS,
            mode_spec=td.ModeSpec(num_modes=3),
            name="mode",
        ),
    ],
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
)

# STL geometry
VERTICES = np.array([[-1.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-1.5, 0.5, -0.5], [-1.5, -0.5, 0.5]])
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
STL_GEO = td.TriangleMesh.from_trimesh(trimesh.Trimesh(VERTICES, FACES))


# custom medium


def cartesian_to_unstructured(
    array: td.SpatialDataArray,
    pert: float = 0.1,
    method: str = "linear",
    seed: int = None,
    same_bounds: bool = True,
) -> Union[td.TriangularGridDataset, td.TetrahedralGridDataset]:
    """Convert a SpatialDataArray into TriangularGridDataset/TetrahedralGridDataset with
    an optional perturbation of point coordinates.

    Parameters
    ----------
    array : td.SpatialDataArray
        Array to convert.
    pert : float = 0.1
        Degree of perturbations of point coordinates.
    method : Literal["linear", "nearest", "direct"] = "linear"
        Interpolation method for transfering data to unstructured grid.
    seed : int = None
        Seed number to use when randomly perturbing point coordinates.
    same_bounds : bool = True
        Preserve boundaries of data array. That is, data remains defined in a rectangular domain.
        This options works best with ``method="direct"``, otherwise boundary nodes will not have values.

    Returns
    -------
    Union[td.TriangularGridDataset, td.TetrahedralGridDataset]
        Unstructured grid dataset.
    """

    xyz = [array.x, array.y, array.z]
    lens = [len(coord) for coord in xyz]

    num_len_zero = sum(length == 1 for length in lens)

    if num_len_zero == 1:
        normal_axis = lens.index(1)
        normal_pos = xyz[normal_axis].values.item()
        xyz.pop(normal_axis)

    dxyz = [np.gradient(coord) for coord in xyz]

    XYZ = np.meshgrid(*xyz, indexing="ij")
    dXYZ = np.meshgrid(*dxyz, indexing="ij")

    shape = np.shape(XYZ[0])

    XYZp = XYZ.copy()
    rng = np.random.default_rng(seed=seed)

    x_pert = (1 - 2 * rng.random(shape)) * pert
    if same_bounds:
        x_pert[0] = 0
        x_pert[-1] = 0
    else:
        x_pert[0] = -np.abs(x_pert[0])
        x_pert[-1] = np.abs(x_pert[-1])

    XYZp[0] = XYZp[0] + dXYZ[0] * x_pert

    y_pert = (1 - 2 * rng.random(shape)) * pert
    if same_bounds:
        y_pert[:, 0] = 0
        y_pert[:, -1] = 0
    else:
        y_pert[:, 0] = -np.abs(y_pert[:, 0])
        y_pert[:, -1] = np.abs(y_pert[:, -1])

    XYZp[1] = XYZp[1] + dXYZ[1] * y_pert

    if num_len_zero == 0:
        z_pert = (1 - 2 * rng.random(shape)) * pert
        if same_bounds:
            z_pert[:, :, 0] = 0
            z_pert[:, :, -1] = 0
        else:
            z_pert[:, :, 0] = -np.abs(z_pert[:, :, 0])
            z_pert[:, :, -1] = np.abs(z_pert[:, :, -1])

        XYZp[2] = XYZp[2] + dXYZ[2] * z_pert

        points = np.transpose([XYZp[0].ravel(), XYZp[1].ravel(), XYZp[2].ravel()])
        if method == "direct":
            values = array
        else:
            values = array.interp(
                x=xr.DataArray(points[:, 0], dims=["index"]),
                y=xr.DataArray(points[:, 1], dims=["index"]),
                z=xr.DataArray(points[:, 2], dims=["index"]),
                method=method,
            )

        # Kuhn triangulation of box
        linear_inds = np.arange(np.prod(lens))
        linear_inds = np.reshape(linear_inds, shape)

        box_vertex_0_inds = linear_inds[:-1, :-1, :-1].ravel()
        box_vertex_1_inds = linear_inds[:-1, :-1, 1:].ravel()
        box_vertex_2_inds = linear_inds[:-1, 1:, :-1].ravel()
        box_vertex_3_inds = linear_inds[:-1, 1:, 1:].ravel()
        box_vertex_4_inds = linear_inds[1:, :-1, :-1].ravel()
        box_vertex_5_inds = linear_inds[1:, :-1, 1:].ravel()
        box_vertex_6_inds = linear_inds[1:, 1:, :-1].ravel()
        box_vertex_7_inds = linear_inds[1:, 1:, 1:].ravel()

        cell_vertex_0 = np.concatenate(
            (
                box_vertex_0_inds,
                box_vertex_0_inds,
                box_vertex_0_inds,
                box_vertex_0_inds,
                box_vertex_0_inds,
                box_vertex_0_inds,
            )
        )
        cell_vertex_1 = np.concatenate(
            (
                box_vertex_7_inds,
                box_vertex_7_inds,
                box_vertex_7_inds,
                box_vertex_7_inds,
                box_vertex_7_inds,
                box_vertex_7_inds,
            )
        )
        cell_vertex_2 = np.concatenate(
            (
                box_vertex_1_inds,
                box_vertex_2_inds,
                box_vertex_2_inds,
                box_vertex_4_inds,
                box_vertex_4_inds,
                box_vertex_1_inds,
            )
        )
        cell_vertex_3 = np.concatenate(
            (
                box_vertex_3_inds,
                box_vertex_3_inds,
                box_vertex_6_inds,
                box_vertex_6_inds,
                box_vertex_5_inds,
                box_vertex_5_inds,
            )
        )

        cells = np.transpose([cell_vertex_0, cell_vertex_1, cell_vertex_2, cell_vertex_3]).copy()

        griddataset = td.TetrahedralGridDataset(
            points=td.PointDataArray(points, dims=("index", "axis")),
            cells=td.CellDataArray(cells, dims=("cell_index", "vertex_index")),
            values=td.IndexedDataArray(values.values.ravel(), dims=("index")),
        )
        return griddataset

    else:
        points = np.transpose([XYZp[0].ravel(), XYZp[1].ravel()])

        # Kuhn triangulation of square
        linear_inds = np.arange(np.prod(lens))
        shape_2d = (len(xyz[0]), len(xyz[1]))
        linear_inds = np.reshape(linear_inds, shape_2d)

        square_vertex_0_inds = linear_inds[:-1, :-1].ravel()
        square_vertex_1_inds = linear_inds[:-1, 1:].ravel()
        square_vertex_2_inds = linear_inds[1:, :-1].ravel()
        square_vertex_3_inds = linear_inds[1:, 1:].ravel()

        cell_vertex_0 = np.concatenate((square_vertex_0_inds, square_vertex_0_inds))
        cell_vertex_1 = np.concatenate((square_vertex_1_inds, square_vertex_2_inds))
        cell_vertex_2 = np.concatenate((square_vertex_3_inds, square_vertex_3_inds))

        cells = np.transpose([cell_vertex_0, cell_vertex_1, cell_vertex_2]).copy()

        xyz_names = ["x", "y", "z"]
        normal_name = xyz_names.pop(normal_axis)
        if method == "direct":
            values = array.isel({normal_name: 0})
        else:
            values = array.isel({normal_name: 0}).interp(
                {
                    xyz_names[0]: xr.DataArray(points[:, 0], dims=["index"]),
                    xyz_names[1]: xr.DataArray(points[:, 1], dims=["index"]),
                },
                method=method,
            )

        griddataset = td.TriangularGridDataset(
            points=td.PointDataArray(points, dims=("index", "axis")),
            cells=td.CellDataArray(cells, dims=("cell_index", "vertex_index")),
            values=td.IndexedDataArray(values.values.ravel(), dims=("index")),
            normal_axis=normal_axis,
            normal_pos=normal_pos,
        )
        return griddataset


def make_spatial_data(
    size,
    bounds,
    lims=(0, 1),
    seed_data=None,
    unstructured=False,
    perturbation=0.1,
    seed_grid=None,
    method="linear",
):
    """Makes a spatial data array."""
    rng = np.random.default_rng(seed=seed_data)
    data = lims[0] + (lims[1] - lims[0]) * rng.random(size)
    arr = td.SpatialDataArray(
        data,
        coords=dict(
            x=np.linspace(bounds[0][0], bounds[1][0], size[0]),
            y=np.linspace(bounds[0][1], bounds[1][1], size[1]),
            z=np.linspace(bounds[0][2], bounds[1][2], size[2]),
        ),
    )
    if unstructured:
        return cartesian_to_unstructured(arr, pert=perturbation, method=method, seed=seed_grid)
    return arr


COORDS = dict(x=[-1.5, -0.5], y=[0, 1], z=[0, 1])
CUSTOM_SIZE = (2, 2, 2)
CUSTOM_BOUNDS = [[-1.5, 0, 0], [-0.5, 1, 1]]
CUSTOM_GRID_SEED = 12345


def make_custom_data(lims, unstructured):
    return make_spatial_data(
        size=CUSTOM_SIZE,
        bounds=CUSTOM_BOUNDS,
        lims=lims,
        unstructured=unstructured,
        seed_grid=CUSTOM_GRID_SEED,
    )


custom_medium = td.CustomMedium(
    permittivity=make_custom_data([1, 2], False),
)
custom_poleresidue = td.CustomPoleResidue(
    eps_inf=make_custom_data([1, 2], False),
    poles=(
        (
            make_custom_data([-1, 0], False),
            make_custom_data([1, 2], False),
        ),
    ),
)
custom_debye = td.CustomDebye(
    eps_inf=make_custom_data([1, 2], False),
    coeffs=(
        (
            make_custom_data([1, 2], False),
            make_custom_data([1, 2], False),
        ),
    ),
)

custom_drude = td.CustomDrude(
    eps_inf=make_custom_data([1, 2], False),
    coeffs=(
        (
            make_custom_data([1, 2], False),
            make_custom_data([1, 2], False),
        ),
    ),
)

custom_lorentz = td.CustomLorentz(
    eps_inf=make_custom_data([1, 2], False),
    coeffs=(
        (
            make_custom_data([1, 2], False),
            make_custom_data([10, 11], False),
            make_custom_data([1, 2], False),
        ),
    ),
)

custom_sellmeier = td.CustomSellmeier(
    coeffs=(
        (
            make_custom_data([0.1, 1.1], False),
            make_custom_data([10, 11], False),
        ),
    ),
)

custom_medium_u = td.CustomMedium(
    permittivity=make_custom_data([1, 2], True),
)
custom_poleresidue_u = td.CustomPoleResidue(
    eps_inf=make_custom_data([1, 2], True),
    poles=(
        (
            make_custom_data([-1, 0], True),
            make_custom_data([1, 2], True),
        ),
    ),
)
custom_debye_u = td.CustomDebye(
    eps_inf=make_custom_data([1, 2], True),
    coeffs=(
        (
            make_custom_data([1, 2], True),
            make_custom_data([1, 2], True),
        ),
    ),
)

custom_drude_u = td.CustomDrude(
    eps_inf=make_custom_data([1, 2], True),
    coeffs=(
        (
            make_custom_data([1, 2], True),
            make_custom_data([1, 2], True),
        ),
    ),
)

custom_lorentz_u = td.CustomLorentz(
    eps_inf=make_custom_data([1, 2], True),
    coeffs=(
        (
            make_custom_data([1, 2], True),
            make_custom_data([10, 11], True),
            make_custom_data([1, 2], True),
        ),
    ),
)

custom_sellmeier_u = td.CustomSellmeier(
    coeffs=(
        (
            make_custom_data([0.1, 1.1], True),
            make_custom_data([10, 11], True),
        ),
    ),
)

# Make a few autograd ArrayBoxes for testing
start_node = VJPNode.new_root()
tracer = new_box(1.0, 0, start_node)
tracer_arr = new_box(np.array([[[1.0]]]), 0, start_node)

SIM_FULL = td.Simulation(
    size=(8.0, 8.0, 8.0),
    run_time=1e-12,
    structures=[
        td.Structure(
            geometry=td.Cylinder(length=1, center=(-1 * tracer, 0, 0), radius=tracer, axis=2),
            medium=td.Medium(permittivity=1 + tracer, name="dieletric"),
            name="traced_dieletric_cylinder",
        ),
        td.Structure(
            geometry=td.Box(size=(1, tracer, tracer), center=(-1 * tracer, 0, 0)),
            medium=td.Medium(permittivity=1 + tracer, name="dieletric"),
            name="traced_dieletric_box",
        ),
        td.Structure(
            geometry=td.PolySlab(
                vertices=[[-1 + tracer * 0.1, 0], [-1 + tracer * 0.1, 0.1], [-1, 0.1]],
                axis=1,
                slab_bounds=(-0.1, 0.1),
            ),
            medium=td.CustomMedium(
                permittivity=td.SpatialDataArray(tracer_arr, coords=dict(x=[-1], y=[0], z=[0]))
            ),
            name="traced custom polyslab",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.Medium(permittivity=2.0, name="dieletric"),
            name="dieletric_box",
        ),
        td.Structure(
            geometry=td.Box(size=(1, td.inf, 1), center=(-1, 0, 0)),
            medium=td.Medium(permittivity=1.0, conductivity=3.0, name="lossy_dieletric"),
            name="lossy_box",
        ),
        td.Structure(
            geometry=td.Sphere(radius=1.0, center=(1.0, 0.0, 1.0)),
            medium=td.Sellmeier(
                coeffs=[(1.03961212, 0.00600069867), (0.231792344, 0.0200179144)], name="sellmeier"
            ),
            name="sellmeier_sphere",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.Lorentz(eps_inf=2.0, coeffs=[(1, 2, 3)], name="lorentz"),
            name="lorentz_box",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.Debye(eps_inf=2.0, coeffs=[(1, 3)]),
        ),
        td.Structure(
            geometry=STL_GEO,
            medium=td.Debye(eps_inf=2.0, coeffs=[(1, 3)]),
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.Drude(eps_inf=2.0, coeffs=[(1, 3)], name="drude"),
            name="drude_box",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 0, 1), center=(-1, 0, 0)),
            medium=td.Medium2D.from_medium(td.Medium(conductivity=0.45), thickness=0.01),
        ),
        td.Structure(
            geometry=td.Box(size=(1, 0, 1), center=(-1, 0, 0)),
            medium=td.PEC2D,
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.AnisotropicMedium(xx=td.PEC, yy=td.Medium(), zz=td.Medium()),
        ),
        td.Structure(
            geometry=td.GeometryGroup(geometries=[td.Box(size=(1, 1, 1), center=(-1, 0, 0))]),
            medium=td.PEC,
            name="pec_group",
        ),
        td.Structure(
            geometry=td.Cylinder(radius=1.0, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
            medium=td.AnisotropicMedium(
                xx=td.Medium(permittivity=1),
                yy=td.Medium(permittivity=2),
                zz=td.Medium(permittivity=3),
            ),
            name="anisotopic_cylinder",
        ),
        td.Structure(
            geometry=td.PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
            ),
            medium=td.PoleResidue(
                eps_inf=1.0, poles=((6206417594288582j, (-3.311074436985222e16j)),)
            ),
            name="pole_slab",
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_medium,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_drude,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_lorentz,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_debye,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_poleresidue,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_sellmeier,
        ),
        td.Structure(
            geometry=td.Box(
                size=(0.1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_medium_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_drude_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_lorentz_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_debye_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_poleresidue_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_sellmeier_u,
        ),
        td.Structure(
            geometry=td.Box(
                size=(1, 1, 1),
                center=(-3.0, 0.5, 0.5),
            ),
            medium=td.Medium(
                nonlinear_spec=td.NonlinearSusceptibility(chi3=0.1, numiters=20),
            ),
        ),
        td.Structure(
            geometry=td.Box(
                size=(0.1, 1, 1),
                center=(-3.0, 0.5, 0.5),
            ),
            medium=td.Medium(
                nonlinear_spec=td.NonlinearSpec(
                    num_iters=10,
                    models=[
                        td.NonlinearSusceptibility(chi3=0.1),
                        td.TwoPhotonAbsorption(
                            beta=1, sigma=1, tau=1, e_e=1, e_h=0.8, c_e=1, c_h=1
                        ),
                        td.KerrNonlinearity(n2=1),
                    ],
                )
            ),
        ),
        td.Structure(
            geometry=td.PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
            ),
            medium=td.PoleResidue(
                eps_inf=1.0, poles=((6206417594288582j, (-3.311074436985222e16j)),)
            ),
        ),
        td.Structure(
            geometry=td.TriangleMesh.from_triangles(
                np.array(
                    [
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                    ]
                )
                + np.array(
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    ]
                )
            ),
            medium=td.Medium(permittivity=5),
            name="dieletric_mesh",
        ),
        td.Structure(
            geometry=td.TriangleMesh.from_stl(
                "tests/data/two_boxes_separate.stl", scale=0.1, origin=(0.5, 0.5, 0.5)
            ),
            medium=td.Medium(permittivity=5),
        ),
        td.Structure(
            geometry=td.ClipOperation(
                geometry_a=td.Box(size=(1, 1, 1), center=(0.9, 0.9, 0.9)),
                geometry_b=td.Box(size=(1, 1, 1), center=(1.1, 1.1, 1.1)),
                operation="symmetric_difference",
            ),
            medium=td.Medium(permittivity=3),
            name="clip_operation",
        ),
        td.Structure(
            geometry=td.Transformed(
                geometry=td.Box(size=(1, 1, 1), center=(1, 1, 1)),
                transform=td.Transformed.rotation(np.pi / 12, 2),
            ),
            medium=td.Medium(permittivity=1.5),
            name="transformed_box",
        ),
    ],
    sources=[
        td.UniformCurrentSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
        ),
        td.PointDipole(
            center=(0, 0.5, 0),
            polarization="Ex",
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
        ),
        td.ModeSource(
            center=(0, 0.5, 0),
            size=(2, 0, 2),
            mode_spec=td.ModeSpec(),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            direction="-",
        ),
        td.PlaneWave(
            size=(0, td.inf, td.inf),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=0.1,
            direction="+",
        ),
        td.GaussianBeam(
            size=(0, 3, 3),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=np.pi / 2,
            direction="+",
            waist_radius=1.0,
        ),
        td.AstigmaticGaussianBeam(
            size=(0, 3, 3),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(1.0, 2.0),
            waist_distances=(3.0, 4.0),
        ),
        td.CustomFieldSource(
            center=(0, 1, 2),
            size=(2, 2, 0),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            field_dataset=td.FieldDataset(
                Ex=td.ScalarFieldDataArray(
                    np.ones((101, 101, 1, 1)),
                    coords=dict(
                        x=np.linspace(-1, 1, 101),
                        y=np.linspace(-1, 1, 101),
                        z=np.array([0]),
                        f=[2e14],
                    ),
                )
            ),
        ),
        td.CustomCurrentSource(
            center=(0, 1, 2),
            size=(2, 2, 0),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            current_dataset=td.FieldDataset(
                Ex=td.ScalarFieldDataArray(
                    np.ones((101, 101, 1, 1)),
                    coords=dict(
                        x=np.linspace(-1, 1, 101),
                        y=np.linspace(-1, 1, 101),
                        z=np.array([0]),
                        f=[2e14],
                    ),
                )
            ),
        ),
        td.TFSF(
            center=(1, 2, -3),
            size=(2.5, 2.5, 0.5),
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            direction="+",
            angle_theta=np.pi / 6,
            angle_phi=np.pi / 5,
            injection_axis=2,
        ),
        td.UniformCurrentSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=td.CustomSourceTime.from_values(
                freq0=2e14, fwidth=4e13, values=np.linspace(0, 10, 1000), dt=1e-12 / 100
            ),
        ),
    ],
    monitors=(
        td.FieldMonitor(
            size=(0, 0, 0), center=(0, 0, 0), fields=["Ex"], freqs=[1.5e14, 2e14], name="field"
        ),
        td.FieldTimeMonitor(size=(0, 0, 0), center=(0, 0, 0), name="field_time", interval=100),
        td.FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name="flux"),
        td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), name="flux_time"),
        td.PermittivityMonitor(size=(1, 1, 0.1), name="eps", freqs=[1e14]),
        td.ModeMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            name="mode",
            freqs=[2e14, 2.5e14],
            mode_spec=td.ModeSpec(),
        ),
        td.FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_angle",
            custom_origin=(1, 2, 3),
            phi=[0, np.pi / 6],
            theta=np.linspace(np.pi / 4, np.pi / 4 + np.pi / 2, 100),
        ),
        td.FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_cartesian",
            custom_origin=(1, 2, 3),
            x=[-1, 0, 1],
            y=[-2, -1, 0, 1, 2],
            proj_axis=2,
            proj_distance=5,
        ),
        td.FieldProjectionKSpaceMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_kspace",
            custom_origin=(1, 2, 3),
            proj_axis=2,
            ux=[0.02, 0.04],
            uy=[0.03, 0.04, 0.05],
        ),
        td.FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_angle_exact",
            custom_origin=(1, 2, 3),
            phi=[0, np.pi / 8],
            theta=np.linspace(np.pi / 4, np.pi / 4 + np.pi / 2, 100),
            far_field_approx=False,
        ),
        td.DiffractionMonitor(
            size=(0, td.inf, td.inf),
            center=(0, 0, 0),
            name="diffraction",
            freqs=[1e14, 2e14],
        ),
    ),
    symmetry=(0, 0, 0),
    boundary_spec=td.BoundarySpec(
        x=td.Boundary(plus=td.PML(num_layers=20), minus=td.Absorber(num_layers=100)),
        y=td.Boundary.bloch(bloch_vec=1),
        z=td.Boundary.periodic(),
    ),
    shutoff=1e-4,
    courant=0.8,
    subpixel=False,
    grid_spec=td.GridSpec(
        grid_x=td.AutoGrid(),
        grid_y=td.CustomGrid(dl=100 * [0.04]),
        grid_z=td.UniformGrid(dl=0.05),
        override_structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            )
        ],
    ),
)


def get_spatial_coords_dict(simulation: td.Simulation, monitor: td.Monitor, field_name: str):
    """Returns MonitorData coordinates associated with a Monitor object"""
    grid = simulation.discretize_monitor(monitor)
    spatial_coords = grid.boundaries if monitor.colocate else grid[field_name]
    spatial_coords_dict = spatial_coords.dict()

    coords = {}
    for axis, dim in enumerate("xyz"):
        if monitor.size[axis] == 0:
            coords[dim] = [monitor.center[axis]]
        elif monitor.colocate:
            coords[dim] = spatial_coords_dict[dim][:-1]
        else:
            coords[dim] = spatial_coords_dict[dim]

    return coords


def run_emulated(simulation: td.Simulation, path=None, **kwargs) -> td.SimulationData:
    """Emulates a simulation run."""
    from scipy.ndimage.filters import gaussian_filter

    x = kwargs.get("x0", 1.0)

    def make_data(
        coords: dict, data_array_type: type, is_complex: bool = False
    ) -> td.components.data.data_array.DataArray:
        """make a random DataArray out of supplied coordinates and data_type."""
        data_shape = [len(coords[k]) for k in data_array_type._dims]
        np.random.seed(1)
        data = DATA_GEN_FN(data_shape)

        data = (1 + 0.5j) * data if is_complex else data
        data = gaussian_filter(data, sigma=1.0)  # smooth out the data a little so it isnt random
        data_array = data_array_type(x * data, coords=coords)
        return data_array

    def make_field_data(monitor: td.FieldMonitor) -> td.FieldData:
        """make a random FieldData from a FieldMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)

            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarFieldDataArray, is_complex=True
            )

        return td.FieldData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_field_time_data(monitor: td.FieldTimeMonitor) -> td.FieldTimeData:
        """make a random FieldTimeData from a FieldTimeMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        tmesh = simulation.tmesh
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)

            (idx_begin, idx_end) = monitor.time_inds(tmesh)
            tcoords = tmesh[idx_begin:idx_end]
            coords["t"] = tcoords
            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarFieldTimeDataArray, is_complex=False
            )

        return td.FieldTimeData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_mode_solver_data(monitor: td.ModeSolverMonitor) -> td.ModeSolverData:
        """make a random ModeSolverData from a ModeSolverMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        index_coords = {}
        index_coords["f"] = list(monitor.freqs)
        index_coords["mode_index"] = np.arange(monitor.mode_spec.num_modes)
        index_data_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        index_data = ModeIndexDataArray(
            (1 + 1j) * DATA_GEN_FN(index_data_shape), coords=index_coords
        )
        for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)
            coords["mode_index"] = index_coords["mode_index"]

            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
            )

        return td.ModeSolverData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            n_complex=index_data,
            **field_cmps,
        )

    def make_eps_data(monitor: td.PermittivityMonitor) -> td.PermittivityData:
        """make a random PermittivityData from a PermittivityMonitor."""
        field_mnt = td.FieldMonitor(**monitor.dict(exclude={"type", "fields"}))
        field_data = make_field_data(monitor=field_mnt)
        return td.PermittivityData(
            monitor=monitor,
            eps_xx=field_data.Ex,
            eps_yy=field_data.Ey,
            eps_zz=field_data.Ez,
            grid_expanded=simulation.discretize_monitor(monitor),
        )

    def make_diff_data(monitor: td.DiffractionMonitor) -> td.DiffractionData:
        """make a random DiffractionData from a DiffractionMonitor."""
        f = list(monitor.freqs)
        orders_x = np.linspace(-1, 1, 3)
        orders_y = np.linspace(-2, 2, 5)
        coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
        values = DATA_GEN_FN((len(orders_x), len(orders_y), len(f)))
        data = td.DiffractionDataArray(values, coords=coords)
        field_data = {field: data for field in ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")}
        return td.DiffractionData(monitor=monitor, sim_size=(1, 1), bloch_vecs=(0, 0), **field_data)

    def make_mode_data(monitor: td.ModeMonitor) -> td.ModeData:
        """make a random ModeData from a ModeMonitor."""
        _ = np.arange(monitor.mode_spec.num_modes)
        coords_ind = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        n_complex = make_data(
            coords=coords_ind, data_array_type=td.ModeIndexDataArray, is_complex=True
        )
        coords_amps = dict(direction=["+", "-"])
        coords_amps.update(coords_ind)
        amps = make_data(coords=coords_amps, data_array_type=td.ModeAmpsDataArray, is_complex=True)
        return td.ModeData(
            monitor=monitor,
            n_complex=n_complex,
            amps=amps,
            grid_expanded=simulation.discretize_monitor(monitor),
        )

    def make_flux_data(monitor: td.FluxMonitor) -> td.FluxData:
        """make a random ModeData from a ModeMonitor."""

        coords = dict(f=list(monitor.freqs))
        flux = make_data(coords=coords, data_array_type=td.FluxDataArray, is_complex=False)
        return td.FluxData(monitor=monitor, flux=flux)

    MONITOR_MAKER_MAP = {
        td.FieldMonitor: make_field_data,
        td.FieldTimeMonitor: make_field_time_data,
        td.ModeSolverMonitor: make_mode_solver_data,
        td.ModeMonitor: make_mode_data,
        td.PermittivityMonitor: make_eps_data,
        td.DiffractionMonitor: make_diff_data,
        td.FluxMonitor: make_flux_data,
    }

    data = [MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.monitors]
    sim_data = td.SimulationData(simulation=simulation, data=data)

    if path is not None:
        sim_data.to_file(str(path))

    return sim_data


class BatchDataTest(Tidy3dBaseModel):
    """Holds a collection of :class:`.SimulationData` returned by :class:`.Batch`."""

    task_paths: Dict[str, str] = pd.Field(
        ...,
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: Dict[str, str] = pd.Field(
        ..., title="Task IDs", description="Mapping of task_name to task_id for each task in batch."
    )

    sim_data: Dict[str, td.SimulationData]

    def load_sim_data(self, task_name: str) -> td.SimulationData:
        """Load a :class:`.SimulationData` from file by task name."""
        _ = self.task_paths[task_name]
        _ = self.task_ids[task_name]
        return self.sim_data[task_name]

    def items(self) -> Tuple[str, td.SimulationData]:
        """Iterate through the :class:`.SimulationData` for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: str) -> td.SimulationData:
        """Get the :class:`.SimulationData` for a given ``task_name``."""
        return self.load_sim_data(task_name)


def run_async_emulated(simulations: Dict[str, td.Simulation], **kwargs) -> BatchData:
    """Emulate an async run function."""
    task_ids = {task_name: f"task_id={i}" for i, task_name in enumerate(simulations.keys())}
    task_paths = {task_name: "NONE" for task_name in simulations.keys()}
    sim_data = {task_name: run_emulated(sim) for task_name, sim in simulations.items()}

    return BatchDataTest(task_paths=task_paths, task_ids=task_ids, sim_data=sim_data)


def assert_log_level(
    records: List[Tuple[int, str]], log_level_expected: str, contains_str: str = None
) -> None:
    """Testing tool: Raises error if a log was not recorded as expected.

    Parameters
    ----------
    records : List[Tuple[int, str]]
        List of (log_level: int, message: str) holding all of the captured logs.
    log_level_expected: str
        String version of expected log level (all uppercase). The function checks that this log
        log level is present in the records, **as well as** that no higher log level is present.
    contains_str : str = None
        If specified, errors if not found in any of the log messages that are at level
        ``log_level_expected``.

    Returns
    -------
        None
    """

    import sys

    sys.stderr.write(str(records) + "\n")

    if log_level_expected is None:
        log_level_expected_int = None
    else:
        log_level_expected_int = _get_level_int(log_level_expected)

    # there's a log but the log level is None (problem)
    if records and not log_level_expected_int:
        raise AssertionError("Log was recorded but requested log level is None.")

    # we expect a log but none is given (problem)
    if log_level_expected_int and not records:
        raise AssertionError("Log was not recorded but requested log level is not None.")

    # both expected and got log, check the log levels match
    if records and log_level_expected:
        string_found = False
        expected_level_present = False
        expected_level_exceeded = False
        for log in records:
            log_level, log_message = log
            if log_level == log_level_expected_int:
                expected_level_present = True
                if contains_str and contains_str in log_message:
                    string_found = True
            elif log_level > log_level_expected_int:
                expected_level_exceeded = True

        if expected_level_exceeded:
            raise AssertionError(
                f"Recorded log level exceeds expected level '{log_level_expected}'."
            )
        if not expected_level_present:
            raise AssertionError(
                f"Expected log level '{log_level_expected}' was not found in record."
            )
        if contains_str and not string_found:
            raise AssertionError(
                f"Log record at level '{log_level_expected}' did not contain '{contains_str}'."
            )


@pd.dataclasses.dataclass
class AssertLogLevel:
    """Context manager to check log level for records logged within its context."""

    records: Any
    log_level_expected: Union[str, None]
    contains_str: str = None

    def __enter__(self):
        # record number of records going into this context
        self.num_records_before = len(self.records)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # only check NEW records recorded since __enter__
        records_check = self.records[self.num_records_before :]
        assert_log_level(
            records=records_check,
            log_level_expected=self.log_level_expected,
            contains_str=self.contains_str,
        )


def get_test_root_dir():
    """return the root folder of test code"""

    return Path(__file__).parent


def get_nested_shape(nested_obj: Any) -> Any:
    """
    Recursively iterate through a nested object replacing values with None.
    Empty list/tuple/dict are replaced with None.
    Builds the structure for comparison to other nested objects.
    Used to check structure hasn't changed when nested_obj data has been altered.
    Similar concept to .shape method for numpy arrays.

    Parameters
    ----------
    nested_obj : Any
        A nested object to be reduced to its shape.

    Returns
    -------
    Any
        The nested object with values replaced with None whilst keeping the same nested structure.
    """
    if isinstance(nested_obj, dict):
        if len(nested_obj) == 0:
            return None
        else:
            return {key: get_nested_shape(nested_obj[key]) for key in nested_obj}

    # Tuple of dicts, enter and continue iteration
    elif isinstance(nested_obj, (tuple, list)):
        if len(nested_obj) == 0:
            return None
        else:
            return type(nested_obj)(get_nested_shape(val) for val in nested_obj)

    # Replace everything else with None
    else:
        return None
