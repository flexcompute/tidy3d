from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import trimesh
import xarray as xr
from autograd.core import VJPNode
from autograd.tracer import new_box
from pydantic import Field

import tidy3d as td
from tidy3d._testing.synthetic_monitor_data import (
    get_spatial_coords_dict as _get_spatial_coords_dict,
)
from tidy3d._testing.synthetic_monitor_data import (
    make_simulation_data as _make_simulation_data,
)
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
    monitors=(
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
    ),
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
)

# STL geometry
VERTICES = np.array([[-1.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-1.5, 0.5, -0.5], [-1.5, -0.5, 0.5]])
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
STL_GEO = td.TriangleMesh.from_trimesh(trimesh.Trimesh(VERTICES, FACES))


def cartesian_to_unstructured(
    array: td.SpatialDataArray,
    pert: float = 0.1,
    method: str = "linear",
    seed: Optional[int] = None,
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

    XYZp = np.array(XYZ).copy()
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
        coords={
            "x": np.linspace(bounds[0][0], bounds[1][0], size[0]),
            "y": np.linspace(bounds[0][1], bounds[1][1], size[1]),
            "z": np.linspace(bounds[0][2], bounds[1][2], size[2]),
        },
    )
    if unstructured:
        return cartesian_to_unstructured(arr, pert=perturbation, method=method, seed=seed_grid)
    return arr


COORDS = {"x": [-1.5, -0.5], "y": [0, 1], "z": [0, 1]}
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


SIM_FULL_COMMON_MONITORS = (
    td.FieldMonitor(
        size=(0, 0, 0), center=(0, 0, 0), fields=("Ex",), freqs=[1.5e14, 2e14], name="field"
    ),
    td.FieldTimeMonitor(size=(0, 0, 0), center=(0, 0, 0), name="field_time", interval=100),
    td.AuxFieldTimeMonitor(
        size=(0, 0, 0), center=(0, 0, 0), fields=("Nfx",), name="aux_field_time", interval=100
    ),
    td.FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name="flux"),
    td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), name="flux_time"),
    td.PermittivityMonitor(size=(1, 1, 0.1), name="eps", freqs=[1e14]),
    td.MediumMonitor(size=(1, 1, 0.1), name="mat", freqs=[1e14]),
    td.ModeMonitor(
        size=(1, 1, 0),
        center=(0, 0, 0),
        name="mode",
        freqs=[2e14, 2.5e14],
        mode_spec=td.ModeSpec(),
    ),
)

SIM_FULL_PROJECTION_MONITORS = (
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
    td.DirectivityMonitor(
        center=(0, 0, 0),
        size=(0, 2, 2),
        freqs=[250e12, 300e12],
        name="directivity",
        custom_origin=(1, 2, 3),
        phi=[0, np.pi / 6],
        theta=np.linspace(np.pi / 4, np.pi / 4 + np.pi / 2, 100),
    ),
)

SIM_FULL = td.Simulation(
    size=(8.0, 8.0, 8.0),
    run_time=1e-12,
    structures=(
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
                permittivity=td.SpatialDataArray(tracer_arr, coords={"x": [-1], "y": [0], "z": [0]})
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
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.AnisotropicMedium(xx=td.PMC, yy=td.Medium(), zz=td.Medium()),
        ),
        # Test a fully anistropic medium
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.FullyAnisotropicMedium(permittivity=[[6, 2, 3], [2, 7, 4], [3, 4, 9]]),
            name="fully_anisotropic_box",
        ),
        td.Structure(
            geometry=td.GeometryGroup(geometries=(td.Box(size=(1, 1, 1), center=(-1, 0, 0)),)),
            medium=td.PEC,
            name="pec_group",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=td.PMC,
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
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=(-1, 1)
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
                    models=(
                        td.NonlinearSusceptibility(chi3=0.1),
                        td.TwoPhotonAbsorption(
                            beta=1, sigma=1, tau=1, e_e=1, e_h=0.8, c_e=1, c_h=1
                        ),
                        td.KerrNonlinearity(n2=1),
                    ),
                )
            ),
        ),
        td.Structure(
            geometry=td.PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=(-1, 1)
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
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(1, 1, 1)),
            medium=td.MultiPhysicsMedium(
                optical=td.Medium(permittivity=4.0),
                charge=td.ChargeInsulatorMedium(permittivity=2),
                name="SiO2",
            ),
        ),
    ),
    sources=(
        td.UniformCurrentSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=td.GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            current_amplitude_definition="total",
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
                    coords={
                        "x": np.linspace(-1, 1, 101),
                        "y": np.linspace(-1, 1, 101),
                        "z": np.array([0]),
                        "f": [2e14],
                    },
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
                    coords={
                        "x": np.linspace(-1, 1, 101),
                        "y": np.linspace(-1, 1, 101),
                        "z": np.array([0]),
                        "f": [2e14],
                    },
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
            current_amplitude_definition="total",
        ),
    ),
    monitors=(
        *SIM_FULL_COMMON_MONITORS,
        td.DiffractionMonitor(
            size=(0, td.inf, td.inf),
            center=(0, 0, 0),
            name="diffraction",
            freqs=[1e14, 2e14],
        ),
    ),
    lumped_elements=(
        td.LumpedResistor(
            center=(2, 2, 0), size=(0.2, 0.2, 0), name="Resistor", resistance=42, voltage_axis=0
        ),
        td.CoaxialLumpedResistor(
            center=(3, 2, 0),
            outer_diameter=2.0,
            inner_diameter=0.2,
            name="Coax Resistor",
            resistance=42,
            normal_axis=0,
        ),
        td.LinearLumpedElement(
            center=(1, 2, 0),
            size=(0.2, 0.2, 0),
            name="LCParallel",
            network=td.RLCNetwork(inductance=1e-9, capacitance=10e-12, network_topology="parallel"),
            voltage_axis=0,
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
        override_structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
        ),
    ),
)

SIM_FULL_FIELD_PROJECTION = SIM_FULL.updated_copy(
    monitors=(*SIM_FULL_COMMON_MONITORS, *SIM_FULL_PROJECTION_MONITORS),
    boundary_spec=td.BoundarySpec(
        x=td.Boundary(plus=td.PML(num_layers=20), minus=td.Absorber(num_layers=100)),
        y=td.Boundary.pml(num_layers=12),
        z=td.Boundary.pml(num_layers=12),
    ),
)


FULL_STEADY_HEAT = td.HeatChargeSimulation(
    center=(0, 0, 0),
    size=(2, 2, 2),
    medium=td.MultiPhysicsMedium(
        heat=td.FluidMedium(), charge=td.ChargeInsulatorMedium(), name="air"
    ),
    structures=[
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(0, 1, 0)),
            medium=td.MultiPhysicsMedium(
                heat=td.FluidMedium(), charge=td.ChargeInsulatorMedium(), name="temperature0_box"
            ),
            name="temperature0_box",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(0, -1, 0)),
            medium=td.MultiPhysicsMedium(
                heat=td.FluidMedium(), charge=td.ChargeInsulatorMedium(), name="temperature1_box"
            ),
            name="temperature1_box",
        ),
        td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
            medium=td.MultiPhysicsMedium(
                heat=td.SolidMedium.from_si_units(conductivity=1.0, capacity=1.0, density=1.0),
                charge=td.ChargeConductorMedium(conductivity=1.0),
                name="solid_box",
            ),
            name="solid_box",
        ),
    ],
    boundary_spec=[
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["temperature0_box", "solid_box"]),
            condition=td.TemperatureBC(temperature=300.0),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["temperature1_box", "solid_box"]),
            condition=td.TemperatureBC(temperature=320.0),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["air", "solid_box"]),
            condition=td.HeatFluxBC(flux=0.0),
        ),
    ],
    monitors=[
        td.TemperatureMonitor(
            center=(0, 0, 0),
            size=(1, 1, 1),
            unstructured=True,
            name="temperature_monitor",
        )
    ],
    sources=[td.HeatSource(rate=1.0, structures=["solid_box"])],
    grid_spec=td.UniformUnstructuredGrid(dl=0.05),
    symmetry=(1, 0, 0),
)

FULL_UNSTEADY_HEAT = FULL_STEADY_HEAT.updated_copy(
    analysis_spec=td.UnsteadyHeatAnalysis(
        initial_temperature=300.0,
        unsteady_spec=td.UnsteadySpec(time_step=1e-3, total_time_steps=1000),
    )
)


FULL_CONDUCTION = FULL_STEADY_HEAT.updated_copy(
    monitors=[
        td.SteadyPotentialMonitor(
            center=(0, 0, 0), size=(1, 1, 1), name="potential_monitor", unstructured=True
        ),
    ],
    boundary_spec=[
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["temperature0_box", "solid_box"]),
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=5.0)),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["temperature1_box", "solid_box"]),
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0.0)),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["air", "solid_box"]),
            condition=td.InsulatingBC(),
        ),
    ],
    sources=[],
)


FULL_SEMICONDUCTOR = td.SemiconductorMedium(
    permittivity=11,
    N_d=0,
    N_a=0,
    N_c=td.ConstantEffectiveDOS(N=2e19),
    N_v=td.ConstantEffectiveDOS(N=2e19),
    E_g=td.ConstantEnergyBandGap(eg=1.0),
    mobility_n=td.ConstantMobilityModel(mu=1500),
    mobility_p=td.CaugheyThomasMobility(
        mu_min=44.9,
        mu=470.5,
        ref_N=2.23e17,
        exp_N=0.719,
        exp_1=-0.57,
        exp_2=-2.33,
        exp_3=2.4,
        exp_4=-0.146,
    ),
    R=[
        td.ShockleyReedHallRecombination(tau_n=3.3e-6, tau_p=4e-6),
        td.RadiativeRecombination(r_const=1.6e-14),
        td.AugerRecombination(c_n=2.8e-31, c_p=9.9e-32),
    ],
    delta_E_g=td.SlotboomBandGapNarrowing(
        v1=6.92e-3,
        n2=1.3e17,
        c2=0.5,
        min_N=1e15,
    ),
)

FULL_CHARGE = td.HeatChargeSimulation(
    center=(0, 0, 0),
    size=(3, 3, 0),
    medium=td.MultiPhysicsMedium(
        charge=td.ChargeInsulatorMedium(), heat=td.FluidMedium(), name="air"
    ),
    structures=[
        # oxide
        td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(1.999, 2, 1)),
            medium=td.MultiPhysicsMedium(
                heat=td.SolidMedium(conductivity=1.0, capacity=1.0, density=1.0), name="oxide"
            ),
        ),
        # p-side
        td.Structure(
            geometry=td.Box(center=(-0.5, 0, 0), size=(1, 1, 1)),
            medium=FULL_SEMICONDUCTOR.updated_copy(N_a=1e18, name="p_side"),
        ),
        # n-side
        td.Structure(
            geometry=td.Box(center=(0.5, 0, 0), size=(1, 1, 1)),
            medium=FULL_SEMICONDUCTOR.updated_copy(N_d=1e18, name="n_side"),
        ),
    ],
    boundary_spec=[
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["p_side", "air"]),
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[-0.5, 0.0, 1])),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["n_side", "air"]),
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0.0)),
        ),
        td.HeatChargeBoundarySpec(
            placement=td.MediumMediumInterface(mediums=["oxide", "air"]),
            condition=td.InsulatingBC(),
        ),
    ],
    monitors=[
        td.SteadyFreeCarrierMonitor(
            center=(0, 0, 0),
            size=(1, 1, 1),
            name="free_carrier_monitor",
            unstructured=True,
        ),
        td.SteadyPotentialMonitor(
            center=(0, 0, 0),
            size=(1, 1, 1),
            name="potential_monitor",
            unstructured=True,
        ),
        td.SteadyCapacitanceMonitor(
            center=(0, 0, 0),
            size=(1, 1, 1),
            name="capacitance_monitor",
            unstructured=True,
        ),
    ],
    analysis_spec=td.IsothermalSteadyChargeDCAnalysis(
        temperature=300.0,
        convergence_dv=0.1,
        fermi_dirac=False,
        tolerance_settings=td.ChargeToleranceSpec(
            rel_tol=1e-4,
            abs_tol=1e6,
            max_iters=400,
        ),
    ),
    grid_spec=td.UniformUnstructuredGrid(dl=0.05, relative_min_dl=0),
)

SAMPLE_SIMULATIONS = {
    "full_fdtd": SIM_FULL,
    "full_fdtd_field_projection": SIM_FULL_FIELD_PROJECTION,
    "full_steady_heat": FULL_STEADY_HEAT,
    "full_unsteady_heat": FULL_UNSTEADY_HEAT,
    "full_conduction": FULL_CONDUCTION,
    "full_charge": FULL_CHARGE,
}


# Shared synthetic monitor-data generation lives in ``tidy3d._testing`` so other
# repos, including the denormalizer, can import the same helpers.
def get_spatial_coords_dict(simulation: td.Simulation, monitor: td.Monitor, field_name: str):
    """Returns MonitorData coordinates associated with a Monitor object."""

    return _get_spatial_coords_dict(simulation=simulation, monitor=monitor, field_name=field_name)


def run_emulated(simulation: td.Simulation, path=None, **kwargs) -> td.SimulationData:
    """Emulates a simulation run."""

    return _make_simulation_data(
        simulation,
        path=path,
        x0=kwargs.get("x0", 1.0),
        data_gen_fn=DATA_GEN_FN,
    )


def assert_single_value_error_loc(excinfo, expected_loc, message_contains=None):
    """Assert a single pydantic ``value_error`` with the expected location."""
    errors = excinfo.value.errors(include_input=False, include_url=False)
    assert len(errors) == 1
    assert errors[0]["type"] == "value_error"
    assert errors[0]["loc"] == expected_loc
    if message_contains is not None:
        assert message_contains in errors[0]["msg"]


class BatchDataTest(Tidy3dBaseModel):
    """Holds a collection of :class:`.SimulationData` returned by :class:`.Batch`."""

    task_paths: dict[str, str] = Field(
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: dict[str, str] = Field(
        title="Task IDs",
        description="Mapping of task_name to task_id for each task in batch.",
    )

    sim_data: dict[str, td.SimulationData]

    def load_sim_data(self, task_name: str) -> td.SimulationData:
        """Load a :class:`.SimulationData` from file by task name."""
        _ = self.task_paths[task_name]
        _ = self.task_ids[task_name]
        return self.sim_data[task_name]

    def items(self) -> tuple[str, td.SimulationData]:
        """Iterate through the :class:`.SimulationData` for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: str) -> td.SimulationData:
        """Get the :class:`.SimulationData` for a given ``task_name``."""
        return self.load_sim_data(task_name)


def run_async_emulated(simulations: dict[str, td.Simulation], **kwargs) -> BatchData:
    """Emulate an async run function."""
    task_ids = {task_name: f"task_id={i}" for i, task_name in enumerate(simulations.keys())}
    task_paths = dict.fromkeys(simulations.keys(), "NONE")
    sim_data = {task_name: run_emulated(sim) for task_name, sim in simulations.items()}

    return BatchDataTest(task_paths=task_paths, task_ids=task_ids, sim_data=sim_data)


def assert_log_level(
    records: list[tuple[int, str]], log_level_expected: str, contains_str: Optional[str] = None
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


def assert_str_in_log(
    records: list[tuple[int, str]],
    log_level_test: str,
    excludes_str: Optional[str] = None,
    contains_str: Optional[str] = None,
) -> None:
    """Testing tool: Raises error if `excludes_str` appears , or `contains_str` doesn't appear at the test log level.
    Unlike ``assert_log_level``, we don't raise error if the ``log_level_test`` is not present in the records.

    Parameters
    ----------
    records : List[Tuple[int, str]]
        List of (log_level: int, message: str) holding all of the captured logs.
    log_level_test: str
        String version of the log level for checking string (all uppercase).
    excludes_str : str = None
        If specified, errors if found in any of the log messages that are at level
        ``log_level_test``.
    contains_str : str = None
        If specified, errors if not found in any of the log messages that are at level
        ``log_level_test``.

    Returns
    -------
        None
    """

    import sys

    sys.stderr.write(str(records) + "\n")

    # do nothing for None log level
    if log_level_test is None:
        return

    log_level_test_int = _get_level_int(log_level_test)
    contains_str_found = False
    for log in records:
        log_level, log_message = log
        if log_level == log_level_test_int:
            if excludes_str is not None and excludes_str in log_message:
                raise AssertionError(
                    f"Log record at level '{log_level_test}' contained '{excludes_str}'."
                )
            if contains_str is not None and contains_str in log_message:
                contains_str_found = True

    if contains_str and not contains_str_found:
        raise AssertionError(
            f"Log record at level '{log_level_test}' did not contain '{contains_str}'."
        )


class AssertLogLevelHandler:
    """Log handler used to store log records during assertion."""

    def __init__(self):
        self.records = []

    def handle(self, level, level_name, message):
        self.records.append((level, message))


@dataclasses.dataclass
class AbstractAssertLog:
    """Context manager to check logs."""

    log_level_expected: Union[str, None]
    contains_str: str = None

    @property
    def records(self):
        """Get the records from the handler."""
        return self.handler.records if hasattr(self, "handler") else []

    @property
    def num_records(self):
        """Get the number of records."""
        return len(self.records)

    def __enter__(self):
        # Create and register handler
        self.handler = AssertLogLevelHandler()
        td.log.handlers["assert_log_level"] = self.handler
        return self


@dataclasses.dataclass
class AssertLogLevel(AbstractAssertLog):
    """Context manager to check log level for records logged within its context."""

    def __exit__(self, exc_type, exc_value, traceback):
        # Check the records and clean up
        assert_log_level(
            records=self.records,
            log_level_expected=self.log_level_expected,
            contains_str=self.contains_str,
        )
        # Remove handler
        del td.log.handlers["assert_log_level"]


@dataclasses.dataclass
class AssertLogStr(AbstractAssertLog):
    """Context manager to check if log contains certain strings at the test log level for records logged within its context."""

    excludes_str: str = None

    def __exit__(self, exc_type, exc_value, traceback):
        # Check the records and clean up
        assert_str_in_log(
            records=self.records,
            log_level_test=self.log_level_expected,
            excludes_str=self.excludes_str,
            contains_str=self.contains_str,
        )
        # Remove handler
        del td.log.handlers["assert_log_level"]


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

    # Tuples/lists: recurse while preserving the container type
    elif isinstance(nested_obj, (tuple, list)):
        if len(nested_obj) == 0:
            return None
        else:
            return type(nested_obj)(get_nested_shape(val) for val in nested_obj)

    # Replace everything else with None
    else:
        return None
