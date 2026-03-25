from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import autograd.numpy as np

import tidy3d as td
from tidy3d.web import run


@dataclass
class PostprocessAdjInputs:
    """Container for the inputs required by ``postprocess_adj``."""

    sim_data_adj: td.SimulationData
    sim_data_orig: td.SimulationData
    sim_data_fwd: td.SimulationData
    sim_fields_keys: tuple[tuple, ...]


DATASET_ROOT = Path(__file__).resolve().parents[3] / "_test_data" / "autograd" / "postprocess_adj"
CUSTOM_MEDIUM_DATASET_ROOT = (
    Path(__file__).resolve().parents[3]
    / "_test_data"
    / "autograd"
    / "postprocess_adj_custom_medium"
)
SIM_DATA_ADJ_NAME = "sim_data_adj.hdf5"
SIM_DATA_ORIG_NAME = "sim_data_orig.hdf5"
SIM_DATA_FWD_NAME = "sim_data_fwd.hdf5"
SIM_FIELDS_KEYS_NAME = "sim_fields_keys.json"


def generate_postprocess_adj_inputs(output_dir: Path) -> PostprocessAdjInputs:
    """Generate inputs for ``postprocess_adj``.

    Parameters
    ----------
    output_dir:
        Directory that can be used to persist intermediate artefacts while generating the dataset.

    Returns
    -------
    PostprocessAdjInputs
        Fully populated structure ready to be passed to ``postprocess_adj``.

    Notes
    -----
    Return ``None`` or raise ``NotImplementedError`` if the dataset cannot be generated in the
    current environment; the caller will convert that into a skipped test instead of a failure.
    """

    N_structures_per_dim = 35
    spacing_per_structure = 0.5
    structure_buffer_per_side = 1

    dim = (N_structures_per_dim - 1) * spacing_per_structure + 2 * structure_buffer_per_side
    dim_z = 4.0

    N_freqs = 2
    wl = 0.65
    freq0 = td.C_0 / wl
    fwidth = 0.2 * freq0

    freqs = np.linspace(freq0 - 0.25 * fwidth, freq0 + 0.25 * fwidth, N_freqs)

    refr_index = 2.5
    permittivity = refr_index**2

    fwd_source = td.PlaneWave(
        center=(0.0, 0.0, -0.25 * dim_z),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
    )

    # dummy adjoint source
    adj_source = td.PlaneWave(
        center=(0.0, 0.0, 0.25 * dim_z),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="-",
    )

    geometries = []
    for x_idx in range(N_structures_per_dim):
        x_pos = -0.5 * dim + structure_buffer_per_side + x_idx * spacing_per_structure
        for y_idx in range(N_structures_per_dim):
            y_pos = -0.5 * dim + structure_buffer_per_side + y_idx * spacing_per_structure

            geometries.append(
                td.Cylinder(
                    center=(x_pos, y_pos, 0.0), radius=0.25 * spacing_per_structure, length=0.5 * wl
                )
            )

    geom_group = td.GeometryGroup(geometries=geometries)

    structure = td.Structure(geometry=geom_group, medium=td.Medium(permittivity=permittivity))

    adj_fld_monitor = td.FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=structure.geometry.bounding_box.size,
        freqs=freqs,
        fields=["Ex", "Ey", "Ez"],
        name="adjoint_fld_0",
        colocate=False,
    )

    adj_perm_monitor = td.PermittivityMonitor(
        center=(0.0, 0.0, 0.0),
        size=structure.geometry.bounding_box.size,
        freqs=freqs,
        name="adjoint_eps_0",
    )

    fwd_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(dim, dim, dim_z),
        structures=[structure],
        monitors=[adj_fld_monitor, adj_perm_monitor],
        sources=[fwd_source],
        run_time=1e-11,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        grid_spec=td.GridSpec.auto(wavelength=wl, min_steps_per_wvl=10),
    )

    adj_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(dim, dim, dim_z),
        structures=[structure],
        monitors=[adj_fld_monitor, adj_perm_monitor],
        sources=[adj_source],
        run_time=1e-11,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        grid_spec=td.GridSpec.auto(wavelength=wl, min_steps_per_wvl=10),
    )

    sim_data_fwd = run(fwd_sim, task_name="perf_sim_fwd")
    sim_data_adj = run(adj_sim, task_name="perf_sim_adj")

    sim_fields_keys = []
    for idx, _ in enumerate(geom_group.geometries):
        sim_fields_keys.append(("structure", 0, "geometry", "geometries", idx, "radius"))

    return PostprocessAdjInputs(
        sim_data_adj=sim_data_adj,
        sim_data_orig=sim_data_fwd,
        sim_data_fwd=sim_data_fwd,
        sim_fields_keys=tuple(sim_fields_keys),
    )


def generate_custom_medium_postprocess_adj_inputs(
    output_dir: Path,
    box_size: tuple[float, float, float] = (2.0, 2.0, 1.0),
    pixel_size: float = 0.05,
    num_freqs: int = 16,
    permittivity: float = 2.25,
    wavelength: float = 1.55,
) -> PostprocessAdjInputs:
    """Generate ``postprocess_adj`` inputs for a ``CustomMedium`` box."""
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive.")
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    freq0 = td.C_0 / wavelength
    fwidth = 0.2 * freq0
    freqs = np.linspace(freq0 - 0.25 * fwidth, freq0 + 0.25 * fwidth, num_freqs)

    def _axis_coords(size_dim: float) -> np.ndarray:
        num_points = max(2, 1 + round(size_dim / pixel_size))
        return np.linspace(-0.5 * size_dim, 0.5 * size_dim, num_points)

    x = _axis_coords(box_size[0])
    y = _axis_coords(box_size[1])
    z = _axis_coords(box_size[2])
    permittivity_data = permittivity * np.ones((len(x), len(y), len(z)))
    coords = {"x": x, "y": y, "z": z}

    structure = td.Structure(
        geometry=td.Box(center=(0.0, 0.0, 0.0), size=box_size),
        medium=td.CustomMedium(permittivity=td.SpatialDataArray(permittivity_data, coords=coords)),
    )

    mesh_overrides = [
        td.MeshOverrideStructure(
            geometry=structure.geometry,
            dl=(pixel_size, pixel_size, pixel_size),
        )
    ]

    sim_size = tuple(3.0 * val for val in box_size)

    fwd_source = td.PlaneWave(
        center=(0.0, 0.0, -0.25 * sim_size[2]),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
    )
    adj_source = td.PlaneWave(
        center=(0.0, 0.0, 0.25 * sim_size[2]),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="-",
    )

    adj_fld_monitor = td.FieldMonitor(
        center=structure.geometry.center,
        size=structure.geometry.size,
        freqs=freqs,
        fields=["Ex", "Ey", "Ez"],
        name="adjoint_fld_0",
        colocate=False,
    )
    adj_perm_monitor = td.PermittivityMonitor(
        center=structure.geometry.center,
        size=structure.geometry.size,
        freqs=freqs,
        name="adjoint_eps_0",
    )

    fwd_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=sim_size,
        structures=[structure],
        monitors=[adj_fld_monitor, adj_perm_monitor],
        sources=[fwd_source],
        run_time=1e-11,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength,
            min_steps_per_wvl=10,
            override_structures=mesh_overrides,
        ),
    )
    adj_sim = fwd_sim.updated_copy(sources=[adj_source])

    sim_data_fwd = run(
        fwd_sim,
        task_name=f"perf_cm_fwd_{num_freqs}freqs",
        path=str(output_dir / SIM_DATA_FWD_NAME),
    )
    sim_data_adj = run(
        adj_sim,
        task_name=f"perf_cm_adj_{num_freqs}freqs",
        path=str(output_dir / SIM_DATA_ADJ_NAME),
    )

    sim_fields_keys = (("structures", 0, "medium", "permittivity"),)
    return PostprocessAdjInputs(
        sim_data_adj=sim_data_adj,
        sim_data_orig=sim_data_fwd,
        sim_data_fwd=sim_data_fwd,
        sim_fields_keys=sim_fields_keys,
    )


def dataset_exists(dataset_dir: Path = DATASET_ROOT) -> bool:
    """Return ``True`` when all persisted artefacts for a dataset are present."""
    required = [
        dataset_dir / SIM_DATA_ADJ_NAME,
        dataset_dir / SIM_DATA_ORIG_NAME,
        dataset_dir / SIM_DATA_FWD_NAME,
        dataset_dir / SIM_FIELDS_KEYS_NAME,
    ]
    return all(path.exists() for path in required)


def persist_postprocess_adj_inputs(
    inputs: PostprocessAdjInputs, dataset_dir: Path = DATASET_ROOT
) -> None:
    """Persist ``postprocess_adj`` inputs to disk for reuse in performance tests."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    inputs.sim_data_adj.to_file(dataset_dir / SIM_DATA_ADJ_NAME)
    inputs.sim_data_orig.to_file(dataset_dir / SIM_DATA_ORIG_NAME)
    inputs.sim_data_fwd.to_file(dataset_dir / SIM_DATA_FWD_NAME)

    serializable_paths = [_serialize_path(path) for path in inputs.sim_fields_keys]
    (dataset_dir / SIM_FIELDS_KEYS_NAME).write_text(json.dumps(serializable_paths, indent=2))


def load_postprocess_adj_inputs(dataset_dir: Path = DATASET_ROOT) -> PostprocessAdjInputs:
    """Load a persisted dataset for ``postprocess_adj`` performance tests."""
    if not dataset_exists(dataset_dir):
        raise FileNotFoundError(
            "Persisted postprocess_adj dataset is missing. "
            "Run the generation flow first to create it."
        )

    sim_data_adj = td.SimulationData.from_file(dataset_dir / SIM_DATA_ADJ_NAME)
    sim_data_orig = td.SimulationData.from_file(dataset_dir / SIM_DATA_ORIG_NAME)
    sim_data_fwd = td.SimulationData.from_file(dataset_dir / SIM_DATA_FWD_NAME)

    serializable = json.loads((dataset_dir / SIM_FIELDS_KEYS_NAME).read_text())
    sim_fields_keys = [_deserialize_path(path) for path in serializable]

    return PostprocessAdjInputs(
        sim_data_adj=sim_data_adj,
        sim_data_orig=sim_data_orig,
        sim_data_fwd=sim_data_fwd,
        sim_fields_keys=tuple(sim_fields_keys),
    )


def _serialize_path(path: Iterable) -> list:
    """Convert a tuple path into JSON serializable data."""
    serialized = []
    for value in path:
        if isinstance(value, (str, int)):
            serialized.append(value)
        elif hasattr(value, "tolist"):
            serialized.append(value.tolist())
        else:
            serialized.append(str(value))
    return serialized


def _deserialize_path(path: Iterable) -> tuple:
    """Convert JSON serialized path data back into tuple form."""
    deserialized = []
    for value in path:
        deserialized.append(value)
    return tuple(deserialized)
