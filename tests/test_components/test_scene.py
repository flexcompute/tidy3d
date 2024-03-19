"""Tests the scene and its validators."""
import pytest
import pydantic.v1 as pd
import matplotlib.pyplot as plt

import numpy as np
import tidy3d as td
from tidy3d.components.scene import MAX_NUM_MEDIUMS, MAX_GEOMETRY_COUNT
from ..utils import SIM_FULL, cartesian_to_unstructured

SCENE = td.Scene()

SCENE_FULL = SIM_FULL.scene


def test_scene_init():
    """make sure a scene can be initialized"""

    scene = td.Scene(
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(permittivity=1.0, conductivity=3.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=td.Medium()
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ],
        medium=td.Medium(permittivity=3.0),
    )

    _ = scene.mediums
    _ = scene.medium_map
    _ = scene.background_structure


def test_validate_components_none():
    assert SCENE._validate_num_mediums(val=None) is None


def test_plot_eps():
    ax = SCENE_FULL.plot_eps(x=0)
    SCENE_FULL._add_cbar_eps(eps_min=1, eps_max=2, ax=ax)
    plt.close()


def test_plot_eps_bounds():
    _ = SCENE_FULL.plot_eps(x=0, hlim=[-0.45, 0.45])
    plt.close()
    _ = SCENE_FULL.plot_eps(x=0, vlim=[-0.45, 0.45])
    plt.close()
    _ = SCENE_FULL.plot_eps(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
    plt.close()


def test_plot():
    SCENE_FULL.plot(x=0)
    plt.close()


def test_plot_1d_scene():
    s = td.Scene(structures=[td.Structure(geometry=td.Box(size=(0, 0, 1)), medium=td.Medium())])
    _ = s.plot(y=0)
    plt.close()


def test_plot_bounds():
    _ = SCENE_FULL.plot(x=0, hlim=[-0.45, 0.45])
    plt.close()
    _ = SCENE_FULL.plot(x=0, vlim=[-0.45, 0.45])
    plt.close()
    _ = SCENE_FULL.plot(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
    plt.close()


def test_structure_alpha():
    _ = SCENE_FULL.plot_structures_eps(x=0, alpha=None)
    plt.close()
    _ = SCENE_FULL.plot_structures_eps(x=0, alpha=-1)
    plt.close()
    _ = SCENE_FULL.plot_structures_eps(x=0, alpha=1)
    plt.close()
    _ = SCENE_FULL.plot_structures_eps(x=0, alpha=0.5)
    plt.close()
    _ = SCENE_FULL.plot_structures_eps(x=0, alpha=0.5, cbar=True)
    plt.close()
    new_structs = [
        td.Structure(geometry=s.geometry, medium=SCENE_FULL.medium) for s in SCENE_FULL.structures
    ]
    S2 = SCENE_FULL.copy(update=dict(structures=new_structs))
    _ = S2.plot_structures_eps(x=0, alpha=0.5)
    plt.close()


def test_filter_structures():
    s1 = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=SCENE.medium)
    s2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(1, 1, 1)), medium=SCENE.medium)
    plane = td.Box(center=(0, 0, 1.5), size=(td.inf, td.inf, 0))
    SCENE._filter_structures_plane_medium(structures=[s1, s2], plane=plane)


def test_get_structure_plot_params():
    pp = SCENE_FULL._get_structure_plot_params(mat_index=0, medium=SCENE_FULL.medium)
    assert pp.facecolor == "white"
    pp = SCENE_FULL._get_structure_plot_params(mat_index=1, medium=td.PEC)
    assert pp.facecolor == "gold"
    pp = SCENE_FULL._get_structure_eps_plot_params(
        medium=SCENE_FULL.medium, freq=1, eps_min=1, eps_max=2
    )
    assert float(pp.facecolor) == 1.0
    pp = SCENE_FULL._get_structure_eps_plot_params(medium=td.PEC, freq=1, eps_min=1, eps_max=2)
    assert pp.facecolor == "gold"


def test_num_mediums():
    """Make sure we error if too many mediums supplied."""

    structures = []
    for i in range(MAX_NUM_MEDIUMS):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 1))
        )
    _ = td.Scene(
        structures=structures,
    )

    with pytest.raises(pd.ValidationError):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 2))
        )
        _ = td.Scene(structures=structures)


def _test_names_default():
    """makes sure default names are set"""

    scene = td.Scene(
        size=(2.0, 2.0, 2.0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=td.Medium()
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ],
    )

    for i, structure in enumerate(scene.structures):
        assert structure.name == f"structures[{i}]"


def test_names_unique():
    with pytest.raises(pd.ValidationError):
        _ = td.Scene(
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                    medium=td.Medium(permittivity=2.0),
                    name="struct1",
                ),
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                    medium=td.Medium(permittivity=2.0),
                    name="struct1",
                ),
            ],
        )


@pytest.mark.parametrize("z", [[5, 6], [5.5]])
@pytest.mark.parametrize("unstructured", [True, False])
def test_perturbed_mediums_copy(unstructured, z):
    # Non-dispersive
    pp_real = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=-0.01,
            temperature_ref=300,
            temperature_range=(200, 500),
        ),
    )

    pp_complex = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=0.01j,
            temperature_ref=300,
            temperature_range=(200, 500),
        ),
        charge=td.LinearChargePerturbation(
            electron_coeff=-1e-21,
            electron_ref=0,
            electron_range=(0, 1e20),
            hole_coeff=-2e-21,
            hole_ref=0,
            hole_range=(0, 0.5e20),
        ),
    )

    coords = dict(x=[1, 2], y=[3, 4], z=z)
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, len(z))), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.ones((2, 2, len(z))), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.ones((2, 2, len(z))), coords=coords)

    if unstructured:
        seed = 654
        temperature = cartesian_to_unstructured(temperature, seed=seed)
        electron_density = cartesian_to_unstructured(electron_density, seed=seed)
        hole_density = cartesian_to_unstructured(hole_density, seed=seed)

    pmed1 = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp_real)

    pmed2 = td.PerturbationPoleResidue(
        poles=[(1j, 3), (2j, 4)],
        poles_perturbation=[(None, pp_real), (pp_complex, None)],
    )

    struct = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=pmed2)

    scene = td.Scene(
        medium=pmed1,
        structures=[struct],
    )

    # no perturbations provided -> regular mediums
    new_scene = scene.perturbed_mediums_copy()

    assert isinstance(new_scene.medium, td.Medium)
    assert isinstance(new_scene.structures[0].medium, td.PoleResidue)

    # perturbations provided -> custom mediums
    new_scene = scene.perturbed_mediums_copy(temperature)
    new_scene = scene.perturbed_mediums_copy(temperature, None, hole_density)
    new_scene = scene.perturbed_mediums_copy(temperature, electron_density, hole_density)

    assert isinstance(new_scene.medium, td.CustomMedium)
    assert isinstance(new_scene.structures[0].medium, td.CustomPoleResidue)


def test_max_geometry_validation():
    too_many = [td.Box(size=(1, 1, 1)) for _ in range(MAX_GEOMETRY_COUNT + 1)]

    fine = [
        td.Structure(
            geometry=td.ClipOperation(
                operation="union",
                geometry_a=td.Box(size=(1, 1, 1)),
                geometry_b=td.GeometryGroup(geometries=too_many),
            ),
            medium=td.Medium(permittivity=2.0),
        ),
        td.Structure(
            geometry=td.GeometryGroup(geometries=too_many),
            medium=td.Medium(permittivity=2.0),
        ),
    ]
    _ = td.Scene(structures=fine)

    not_fine = [
        td.Structure(
            geometry=td.ClipOperation(
                operation="difference",
                geometry_a=td.Box(size=(1, 1, 1)),
                geometry_b=td.GeometryGroup(geometries=too_many),
            ),
            medium=td.Medium(permittivity=2.0),
        ),
    ]
    with pytest.raises(pd.ValidationError, match=f" {MAX_GEOMETRY_COUNT + 2} "):
        _ = td.Scene(structures=not_fine)
