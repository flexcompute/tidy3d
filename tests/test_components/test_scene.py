"""Tests the scene and its validators."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
import tidy3d.components.scene as scene_mod
from tidy3d.components import scene
from tidy3d.components.viz import STRUCTURE_EPS_CMAP, STRUCTURE_EPS_CMAP_R
from tidy3d.exceptions import SetupError

from ..utils import SIM_FULL, cartesian_to_unstructured

SCENE = td.Scene()

SCENE_FULL = SIM_FULL.scene
TEST_MAX_NUM_MEDIUMS = 3


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
    assert type(SCENE)._validate_mediums(val=None) is None


def test_plot_eps():
    ax = SCENE_FULL.plot_eps(x=0)
    SCENE_FULL._add_cbar_eps(eps_min=1, eps_max=2, ax=ax)
    plt.close()


def test_plot_eps_multiphysics():
    s = td.Scene(
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0.5, 0.5)),
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=3.9),
                    charge=td.ChargeInsulatorMedium(permittivity=3.9),
                    name="SiO2",
                ),
            )
        ]
    )
    assert s.structures[0].medium.name == "SiO2"
    s.plot_eps(x=0)


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
    S2 = SCENE_FULL.copy(update={"structures": tuple(new_structs)})
    _ = S2.plot_structures_eps(x=0, alpha=0.5)
    plt.close()


def test_plot_with_units():
    scene_with_units = SCENE_FULL.updated_copy(plot_length_units="nm")
    scene_with_units.plot(x=-0.5)


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
    expected_color = plt.get_cmap(STRUCTURE_EPS_CMAP)(0.0)
    assert np.allclose(pp.facecolor, expected_color)
    pp = SCENE_FULL._get_structure_eps_plot_params(medium=td.PEC, freq=1, eps_min=1, eps_max=2)
    assert pp.facecolor == "gold"


def test_structure_eps_color_mapping():
    medium_min = td.Medium(permittivity=1.0)
    medium_max = td.Medium(permittivity=5.0)
    norm = mpl.colors.Normalize(vmin=1.0, vmax=5.0)

    pp_min = SCENE_FULL._get_structure_eps_plot_params(
        medium=medium_min,
        freq=1,
        eps_min=1.0,
        eps_max=5.0,
        norm=norm,
        reverse=False,
    )
    expected_min = plt.get_cmap(STRUCTURE_EPS_CMAP)(norm(1.0))
    assert np.allclose(pp_min.facecolor, expected_min)

    pp_max = SCENE_FULL._get_structure_eps_plot_params(
        medium=medium_max,
        freq=1,
        eps_min=1.0,
        eps_max=5.0,
        norm=norm,
        reverse=False,
    )
    expected_max = plt.get_cmap(STRUCTURE_EPS_CMAP)(norm(5.0))
    assert np.allclose(pp_max.facecolor, expected_max)

    pp_min_reverse = SCENE_FULL._get_structure_eps_plot_params(
        medium=medium_min,
        freq=1,
        eps_min=1.0,
        eps_max=5.0,
        norm=norm,
        reverse=True,
    )
    expected_min_reverse = plt.get_cmap(STRUCTURE_EPS_CMAP_R)(norm(1.0))
    assert np.allclose(pp_min_reverse.facecolor, expected_min_reverse)

    pp_max_reverse = SCENE_FULL._get_structure_eps_plot_params(
        medium=medium_max,
        freq=1,
        eps_min=1.0,
        eps_max=5.0,
        norm=norm,
        reverse=True,
    )
    expected_max_reverse = plt.get_cmap(STRUCTURE_EPS_CMAP_R)(norm(5.0))
    assert np.allclose(pp_max_reverse.facecolor, expected_max_reverse)


@pytest.mark.parametrize(
    "medium, eps_min, eps_max, reverse, expected",
    [
        pytest.param(
            td.Medium(permittivity=1.0), 1.0, 5.0, False, (1.0, 1.0, 1.0, 1.0), id="min-forward"
        ),
        pytest.param(
            td.Medium(permittivity=5.0), 1.0, 5.0, False, (0.0, 0.0, 0.0, 1.0), id="max-forward"
        ),
        pytest.param(
            td.Medium(permittivity=1.0), 1.0, 5.0, True, (0.0, 0.0, 0.0, 1.0), id="min-reverse"
        ),
        pytest.param(
            td.Medium(permittivity=5.0), 1.0, 5.0, True, (1.0, 1.0, 1.0, 1.0), id="max-reverse"
        ),
        pytest.param(
            td.Medium(permittivity=2.0), 2.0, 2.0, False, (0.5, 0.5, 0.5, 1.0), id="equal-forward"
        ),
        pytest.param(
            td.Medium(permittivity=2.0), 2.0, 2.0, True, (0.5, 0.5, 0.5, 1.0), id="equal-reverse"
        ),
    ],
)
def test_structure_eps_color_mapping_no_matplotlib(
    monkeypatch, medium, eps_min, eps_max, reverse, expected
):
    monkeypatch.setattr(scene_mod, "mpl", None)

    params = SCENE_FULL._get_structure_eps_plot_params(
        medium=medium,
        freq=1,
        eps_min=eps_min,
        eps_max=eps_max,
        reverse=reverse,
    )

    assert np.allclose(params.facecolor, expected)


def test_num_mediums(monkeypatch):
    """Make sure we error if too many mediums supplied."""
    monkeypatch.setattr(scene, "MAX_NUM_MEDIUMS", TEST_MAX_NUM_MEDIUMS)
    structures = []
    for i in range(TEST_MAX_NUM_MEDIUMS):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 1))
        )
    _ = td.Scene(
        structures=structures,
    )

    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
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

    coords = {"x": [1, 2], "y": [3, 4], "z": z}
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


# def test_max_geometry_validation():
#     too_many = [td.Box(size=(1, 1, 1)) for _ in range(MAX_GEOMETRY_COUNT + 1)]

#     fine = [
#         td.Structure(
#             geometry=td.ClipOperation(
#                 operation="union",
#                 geometry_a=td.Box(size=(1, 1, 1)),
#                 geometry_b=td.GeometryGroup(geometries=too_many),
#             ),
#             medium=td.Medium(permittivity=2.0),
#         ),
#         td.Structure(
#             geometry=td.GeometryGroup(geometries=too_many),
#             medium=td.Medium(permittivity=2.0),
#         ),
#     ]
#     _ = td.Scene(structures=fine)

#     not_fine = [
#         td.Structure(
#             geometry=td.ClipOperation(
#                 operation="difference",
#                 geometry_a=td.Box(size=(1, 1, 1)),
#                 geometry_b=td.GeometryGroup(geometries=too_many),
#             ),
#             medium=td.Medium(permittivity=2.0),
#         ),
#     ]
#     with pytest.raises(ValidationError, match=f" {MAX_GEOMETRY_COUNT + 2} "):
#         _ = td.Scene(structures=not_fine)


def test_structure_manual_priority():
    """make sure structure is properly orderd based on the priority settings."""

    box = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )
    structures = []
    priorities = [2, 4, -1, -4, 0]
    for priority in priorities:
        structures.append(box.updated_copy(priority=priority))
    scene = td.Scene(
        structures=structures,
    )

    sorted_priorities = [s.priority for s in scene.sorted_structures]
    assert all(np.diff(sorted_priorities) >= 0)


def test_structure_automatic_priority():
    """make sure metallic structure has the highest priority in `conductor` mode."""

    box = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )
    box_pec = box.updated_copy(medium=td.PEC)
    box_lossymetal = box.updated_copy(
        medium=td.LossyMetalMedium(conductivity=1.0, frequency_range=(1e14, 2e14))
    )
    structures = [box_pec, box_lossymetal, box]
    scene = td.Scene(
        structures=structures,
        structure_priority_mode="equal",
    )

    # in equal mode, the order is preserved
    scene.sorted_structures == structures

    # conductor mode
    scene = scene.updated_copy(structure_priority_mode="conductor")
    assert scene.sorted_structures[-1].medium == td.PEC
    assert isinstance(scene.sorted_structures[-2].medium, td.LossyMetalMedium)


def test_plot_property():
    """Make sure that plot_structures_property works for different inputs"""

    display_plots = False

    # generate mediums
    semicon = td.material_library["cSi"].variants["Si_MultiPhysics"].medium.charge
    mpm = td.MultiPhysicsMedium(
        optical=td.Medium(permittivity=11.7),
        charge=semicon,
        name="Si_MultiPhysics",
    )

    def try_plotting(mpm, display=False, scale=None):
        # Structure
        struct = td.Structure(
            geometry=td.Box(size=(2, 2, 2), center=(0, 0, 0)),
            medium=mpm,
        )

        # Scene
        scene = td.Scene(
            medium=td.Medium(permittivity=1.0),
            structures=[struct],
        )

        _, ax = plt.subplots(1, 4, figsize=(20, 4))
        if scale:
            scene.plot_structures_property(z=0, property="N_a", ax=ax[0], scale=scale)
            scene.plot_structures_property(z=0, property="N_d", ax=ax[1], scale=scale)
            scene.plot_structures_property(z=0, property="doping", ax=ax[2], scale=scale)
            scene.plot_structures_property(z=0, ax=ax[3], scale=scale)  # eps
        else:
            scene.plot_structures_property(z=0, property="N_a", ax=ax[0])
            scene.plot_structures_property(z=0, property="N_d", ax=ax[1])
            scene.plot_structures_property(z=0, property="doping", ax=ax[2])
            scene.plot_structures_property(z=0, ax=ax[3])  # eps
        if display:
            plt.show()

    # constant doping
    const_doping = td.ConstantDoping(concentration=1e15)
    mpm = mpm.updated_copy(charge=semicon.updated_copy(N_a=[const_doping], N_d=[const_doping]))
    try_plotting(mpm, display=display_plots)
    try_plotting(mpm, display=display_plots, scale="symlog")

    # add some Gaussian doping
    gaussian_box = td.GaussianDoping(
        center=(0, 0, 0), size=(2, 2, 2), ref_con=1e15, concentration=1e18, width=0.1, source="xmin"
    )
    mpm = mpm.updated_copy(charge=semicon.updated_copy(N_a=[gaussian_box], N_d=[gaussian_box]))
    try_plotting(mpm, display=display_plots)
    try_plotting(mpm, display=display_plots, scale="symlog")

    # now try with a custom doping
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    z = np.linspace(-1, 1, 30)
    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    data = X * Y * 1e18
    concentration = td.SpatialDataArray(
        data=data,
        coords={"x": x, "y": y, "z": z},
    )
    # _, ax = plt.subplots()
    # concentration.isel(z=0).plot(ax=ax)
    # plt.show()
    custom_box1 = td.CustomDoping(center=(0, 0, 0), size=(2, 2, 2), concentration=concentration)
    mpm = mpm.updated_copy(charge=semicon.updated_copy(N_a=[custom_box1], N_d=[custom_box1]))
    try_plotting(mpm, display=display_plots)
    try_plotting(mpm, display=display_plots, scale="symlog")


def test_log_scale_with_custom_limits():
    """Test log scale with custom limits."""

    # Create a scene with different permittivity values
    scene = td.Scene(
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=1.0),
                ),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=10.0),
                ),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(1, 0, 0)),
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=100.0),
                ),
            ),
        ],
        medium=td.MultiPhysicsMedium(
            optical=td.Medium(permittivity=1.0),
        ),
    )

    # Test log scale with custom limits
    _ = scene.plot_eps(x=0, scale="log", eps_lim=(1e-2, 100))
    plt.close()

    _ = scene.plot_eps(x=0, scale="log", eps_lim=(1e-5, 100))
    plt.close()

    with pytest.raises(SetupError, match="Log scale cannot be used with non-positive values."):
        _ = scene.plot_eps(x=0, scale="log", eps_lim=(-1e-2, 100))
        plt.close()

    _ = scene.plot_structures_property(x=0, property="eps", scale="log", limits=(1e-2, 100))
    plt.close()

    with pytest.raises(SetupError, match="Log scale cannot be used with non-positive values."):
        _ = scene.plot_structures_property(x=0, property="eps", scale="log", limits=(-2e-2, 100))
        plt.close()

    # Test that invalid scale raises error
    with pytest.raises(
        SetupError, match="The scale 'invalid' is not supported for plotting structures property."
    ):
        _ = scene.plot_structures_property(x=0, property="eps", scale="invalid")
    plt.close()


def test_perturbed_mediums_unique_names():
    """Test that perturbed_mediums_copy generates unique medium names based on structure."""
    from ..utils import AssertLogLevel

    # Setup temperature field for perturbation - large enough to cover both structures
    coords = {"x": [-1, 2], "y": [-1, 2], "z": [-1, 2]}
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=coords)

    # Create a perturbation medium with a name
    pp = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(coeff=0.01, temperature_ref=300),
    )
    pmed = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp, name="Si")

    # Two structures using the same perturbation medium - one with name, one without
    struct1 = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(0.5, 0.5, 0.5)),
        medium=pmed,
        name="core",
    )
    struct2 = td.Structure(
        geometry=td.Box(center=(1, 1, 1), size=(0.5, 0.5, 0.5)),
        medium=pmed,
        # no name - should use index-based suffix
    )

    original_scene = td.Scene(structures=[struct1, struct2])

    # No warning about duplicate names since unique names are generated
    with AssertLogLevel(None):
        perturbed_scene = original_scene.perturbed_mediums_copy(temperature=temperature)

    # Verify the perturbed mediums have unique names based on structure
    med1 = perturbed_scene.structures[0].medium
    med2 = perturbed_scene.structures[1].medium

    assert med1.name == "Si[core]"  # Uses structure name
    assert med2.name == "Si[structures[1]]"  # Uses index since no structure name
    assert med1 is not med2  # Different objects
    # Both derived from the same parent
    assert med1.derived_from == med2.derived_from == pmed

    # Test with unnamed medium - should remain unnamed (no suffix added)
    pmed_unnamed = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp)
    struct_unnamed = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(0.5, 0.5, 0.5)),
        medium=pmed_unnamed,
        name="test",
    )
    scene_unnamed = td.Scene(structures=[struct_unnamed])

    with AssertLogLevel(None):
        perturbed_unnamed = scene_unnamed.perturbed_mediums_copy(temperature=temperature)

    # Unnamed medium should remain unnamed
    assert perturbed_unnamed.structures[0].medium.name is None

    # Test no-op case: calling without perturbation data should NOT rename mediums
    with AssertLogLevel(None):
        no_op_scene = original_scene.perturbed_mediums_copy()  # No perturbation data

    # Medium should be unchanged - same type and name (not renamed with suffix)
    assert isinstance(no_op_scene.structures[0].medium, td.PerturbationMedium)
    assert no_op_scene.structures[0].medium.name == "Si"  # Name unchanged, no suffix added
    assert no_op_scene.structures[1].medium.name == "Si"  # Both still have same name
