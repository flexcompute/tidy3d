"""Tests the scene and its validators."""
import pytest
import pydantic
import matplotlib.pyplot as plt

import numpy as np
import tidy3d as td
from tidy3d.components.simulation import MAX_NUM_MEDIUMS
from ..utils import assert_log_level, log_capture, STL_GEO, custom_medium, custom_drude
from ..utils import custom_debye, custom_lorentz, custom_poleresidue, custom_sellmeier

SCENE = td.Scene(size=(1, 1, 1))

SCENE_FULL = td.Scene(
    size=(8.0, 8.0, 8.0),
    structures=[
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
            medium=td.Sellmeier(coeffs=[(1.03961212, 0.00600069867), (0.231792344, 0.0200179144)], name="sellmeier"),
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
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=td.Medium(
                nonlinear_spec=td.NonlinearSusceptibility(chi3=0.1, numiters=20),
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
    ],
)


def test_scene_init():
    """make sure a scene can be initialized"""

    sim = td.Scene(
        size=(2.0, 2.0, 2.0),
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

    _ = sim.mediums
    _ = sim.medium_map
    _ = sim.background_structure


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_scene_bounds(shift_amount, log_level, log_capture):
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.Scene(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, "03b")) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2 * (bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = shift_amount * amp * sign
            if np.sum(center) < 1e-12:
                continue
            place_box(tuple(center))
    assert_log_level(log_capture, log_level)


def test_validate_components_none():

    assert SCENE._structures_not_at_edges(val=None, values=SCENE.dict()) is None
    assert SCENE._validate_num_mediums(val=None) is None


def test_plot_eps():
    ax = SCENE_FULL.plot_eps(x=0)
    SCENE_FULL._add_cbar_eps(eps_min=1, eps_max=2, ax=ax)
    plt.close()


#def test_plot_eps_bounds():
#    _ = SCENE_FULL.plot_eps(x=0, hlim=[-0.45, 0.45])
#    plt.close()
#    _ = SCENE_FULL.plot_eps(x=0, vlim=[-0.45, 0.45])
#    plt.close()
#    _ = SCENE_FULL.plot_eps(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
#    plt.close()


def test_plot():
    SCENE_FULL.plot(x=0)
    plt.close()


def test_plot_1d_scene():
    s = td.Scene(
        size=(0, 0, 1),
    )
    _ = s.plot(y=0)
    plt.close()


#def test_plot_bounds():
#    _ = SCENE_FULL.plot(x=0, hlim=[-0.45, 0.45])
#    plt.close()
#    _ = SCENE_FULL.plot(x=0, vlim=[-0.45, 0.45])
#    plt.close()
#    _ = SCENE_FULL.plot(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
#    plt.close()


#def test_plot_3d():
#    SCENE_FULL.plot_3d()
#    plt.close()


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


@pytest.mark.parametrize(
    "box_size,log_level",
    [
        ((0.1, 0.1, 0.1), None),
        ((1, 0.1, 0.1), "WARNING"),
        ((0.1, 1, 0.1), "WARNING"),
        ((0.1, 0.1, 1), "WARNING"),
    ],
)
def test_scene_structure_extent(log_capture, box_size, log_level):
    """Make sure we warn if structure extends exactly to scene edges."""

    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))
    _ = td.Scene(
        size=(1, 1, 1),
        structures=[box],
    )

    assert_log_level(log_capture, log_level)


def test_num_mediums():
    """Make sure we error if too many mediums supplied."""

    structures = []
    for i in range(MAX_NUM_MEDIUMS):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 1))
        )
    _ = td.Scene(
        size=(5, 5, 5),
        structures=structures,
    )

    with pytest.raises(pydantic.ValidationError):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 2))
        )
        _ = td.Scene(size=(5, 5, 5), structures=structures)


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

    with pytest.raises(pydantic.ValidationError):
        _ = td.Scene(
            size=(2.0, 2.0, 2.0),
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


def test_allow_gain():
    """Test if simulation allows gain."""

    medium = td.Medium(permittivity=2.0)
    medium_gain = td.Medium(permittivity=2.0, allow_gain=True)
    medium_ani = td.AnisotropicMedium(xx=medium, yy=medium, zz=medium)
    medium_gain_ani = td.AnisotropicMedium(xx=medium, yy=medium_gain, zz=medium)

    # Test simulation medium
    scene = td.Scene(size=(10, 10, 10), medium=medium)
    assert not scene.allow_gain
    scene = scene.updated_copy(medium=medium_gain)
    assert scene.allow_gain

    # Test structure with anisotropic gain medium
    struct = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=medium_ani)
    struct_gain = struct.updated_copy(medium=medium_gain_ani)
    scene = td.Scene(
        size=(1, 1, 1),
        medium=medium,
        structures=[struct],
    )
    assert not scene.allow_gain
    scene = scene.updated_copy(structures=[struct_gain])
    assert scene.allow_gain


def test_perturbed_mediums_copy():

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

    coords = dict(x=[1, 2], y=[3, 4], z=[5, 6])
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.ones((2, 2, 2)), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.ones((2, 2, 2)), coords=coords)

    pmed1 = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp_real)

    pmed2 = td.PerturbationPoleResidue(
        poles=[(1j, 3), (2j, 4)],
        poles_perturbation=[(None, pp_real), (pp_complex, None)],
    )

    struct = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=pmed2)

    scene = td.Scene(
        size=(1, 1, 1),
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
