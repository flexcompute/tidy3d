"""Tests mediums."""

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td

from ..utils import AssertLogLevel, cartesian_to_unstructured


@pytest.mark.parametrize("unstructured", [False, True])
def test_perturbation_medium(unstructured):
    # fields to sample at
    coords = dict(x=[1, 2], y=[3, 4], z=[5, 6])
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.ones((2, 2, 2)), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.ones((2, 2, 2)), coords=coords)

    if unstructured:
        temperature = cartesian_to_unstructured(temperature, seed=7747)
        electron_density = cartesian_to_unstructured(electron_density, seed=7747)
        hole_density = cartesian_to_unstructured(hole_density, seed=7747)

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

    # different ways of defining

    with AssertLogLevel(None):
        pmed_direct = td.PerturbationMedium(permittivity=10, permittivity_perturbation=pp_real)
        pmed_perm = td.PerturbationMedium(
            permittivity=10, perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real)
        )
        pmed_index = td.PerturbationMedium(
            permittivity=10,
            perturbation_spec=td.IndexPerturbation(delta_n=pp_real, freq=td.C_0),
        )

    with AssertLogLevel("WARNING"):
        pmed_direct = td.PerturbationMedium(permittivity=1.21, permittivity_perturbation=pp_real)
    with AssertLogLevel("WARNING"):
        pmed_perm = td.PerturbationMedium(
            permittivity=1.21, perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real)
        )
    with AssertLogLevel("WARNING"):
        pmed_index = td.PerturbationMedium(
            permittivity=1.21,
            perturbation_spec=td.IndexPerturbation(delta_n=pp_real, freq=td.C_0),
        )

    # test from_unperturbed function
    pmed_direct_from_med = td.PerturbationMedium.from_unperturbed(
        medium=td.Medium(permittivity=1.21),
        permittivity_perturbation=pp_real,
    )
    pmed_perm_from_med = td.PerturbationMedium.from_unperturbed(
        medium=td.Medium(permittivity=1.21),
        perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
    )

    pmed_index_from_med = td.PerturbationMedium.from_unperturbed(
        medium=td.Medium(permittivity=1.21),
        perturbation_spec=td.IndexPerturbation(delta_n=pp_real, freq=td.C_0),
    )
    assert pmed_direct == pmed_direct_from_med
    assert pmed_perm == pmed_perm_from_med
    assert pmed_index == pmed_index_from_med

    for pmed in [pmed_direct, pmed_perm, pmed_index]:
        cmed = pmed.perturbed_copy()
        # regular medium if no perturbations
        assert isinstance(cmed, td.Medium)

        cmed = pmed.perturbed_copy(temperature, electron_density)
        cmed = pmed.perturbed_copy(temperature, electron_density, hole_density)

        # correct propagation of parameters
        assert cmed.name == pmed.name
        assert cmed.frequency_range == pmed.frequency_range
        assert cmed.subpixel == pmed.subpixel
        assert cmed.allow_gain == pmed.allow_gain

        # permittivity < 1
        with pytest.raises(pydantic.ValidationError):
            _ = pmed.perturbed_copy(1.1 * temperature)

    # conductivity validators
    with AssertLogLevel(None):
        pmed_direct = td.PerturbationMedium(
            conductivity=3, conductivity_perturbation=pp_real, subpixel=False
        )
        pmed_perm = td.PerturbationMedium(
            conductivity=3,
            perturbation_spec=td.PermittivityPerturbation(delta_sigma=pp_real),
            subpixel=False,
        )

    with AssertLogLevel("WARNING"):
        pmed_direct = td.PerturbationMedium(conductivity_perturbation=pp_real, subpixel=False)
    with AssertLogLevel("WARNING"):
        pmed_perm = td.PerturbationMedium(
            perturbation_spec=td.PermittivityPerturbation(delta_sigma=pp_real), subpixel=False
        )
    with AssertLogLevel("WARNING"):
        pmed_index = td.PerturbationMedium(
            permittivity=2,
            perturbation_spec=td.IndexPerturbation(delta_k=pp_real, freq=td.C_0),
            subpixel=False,
        )

    for pmed in [pmed_direct, pmed_perm, pmed_index]:
        cmed = pmed.perturbed_copy(0.9 * temperature)  # positive conductivity
        assert not cmed.subpixel
        with pytest.raises(pydantic.ValidationError):
            _ = pmed.perturbed_copy(1.1 * temperature)  # negative conductivity

        # negative conductivity but allow gain
        pmed = td.PerturbationMedium(conductivity_perturbation=pp_real, allow_gain=True)
        _ = pmed.perturbed_copy(1.1 * temperature)

    # complex perturbation
    with pytest.raises(pydantic.ValidationError):
        pmed = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp_complex)

    # overdefinition
    with pytest.raises(pydantic.ValidationError):
        _ = td.PerturbationMedium(
            permittivity=1.21,
            permittivity_perturbation=pp_real,
            perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
        )

    # Dispersive
    with AssertLogLevel(None):
        pmed_direct = td.PerturbationPoleResidue(
            eps_inf=10,
            poles=[(1j, 3), (2j, 4)],
            eps_inf_perturbation=pp_real,
            poles_perturbation=[(None, pp_real), (pp_complex, None)],
            subpixel=False,
            allow_gain=True,
        )

        pmed_perm = td.PerturbationPoleResidue(
            eps_inf=10,
            poles=[(1j, 3), (2j, 4)],
            perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
            subpixel=False,
            allow_gain=True,
        )

        pmed_index = td.PerturbationPoleResidue(
            eps_inf=10,
            poles=[(1j, 3), (2j, 4)],
            perturbation_spec=td.IndexPerturbation(delta_n=pp_real, freq=td.C_0),
            subpixel=False,
            allow_gain=True,
        )

    with AssertLogLevel("WARNING"):
        pmed_direct = td.PerturbationPoleResidue(
            eps_inf=0.2,
            poles=[(1j, 3), (2j, 4)],
            eps_inf_perturbation=pp_real,
            poles_perturbation=[(None, pp_real), (pp_complex, None)],
            subpixel=False,
            allow_gain=True,
        )

    with AssertLogLevel("WARNING"):
        pmed_perm = td.PerturbationPoleResidue(
            eps_inf=0.2,
            poles=[(1j, 3), (2j, 4)],
            perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
            subpixel=False,
            allow_gain=True,
        )

    with AssertLogLevel("WARNING"):
        pmed_index = td.PerturbationPoleResidue(
            eps_inf=0.05,
            poles=[(0, 0.0001)],
            perturbation_spec=td.IndexPerturbation(delta_k=pp_real, freq=td.C_0),
            subpixel=False,
            allow_gain=True,
        )

    # test from_unperturbed function
    pmed_direct_from_med = td.PerturbationPoleResidue.from_unperturbed(
        medium=td.PoleResidue(
            eps_inf=0.2,
            poles=[(1j, 3), (2j, 4)],
            allow_gain=True,
        ),
        eps_inf_perturbation=pp_real,
        poles_perturbation=[(None, pp_real), (pp_complex, None)],
        subpixel=False,
    )

    pmed_perm_from_med = td.PerturbationPoleResidue.from_unperturbed(
        medium=td.PoleResidue(
            eps_inf=0.2,
            poles=[(1j, 3), (2j, 4)],
            allow_gain=True,
        ),
        perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
        subpixel=False,
    )

    pmed_index_from_med = td.PerturbationPoleResidue.from_unperturbed(
        medium=td.PoleResidue(
            eps_inf=0.05,
            poles=[(0, 0.0001)],
            allow_gain=True,
        ),
        perturbation_spec=td.IndexPerturbation(delta_k=pp_real, freq=td.C_0),
        subpixel=False,
    )

    assert pmed_direct == pmed_direct_from_med
    assert pmed_perm == pmed_perm_from_med
    assert pmed_index == pmed_index_from_med

    for pmed in [pmed_direct, pmed_perm, pmed_index]:
        cmed = pmed.perturbed_copy()
        # regular medium if no perturbations
        assert isinstance(cmed, td.PoleResidue)

        cmed = pmed.perturbed_copy(temperature, None, hole_density)
        cmed = pmed.perturbed_copy(temperature, electron_density, hole_density)

        # correct propagation of parameters
        assert cmed.name == pmed.name
        assert cmed.frequency_range == pmed.frequency_range
        assert cmed.subpixel == pmed.subpixel
        assert cmed.allow_gain == pmed.allow_gain

        # eps_inf < 0
        with pytest.raises(pydantic.ValidationError):
            _ = pmed.perturbed_copy(1.1 * temperature)

    # mismatch between base parameter and perturbations
    with pytest.raises(pydantic.ValidationError):
        pmed = td.PerturbationPoleResidue(
            poles=[(1j, 3), (2j, 4)],
            poles_perturbation=[(None, pp_real)],
        )

    # overdefinition
    with pytest.raises(pydantic.ValidationError):
        _ = td.PerturbationPoleResidue(
            eps_inf=1.21,
            poles=[(1j, 3), (2j, 4)],
            eps_inf_perturbation=pp_real,
            poles_perturbation=[(None, pp_real), (pp_complex, None)],
            perturbation_spec=td.PermittivityPerturbation(delta_eps=pp_real),
            subpixel=False,
            allow_gain=True,
        )


@pytest.mark.parametrize("dispersive", [False, True])
def test_correct_values(dispersive):
    pp_large = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=-0.01,
            temperature_ref=300,
        ),
    )

    pp_small = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=0.0001,
            temperature_ref=300,
        ),
    )

    t_arr = td.SpatialDataArray([[[333]]], coords=dict(x=[0], y=[0], z=[0]))

    pp_large_sampled = pp_large.apply_data(temperature=t_arr).values[0, 0, 0]
    pp_small_sampled = pp_small.apply_data(temperature=t_arr).values[0, 0, 0]

    freq0 = td.C_0 / 0.7
    freqs = np.linspace(td.C_0 / 0.5, td.C_0 / 0.8, 11)

    si = td.material_library["aSi"]["Horiba"]
    perturbation_class = td.PerturbationPoleResidue

    if not dispersive:
        n, k = si.nk_model(freq0)
        si = td.Medium.from_nk(n, k, freq0)
        perturbation_class = td.PerturbationMedium

    si_perm = perturbation_class.from_unperturbed(
        medium=si,
        perturbation_spec=td.PermittivityPerturbation(
            delta_eps=pp_large,
            delta_sigma=pp_small,
        ),
    )

    si_eps = np.real([si.eps_model(freq) for freq in freqs])
    si_sigma = np.real([si.sigma_model(freq) for freq in freqs])

    si_n, si_k = si.nk_model(freq0)

    si_perm_perturb = si_perm.perturbed_copy(temperature=t_arr)

    si_perm_perturb_eps = np.real([si_perm_perturb.eps_model(freq) for freq in freqs])
    si_perm_perturb_sigma = np.real([si_perm_perturb.sigma_model(freq) for freq in freqs])

    print(pp_large_sampled)
    print(pp_small_sampled)
    print(si_eps)
    print(si_perm_perturb_eps)
    print(si_sigma)
    print(si_perm_perturb_sigma)

    assert np.allclose(si_eps + pp_large_sampled, si_perm_perturb_eps)
    assert np.allclose(si_sigma + pp_small_sampled, si_perm_perturb_sigma)

    si_index = perturbation_class.from_unperturbed(
        medium=si,
        perturbation_spec=td.IndexPerturbation(
            delta_n=pp_large,
            delta_k=pp_small,
            freq=freq0,
        ),
    )

    si_index_perturb = si_index.perturbed_copy(temperature=t_arr)

    si_index_perturb_n, si_index_perturb_k = si_index_perturb.nk_model(freq0)

    print(si_n, si_k)
    print(si_index_perturb_n, si_index_perturb_k)

    assert np.isclose(si_n + pp_large_sampled, si_index_perturb_n)
    assert np.isclose(si_k + pp_small_sampled, si_index_perturb_k)
