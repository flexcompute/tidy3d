"""Tests parameter perturbations."""

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td

from ..utils import AssertLogLevel, cartesian_to_unstructured

sp_arr = td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6]))
sp_arr_u = cartesian_to_unstructured(sp_arr)
sp_arr_2d = td.SpatialDataArray(300 * np.ones((2, 1, 2)), coords=dict(x=[1, 2], y=[3.5], z=[5, 6]))
sp_arr_2d_u = cartesian_to_unstructured(sp_arr_2d)

sp_arrs = [sp_arr, sp_arr_u, sp_arr_2d, sp_arr_2d_u]


def test_heat_perturbation():
    perturb = td.LinearHeatPerturbation(
        coeff=0.01,
        temperature_ref=300,
        temperature_range=(200, 400),
    )

    # test automatic calculation of ranges
    assert perturb.perturbation_range == (-100 * 0.01, 100 * 0.01)

    # when coeff is 0 and temperatrue range is not set
    # check that 0 * inf is handled properly
    for coeff in [0, 0j]:
        perturb_zero_coeff = td.LinearHeatPerturbation(
            coeff=coeff,
            temperature_ref=300,
        )
        assert np.all(perturb_zero_coeff.perturbation_range == (0, 0))

    # test complex type detection
    assert not perturb.is_complex

    with pytest.raises(pydantic.ValidationError):
        perturb = td.LinearHeatPerturbation(
            coeff=0.01,
            temperature_ref=-300,
            temperature_range=(200, 400),
        )

    with pytest.raises(pydantic.ValidationError):
        perturb = td.LinearHeatPerturbation(
            coeff=0.01,
            temperature_ref=300,
            temperature_range=(-200, 400),
        )

    # test sample function on different arguments
    sampled = perturb.sample(350)
    assert isinstance(sampled, float)
    sampled = perturb.sample([310, 320])
    assert isinstance(sampled, np.ndarray)
    sampled = perturb.sample(np.array([310, 320]))
    assert isinstance(sampled, np.ndarray)

    for arr in sp_arrs:
        sampled = perturb.sample(arr)
        assert isinstance(sampled, type(arr))

    # complex temperature
    with pytest.raises(ValueError):
        _ = perturb.sample(350j)

    # test plotting
    for val in ["real", "imag", "abs", "abs^2", "phase"]:
        ax = perturb.plot(temperature=np.linspace(200, 400, 10), val=val)

    # incorrect plotting value
    with pytest.raises(ValueError):
        _ = perturb.plot(temperature=np.linspace(200, 400, 10), val="angle")

    # test custom heat perturbation
    perturb_data = td.HeatDataArray([1 + 1j, 3 + 1j, 1j], coords=dict(T=[200, 300, 400]))

    for interp_method in ["linear", "nearest"]:
        perturb = td.CustomHeatPerturbation(
            perturbation_values=perturb_data, interp_method=interp_method
        )

        # test automatic calculation of ranges
        assert perturb.temperature_range == (200, 400)
        assert perturb.perturbation_range == (1j, 3 + 1j)

        # warning if trying to provide temperature range by hands
        with AssertLogLevel("WARNING"):
            _ = td.CustomHeatPerturbation(
                perturbation_values=perturb_data,
                interp_method=interp_method,
                temperature_range=(300, 500),
            )

        # no warning if temperature range is correct
        with AssertLogLevel(None):
            _ = td.CustomHeatPerturbation(
                perturbation_values=perturb_data,
                interp_method=interp_method,
                temperature_range=(200, 400),
            )

        # test complex type detection
        assert perturb.is_complex

        # test sample function on different arguments
        test_value_in = perturb.sample(380)
        assert isinstance(test_value_in, complex)
        test_value_out = perturb.sample(450)
        assert isinstance(test_value_out, complex)
        sampled = perturb.sample([310, 320])
        assert isinstance(sampled, np.ndarray)
        sampled = perturb.sample(np.array([310, 320]))
        assert isinstance(sampled, np.ndarray)

        for arr in sp_arrs:
            sampled = perturb.sample(arr)
            assert isinstance(sampled, type(arr))

        # test plotting
        perturb.plot(temperature=np.linspace(200, 400, 10), val="real", ax=ax)

        # check interpolation works as expected
        if interp_method == "linear":
            assert test_value_in != perturb_data.data[2]
        elif interp_method == "nearest":
            assert test_value_in == perturb_data.data[2]

        # test clipping works
        assert test_value_out == perturb_data.data[2]

    # test not allowed interpolation method
    with pytest.raises(pydantic.ValidationError):
        perturb = td.CustomHeatPerturbation(
            perturbation_values=perturb_data,
            interp_method="quadratic",
        )

    plt.close("all")


def test_charge_perturbation():
    perturb = td.LinearChargePerturbation(
        electron_coeff=1e-21,
        electron_ref=0,
        electron_range=(0, 1e20),
        hole_coeff=2e-21,
        hole_ref=0,
        hole_range=(0, 0.5e20),
    )

    # test automatic calculation of ranges
    assert perturb.perturbation_range == (0, 1e20 * 1e-21 + 2e20 * 0.5e-21)

    # test complex type detection
    assert not perturb.is_complex

    with pytest.raises(pydantic.ValidationError):
        perturb = td.LinearChargePerturbation(
            electron_coeff=1e-21,
            electron_ref=0,
            electron_range=(0, 1e20),
            hole_coeff=2e-21,
            hole_ref=-1e15,
            hole_range=(0, 0.5e20),
        )

    with pytest.raises(pydantic.ValidationError):
        perturb = td.LinearChargePerturbation(
            electron_coeff=1e-21,
            electron_ref=0,
            electron_range=(-1e15, 1e20),
            hole_coeff=2e-21,
            hole_ref=0,
            hole_range=(0, 0.5e20),
        )

    def test_sample(perturb):
        sampled = perturb.sample(electron_density=[1e17, 1e18, 1e19], hole_density=[2e17, 2e18])
        assert isinstance(sampled, np.ndarray)
        assert np.shape(sampled) == (3, 2)
        sampled = perturb.sample(electron_density=1e17, hole_density=[2e17, 2e18])
        assert isinstance(sampled, np.ndarray)

        for arr in sp_arrs:
            sampled = perturb.sample(electron_density=1e15 * arr, hole_density=2e15 * arr)
            assert isinstance(sampled, type(arr))
            sampled = perturb.sample(electron_density=1e15 * arr, hole_density=0)
            assert isinstance(sampled, type(arr))
            sampled = perturb.sample(electron_density=1e15, hole_density=2e15 * arr)
            assert isinstance(sampled, type(arr))

        # test mixing spatial data array and simple array
        for arr in sp_arrs:
            with pytest.raises(td.exceptions.DataError):
                _ = perturb.sample(
                    electron_density=arr,
                    hole_density=[2e17, 2e18],
                )

        with pytest.raises(td.exceptions.DataError):
            _ = perturb.sample(electron_density=sp_arr, hole_density=sp_arr_u)

        with pytest.raises(td.exceptions.DataError):
            _ = perturb.sample(electron_density=sp_arr, hole_density=sp_arr_2d)

        with pytest.raises(td.exceptions.DataError):
            _ = perturb.sample(electron_density=sp_arr_2d_u, hole_density=sp_arr_u)

        with pytest.raises(ValueError):
            _ = perturb.sample(electron_density=1e19j, hole_density=2e19)

        with pytest.raises(ValueError):
            _ = perturb.sample(electron_density=1e19, hole_density=2e19j)

    # test sample function on different arguments
    test_sample(perturb)
    shape = (3, 4)
    sampled = perturb.sample(electron_density=np.ones(shape), hole_density=np.zeros(shape))
    assert isinstance(sampled, np.ndarray)
    assert np.shape(sampled) == shape
    sampled = perturb.sample(electron_density=1e19, hole_density=2e19)
    assert isinstance(sampled, float)

    # test plotting
    for val in ["real", "imag", "abs", "abs^2", "phase"]:
        ax_2d = perturb.plot(
            electron_density=np.logspace(16, 19, 4),
            hole_density=np.logspace(17, 18, 3),
            val=val,
        )

        ax_1d_e = perturb.plot(
            electron_density=np.logspace(16, 19, 4),
            hole_density=1,
            val=val,
        )

        ax_1d_h = perturb.plot(
            electron_density=2,
            hole_density=np.logspace(16, 19, 4),
            val=val,
        )

    # incorrect plotting value
    with pytest.raises(ValueError):
        _ = perturb.plot(
            electron_density=np.logspace(16, 19, 4),
            hole_density=np.logspace(17, 18, 3),
            val="angle",
        )

    # test custom charge perturbation
    perturb_data = td.ChargeDataArray(
        [[1 + 1j, 3 + 1j, 1j], [2 + 2j, 2j, 2 + 2j]],
        coords=dict(n=[2e17, 2e18], p=[1e16, 1e17, 1e18]),
    )

    for interp_method in ["linear", "nearest"]:
        perturb = td.CustomChargePerturbation(
            perturbation_values=perturb_data, interp_method=interp_method
        )

        # test automatic calculation of ranges
        assert perturb.electron_range == (2e17, 2e18)
        assert perturb.hole_range == (1e16, 1e18)
        assert perturb.perturbation_range == (1j, 3 + 1j)

        # test complex type detection
        assert perturb.is_complex

        # warning if trying to provide density ranges by hands
        with AssertLogLevel("WARNING"):
            _ = td.CustomChargePerturbation(
                perturbation_values=perturb_data,
                interp_method=interp_method,
                electron_range=(1e17, 2e18),
                hole_range=(1e16, 1e18),
            )

        with AssertLogLevel("WARNING"):
            _ = td.CustomChargePerturbation(
                perturbation_values=perturb_data,
                interp_method=interp_method,
                electron_range=(2e17, 2e18),
                hole_range=(1e16, 1e19),
            )

        # no warning if density ranges are correct
        with AssertLogLevel(None):
            _ = td.CustomChargePerturbation(
                perturbation_values=perturb_data,
                interp_method=interp_method,
                electron_range=(2e17, 2e18),
                hole_range=(1e16, 1e18),
            )

        # test sample function on different arguments
        sampled = perturb.sample(electron_density=1e19, hole_density=2e19)
        assert isinstance(sampled, complex)
        test_sample(perturb)

        # test plotting
        _ = perturb.plot(
            electron_density=np.logspace(16, 19, 4),
            hole_density=np.logspace(17, 18, 3),
            val="abs",
            ax=ax_2d,
        )

        _ = perturb.plot(
            electron_density=np.logspace(16, 19, 4),
            hole_density=1,
            val="abs",
            ax=ax_1d_e,
        )

        _ = perturb.plot(
            electron_density=2,
            hole_density=np.logspace(16, 19, 4),
            val="abs",
            ax=ax_1d_h,
        )

        # check interpolation works as expected
        test_value_in = perturb.sample(electron_density=1.5e18, hole_density=9e17)
        assert isinstance(test_value_in, complex)
        test_value_out = perturb.sample(electron_density=1e19, hole_density=2e19)
        assert isinstance(test_value_out, complex)

        if interp_method == "linear":
            assert test_value_in != perturb_data[-1, -1].item()
        elif interp_method == "nearest":
            assert test_value_in == perturb_data[-1, -1].item()

        # check clipping works as expected
        assert test_value_out == perturb_data[-1, -1].item()

    # test not allowed interpolation method
    with pytest.raises(pydantic.ValidationError):
        perturb = td.CustomChargePerturbation(
            perturbation_values=perturb_data,
            interp_method="quadratic",
        )

    plt.close("all")


def test_parameter_perturbation_basic():
    # empty perturbation model
    with pytest.raises(ValueError):
        _ = td.ParameterPerturbation()


@pytest.mark.parametrize("unstructured", [True, False])
def test_parameter_perturbation(unstructured):
    heat = td.LinearHeatPerturbation(
        coeff=0.01,
        temperature_ref=300,
        temperature_range=(200, 400),
    )

    heat_range = (-100 * 0.01, 100 * 0.01)
    charge_range = (1j, 3 + 1j)

    perturb_data = td.ChargeDataArray(
        [[1 + 1j, 3 + 1j, 1j], [2 + 2j, 2j, 2 + 2j]],
        coords=dict(n=[2e17, 2e18], p=[1e16, 1e17, 1e18]),
    )

    charge = td.CustomChargePerturbation(perturbation_values=perturb_data, interp_method="linear")

    coords = dict(x=[1, 2], y=[3, 4], z=[5, 6])
    coords2 = dict(x=[1, 2], y=[3, 4], z=[5])
    temperature = td.SpatialDataArray(300 * np.random.random((2, 2, 2)), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.random.random((2, 2, 2)), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.random.random((2, 2, 2)), coords=coords)

    temperature2 = td.SpatialDataArray(300 * np.random.random((2, 2, 1)), coords=coords2)

    if unstructured:
        seed = 321
        temperature = cartesian_to_unstructured(temperature, seed=seed)
        electron_density = cartesian_to_unstructured(electron_density, seed=seed)
        hole_density = cartesian_to_unstructured(hole_density, seed=seed)
        temperature2 = cartesian_to_unstructured(temperature2, seed=seed)

    param_perturb = td.ParameterPerturbation(
        heat=heat,
    )
    assert param_perturb.perturbation_list == [heat]
    assert not param_perturb.is_complex
    assert param_perturb.perturbation_range == heat_range

    _ = param_perturb.apply_data(temperature, electron_density, hole_density)

    param_perturb = td.ParameterPerturbation(
        charge=charge,
    )
    assert param_perturb.perturbation_list == [charge]
    assert param_perturb.is_complex
    assert param_perturb.perturbation_range == charge_range

    _ = param_perturb.apply_data(temperature, None, hole_density)
    _ = param_perturb.apply_data(temperature, electron_density, None)

    param_perturb = td.ParameterPerturbation(
        heat=heat,
        charge=charge,
    )
    assert param_perturb.perturbation_list == [heat, charge]
    assert param_perturb.is_complex
    assert param_perturb.perturbation_range == tuple(np.array(heat_range) + np.array(charge_range))

    _ = param_perturb.apply_data(None, electron_density, hole_density)

    # array with mismatching coords
    with pytest.raises(ValueError):
        _ = param_perturb.apply_data(temperature2, electron_density, hole_density)

    # no data
    with pytest.raises(ValueError):
        _ = param_perturb.apply_data()


def test_permittivity_perturbation():
    # auxiliary objects
    heat_pb = td.LinearHeatPerturbation(coeff=0.01, temperature_ref=300)

    charge_pb = td.LinearChargePerturbation(
        electron_ref=0,
        electron_coeff=2e-20,
        electron_range=[0, 1e19],
        hole_ref=0,
        hole_coeff=1e-20,
        hole_range=[0, 2e19],
    )

    t_arr = td.SpatialDataArray([[[350]]], coords=dict(x=[0], y=[0], z=[0]))
    n_arr = td.SpatialDataArray([[[1e18]]], coords=dict(x=[0], y=[0], z=[0]))
    p_arr = td.SpatialDataArray([[[2e18]]], coords=dict(x=[0], y=[0], z=[0]))

    # basic make
    perm_pb = td.PermittivityPerturbation(delta_eps=td.ParameterPerturbation(heat=heat_pb))

    delta_eps_range, delta_sigma_range = perm_pb._delta_eps_delta_sigma_ranges()
    assert np.all(delta_eps_range != (0, 0))
    assert np.all(delta_sigma_range == (0, 0))

    delta_eps_sampled, delta_sigma_sampled = perm_pb._sample_delta_eps_delta_sigma(
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )
    assert delta_eps_sampled.values[0, 0, 0] == heat_pb.coeff * (
        t_arr.values[0, 0, 0] - heat_pb.temperature_ref
    )
    assert delta_sigma_sampled is None

    perm_pb = td.PermittivityPerturbation(delta_sigma=td.ParameterPerturbation(charge=charge_pb))

    delta_eps_range, delta_sigma_range = perm_pb._delta_eps_delta_sigma_ranges()
    assert np.all(delta_eps_range == (0, 0))
    assert np.all(delta_sigma_range != (0, 0))

    delta_eps_sampled, delta_sigma_sampled = perm_pb._sample_delta_eps_delta_sigma(
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )
    assert delta_eps_sampled is None
    assert delta_sigma_sampled.values[0, 0, 0] == charge_pb.electron_coeff * (
        n_arr.values[0, 0, 0] - charge_pb.electron_ref
    ) + charge_pb.hole_coeff * (p_arr.values[0, 0, 0] - charge_pb.hole_ref)

    perm_pb = td.PermittivityPerturbation(
        delta_eps=td.ParameterPerturbation(charge=charge_pb),
        delta_sigma=td.ParameterPerturbation(heat=heat_pb),
    )

    delta_eps_range, delta_sigma_range = perm_pb._delta_eps_delta_sigma_ranges()
    assert np.all(delta_eps_range != (0, 0))
    assert np.all(delta_sigma_range != (0, 0))

    delta_eps_sampled, delta_sigma_sampled = perm_pb._sample_delta_eps_delta_sigma(
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )
    assert delta_eps_sampled.values[0, 0, 0] == charge_pb.electron_coeff * (
        n_arr.values[0, 0, 0] - charge_pb.electron_ref
    ) + charge_pb.hole_coeff * (p_arr.values[0, 0, 0] - charge_pb.hole_ref)
    assert delta_sigma_sampled.values[0, 0, 0] == heat_pb.coeff * (
        t_arr.values[0, 0, 0] - heat_pb.temperature_ref
    )

    # empty perturbation model
    with pytest.raises(ValueError):
        _ = td.PermittivityPerturbation()

    # complex perturbations
    with pytest.raises(ValueError):
        _ = td.PermittivityPerturbation(
            delta_eps=td.ParameterPerturbation(
                heat=td.LinearHeatPerturbation(coeff=0.1j, temperature_ref=300)
            )
        )

    with pytest.raises(ValueError):
        _ = td.PermittivityPerturbation(
            delta_sigma=td.ParameterPerturbation(
                heat=td.LinearHeatPerturbation(coeff=0.1j, temperature_ref=300)
            )
        )


def test_index_perturbation():
    # auxiliary objects
    heat_pb = td.LinearHeatPerturbation(coeff=0.01, temperature_ref=300)

    charge_pb = td.LinearChargePerturbation(
        electron_ref=0,
        electron_coeff=2e-20,
        electron_range=[0, 1e19],
        hole_ref=0,
        hole_coeff=1e-20,
        hole_range=[0, 2e19],
    )

    freq0 = td.C_0

    t_arr = td.SpatialDataArray([[[350]]], coords=dict(x=[0], y=[0], z=[0]))
    n_arr = td.SpatialDataArray([[[1e18]]], coords=dict(x=[0], y=[0], z=[0]))
    p_arr = td.SpatialDataArray([[[2e18]]], coords=dict(x=[0], y=[0], z=[0]))

    # basic make
    index_pb = td.IndexPerturbation(delta_n=td.ParameterPerturbation(heat=heat_pb), freq=freq0)

    n, k = 8, 0
    omega0 = 2 * np.pi * freq0

    # test range calculation
    delta_eps_range, delta_sigma_range = index_pb._delta_eps_delta_sigma_ranges(n, k)
    assert np.all(delta_eps_range != (0, 0))
    assert np.all(delta_sigma_range == (0, 0))

    # test sampling
    delta_eps_sampled, delta_sigma_sampled = index_pb._sample_delta_eps_delta_sigma(
        n=n,
        k=k,
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )

    dn = heat_pb.coeff * (t_arr.values[0, 0, 0] - heat_pb.temperature_ref)
    dk = 0
    assert np.isclose(
        delta_eps_sampled.values[0, 0, 0], 2 * n * dn + dn**2 - 2 * k * dk - dk**2, rtol=1e-14
    )
    assert delta_sigma_sampled is None

    index_pb = td.IndexPerturbation(delta_k=td.ParameterPerturbation(charge=charge_pb), freq=freq0)

    # test sampling
    delta_eps_sampled, delta_sigma_sampled = index_pb._sample_delta_eps_delta_sigma(
        n=n,
        k=k,
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )

    dn = 0
    dk = charge_pb.electron_coeff * (
        n_arr.values[0, 0, 0] - charge_pb.electron_ref
    ) + charge_pb.hole_coeff * (p_arr.values[0, 0, 0] - charge_pb.hole_ref)
    assert np.isclose(
        delta_eps_sampled.values[0, 0, 0], 2 * n * dn + dn**2 - 2 * k * dk - dk**2, rtol=1e-14
    )
    assert np.isclose(
        delta_sigma_sampled.values[0, 0, 0],
        2 * omega0 * (k * dn + n * dk + dk * dn) * td.EPSILON_0,
        rtol=1e-14,
    )

    delta_eps_range, delta_sigma_range = index_pb._delta_eps_delta_sigma_ranges(n, k)
    assert np.all(delta_eps_range != (0, 0))
    assert np.all(delta_sigma_range != (0, 0))

    index_pb = td.IndexPerturbation(
        delta_n=td.ParameterPerturbation(charge=charge_pb),
        delta_k=td.ParameterPerturbation(heat=heat_pb),
        freq=freq0,
    )

    n, k = 3, 0.001

    delta_eps_range, delta_sigma_range = index_pb._delta_eps_delta_sigma_ranges(n=n, k=k)
    assert np.all(delta_eps_range != (0, 0))
    assert np.all(delta_sigma_range != (0, 0))

    # test sampling
    delta_eps_sampled, delta_sigma_sampled = index_pb._sample_delta_eps_delta_sigma(
        n=n,
        k=k,
        temperature=t_arr,
        electron_density=n_arr,
        hole_density=p_arr,
    )

    dn = charge_pb.electron_coeff * (
        n_arr.values[0, 0, 0] - charge_pb.electron_ref
    ) + charge_pb.hole_coeff * (p_arr.values[0, 0, 0] - charge_pb.hole_ref)
    dk = heat_pb.coeff * (t_arr.values[0, 0, 0] - heat_pb.temperature_ref)
    assert np.isclose(
        delta_eps_sampled.values[0, 0, 0], 2 * n * dn + dn**2 - 2 * k * dk - dk**2, rtol=1e-14
    )
    assert np.isclose(
        delta_sigma_sampled.values[0, 0, 0],
        2 * omega0 * (k * dn + n * dk + dk * dn) * td.EPSILON_0,
        rtol=1e-14,
    )

    # no freq provided
    with pytest.raises(ValueError):
        _ = td.IndexPerturbation(
            delta_n=td.ParameterPerturbation(charge=charge_pb),
            delta_k=td.ParameterPerturbation(heat=heat_pb),
        )

    # empty perturbation model
    with pytest.raises(ValueError):
        _ = td.IndexPerturbation(freq=freq0)

    # complex perturbations
    with pytest.raises(ValueError):
        _ = td.IndexPerturbation(
            delta_n=td.ParameterPerturbation(
                heat=td.LinearHeatPerturbation(coeff=0.1j, temperature_ref=300)
            ),
            freq=freq0,
        )

    with pytest.raises(ValueError):
        _ = td.PermittivityPerturbation(
            delta_k=td.ParameterPerturbation(
                heat=td.LinearHeatPerturbation(coeff=0.1j, temperature_ref=300)
            ),
            freq=freq0,
        )
