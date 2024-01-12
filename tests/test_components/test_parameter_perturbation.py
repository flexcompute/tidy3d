"""Tests parameter perturbations."""
import numpy as np
import matplotlib.pyplot as plt
import pytest
import pydantic as pydantic
import tidy3d as td


def test_heat_perturbation():
    perturb = td.LinearHeatPerturbation(
        coeff=0.01,
        temperature_ref=300,
        temperature_range=(200, 400),
    )

    # test automatic calculation of ranges
    assert perturb.perturbation_range == (-100 * 0.01, 100 * 0.01)

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
    sampled = perturb.sample(
        td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6]))
    )
    assert isinstance(sampled, td.SpatialDataArray)

    # complex temperature
    with pytest.raises(ValueError):
        _ = perturb.sample(350j)

    # test plotting
    ax = perturb.plot(temperature=np.linspace(200, 400, 10), val="abs")

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
        sampled = perturb.sample(
            td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6]))
        )
        assert isinstance(sampled, td.SpatialDataArray)

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

    # test sample function on different arguments
    sampled = perturb.sample(electron_density=1e19, hole_density=2e19)
    assert isinstance(sampled, float)
    sampled = perturb.sample(electron_density=[1e17, 1e18, 1e19], hole_density=[2e17, 2e18])
    assert isinstance(sampled, np.ndarray)
    sampled = perturb.sample(electron_density=1e17, hole_density=[2e17, 2e18])
    assert isinstance(sampled, np.ndarray)
    sampled = perturb.sample(
        electron_density=td.SpatialDataArray(
            1e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
        ),
        hole_density=td.SpatialDataArray(
            2e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
        ),
    )
    assert isinstance(sampled, td.SpatialDataArray)
    sampled = perturb.sample(
        electron_density=0,
        hole_density=td.SpatialDataArray(
            2e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
        ),
    )
    assert isinstance(sampled, td.SpatialDataArray)

    # test mixing spatial data array and simple array
    with pytest.raises(ValueError):
        _ = perturb.sample(
            electron_density=td.SpatialDataArray(
                1e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
            ),
            hole_density=[2e17, 2e18],
        )

    with pytest.raises(ValueError):
        _ = perturb.sample(electron_density=1e19j, hole_density=2e19)

    # test plotting
    ax_2d = perturb.plot(
        electron_density=np.logspace(16, 19, 4),
        hole_density=np.logspace(17, 18, 3),
        val="abs",
    )

    ax_1d_e = perturb.plot(
        electron_density=np.logspace(16, 19, 4),
        hole_density=1,
        val="abs",
    )

    ax_1d_h = perturb.plot(
        electron_density=2,
        hole_density=np.logspace(16, 19, 4),
        val="abs",
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

        # test sample function on different arguments
        test_value_in = perturb.sample(electron_density=1.5e18, hole_density=9e17)
        assert isinstance(test_value_in, complex)
        test_value_out = perturb.sample(electron_density=1e19, hole_density=2e19)
        assert isinstance(test_value_out, complex)
        sampled = perturb.sample(electron_density=[1e17, 1e18, 1e19], hole_density=[2e17, 2e18])
        assert isinstance(sampled, np.ndarray)
        sampled = perturb.sample(electron_density=1e17, hole_density=[2e17, 2e18])
        assert isinstance(sampled, np.ndarray)
        sampled = perturb.sample(
            electron_density=td.SpatialDataArray(
                1e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
            ),
            hole_density=td.SpatialDataArray(
                2e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
            ),
        )
        assert isinstance(sampled, td.SpatialDataArray)
        sampled = perturb.sample(
            electron_density=0,
            hole_density=td.SpatialDataArray(
                2e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
            ),
        )
        assert isinstance(sampled, td.SpatialDataArray)

        # test mixing spatial data array and simple array
        with pytest.raises(ValueError):
            _ = perturb.sample(
                electron_density=td.SpatialDataArray(
                    1e18 * np.ones((2, 2, 2)), coords=dict(x=[1, 2], y=[3, 4], z=[5, 6])
                ),
                hole_density=[2e17, 2e18],
            )

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


def test_parameter_perturbation():
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
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, 2)), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.ones((2, 2, 2)), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.ones((2, 2, 2)), coords=coords)

    temperature2 = td.SpatialDataArray(300 * np.ones((2, 2, 1)), coords=coords2)

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
