"""Test the array factor functions."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
import tidy3d.plugins.microwave as mw
import tidy3d.plugins.smatrix as smatrix

from ..utils import AssertLogLevel


def analytical_array_factor(array_size, spacings, phase_shifts, theta, phi, freqs, medium):
    """Analytical array factor for a rectangular array."""
    n, k = medium.nk_model(freqs)
    k = 2 * np.pi * freqs * n / td.C_0
    n_x, n_y, n_z = array_size
    d_x, d_y, d_z = spacings
    phi_x, phi_y, phi_z = phase_shifts
    psi_x = k[None, :] * d_x * np.sin(theta[:, None]) * np.cos(phi[:, None]) - phi_x
    psi_y = k[None, :] * d_y * np.sin(theta[:, None]) * np.sin(phi[:, None]) - phi_y
    psi_z = k[None, :] * d_z * np.cos(theta[:, None]) - phi_z
    af_x = np.sin(n_x * psi_x / 2) / np.sin(psi_x / 2)
    af_y = np.sin(n_y * psi_y / 2) / np.sin(psi_y / 2)
    af_z = np.sin(n_z * psi_z / 2) / np.sin(psi_z / 2)
    af_exact = af_x * af_y * af_z
    return af_exact


def test_rectangular_array_calculator_basic():
    """Test the array factor for a rectangular array."""

    n_x = 1
    n_y = 2
    n_z = 3

    d_x = 0.4
    d_y = 0.5
    d_z = 0.6

    phi_x = np.pi / 6
    phi_y = np.pi / 4
    phi_z = np.pi / 3

    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
    )

    # test ._antenna_locations
    assert np.allclose(
        array_calculator._antenna_locations,
        np.array(
            [
                [0.0, -d_y / 2, -d_z],
                [0.0, -d_y / 2, 0.0],
                [0.0, -d_y / 2, d_z],
                [0.0, d_y / 2, -d_z],
                [0.0, d_y / 2, 0.0],
                [0.0, d_y / 2, d_z],
            ]
        ),
    )

    # test ._antenna_amps
    assert np.allclose(array_calculator._antenna_amps, np.ones(n_x * n_y * n_z))

    # test ._antenna_phases
    assert np.allclose(
        array_calculator._antenna_phases,
        np.array(
            [
                0,
                phi_z,
                2 * phi_z,
                phi_y + 0,
                phi_y + phi_z,
                phi_y + 2 * phi_z,
            ]
        ),
    )

    # test ._antenna_nominal_size
    assert np.allclose(
        array_calculator._antenna_nominal_size,
        np.array([(n_x - 1) * d_x, (n_y - 1) * d_y, (n_z - 1) * d_z]),
    )

    # test ._antenna_nominal_center
    assert np.allclose(array_calculator._antenna_nominal_center, np.array([0, 0, 0]))

    # test ._extend_dims
    assert array_calculator._extend_dims == [1, 2]

    with pytest.raises(ValidationError):
        # Test invalid array size
        mw.RectangularAntennaArrayCalculator(
            array_size=(0, 4, 5), spacings=(0.5, 0.5, 0.5), phase_shifts=(0, 0, 0)
        )

    with pytest.raises(ValidationError):
        # Test invalid spacings
        mw.RectangularAntennaArrayCalculator(
            array_size=(3, 4, 5), spacings=(-0.5, 0.5, 0.5), phase_shifts=(0, 0, 0)
        )

    with pytest.raises(ValidationError):
        # Test wrong length of amp_multipliers
        mw.RectangularAntennaArrayCalculator(
            array_size=(3, 4, 5),
            spacings=(0.5, 0.5, 0.5),
            phase_shifts=(0, 0, 0),
            amp_multipliers=(0.5, 0.5),
        )

    for i in range(3):
        with pytest.raises(ValidationError):
            # Test wrong length of amp_multipliers components
            amps = [np.ones(3), np.ones(4), np.ones(5)]
            amps[i] = np.ones(10)
            mw.RectangularAntennaArrayCalculator(
                array_size=(3, 4, 5),
                spacings=(0.5, 0.5, 0.5),
                phase_shifts=(0, 0, 0),
                amp_multipliers=amps,
            )


def test_rectangular_array_calculator_array_factor():
    """Test the array factor for a rectangular array."""

    n_x = 1
    n_y = 2
    n_z = 3

    d_x = 0.4
    d_y = 0.5
    d_z = 0.6

    phi_x = np.pi / 6
    phi_y = np.pi / 4
    phi_z = np.pi / 3

    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
    )

    # Test basic array factor calculation
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()
    medium = td.Medium(permittivity=1)
    freqs = np.array([1e9, 2e9, 3e9])

    af = array_calculator.array_factor(theta_grid, phi_grid, freqs, medium)
    assert af.shape == (100, 3)

    af_exact = analytical_array_factor(
        (n_x, n_y, n_z), (d_x, d_y, d_z), (phi_x, phi_y, phi_z), theta_grid, phi_grid, freqs, medium
    )

    assert np.allclose(np.abs(af), np.abs(af_exact))

    # Test array factor with amplitude multipliers
    array_calculator_amps = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        amp_multipliers=(0.5 * np.ones(n_x), 0.4 * np.ones(n_y), 0.3 * np.ones(n_z)),
    )
    af_amps = array_calculator_amps.array_factor(theta_grid, phi_grid, freqs, medium)
    assert af_amps.shape == (100, 3)
    assert np.allclose(np.abs(af_amps), np.abs(af_exact) * 0.5 * 0.4 * 0.3)

    # Test array factor with non-uniform amplitude multipliers
    # we will use amplitude multipliers that make the array effective size different from nominal size
    n_x = 5
    n_y = 8
    n_z = 9
    array_calculator_amps_nonuniform = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        amp_multipliers=(
            np.int64(np.arange(n_x) % 2 == 0),
            np.int64(np.arange(n_y) % 2 == 0),
            np.int64(np.arange(n_z) % 2 == 0),
        ),
    )

    af_amps_nonuniform = array_calculator_amps_nonuniform.array_factor(
        theta_grid, phi_grid, freqs, medium
    )
    af_amps_nonuniform_exact = analytical_array_factor(
        (3, 4, 5),
        (2 * d_x, 2 * d_y, 2 * d_z),
        (2 * phi_x, 2 * phi_y, 2 * phi_z),
        theta_grid,
        phi_grid,
        freqs,
        medium,
    )
    assert af_amps_nonuniform.shape == (100, 3)
    assert np.allclose(np.abs(af_amps_nonuniform), np.abs(af_amps_nonuniform_exact))

    # Test validation
    with pytest.raises(ValueError):
        # Test mismatched theta/phi lengths
        array_calculator.array_factor(theta, phi[:5], freqs, medium)


def make_antenna_sim():
    # Scaling used for millimeters
    mm = 1e3

    # Frequency range
    freq_start = 5e9
    freq_stop = 15e9

    freq0 = (freq_start + freq_stop) / 2  # Centeral frequency
    freqs_target = np.linspace(freq_start, freq_stop, 5)
    fwidth = freq_stop - freq_start  # Bandwidth

    # Wavelength of centeral frequency in Vaccum
    wavelength0 = td.C_0 / freq0

    # Frequency sweep for S-parameters
    freqs = np.linspace(freq_start, freq_stop, 201)

    # Dipole parameters
    dipole_length = 0.47 * wavelength0
    dipole_radius = wavelength0 / 150

    dipole_width = dipole_radius * 4
    dipole_gap = dipole_width
    arm_length = 0.5 * (dipole_length - dipole_gap)
    arm_center = 0.5 * arm_length + 0.5 * dipole_gap

    pml_buffer = 0.25 * wavelength0

    sim_x = dipole_length + 2 * pml_buffer
    sim_y = dipole_width + 2 * pml_buffer
    sim_z = dipole_width + 2 * pml_buffer

    # Material present
    background_medium = td.Medium(permittivity=2)
    PEC = td.PEC2D  # Thickness-free PEC medium

    arm_a = td.Structure(
        geometry=td.Box(center=[-arm_center, 0, 0], size=[arm_length, dipole_width, 0]),
        medium=PEC,
    )

    arm_b = td.Structure(
        geometry=td.Box(center=[arm_center, 0, 0], size=[arm_length, dipole_width, 0]),
        medium=PEC,
    )

    background = td.Structure(
        geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, td.inf]),
        medium=background_medium,
    )

    background_finite = td.Structure(
        geometry=td.Box(
            center=[0.1 * wavelength0, 0.2 * wavelength0, 0.3 * wavelength0],
            size=[3 * wavelength0, 4 * wavelength0, 5 * wavelength0],
        ),
        medium=background_medium,
    )

    # Define observation parameters used for projection monitors
    theta = np.linspace(0, np.pi, 101)
    phi = np.linspace(0, 2 * np.pi, 101)

    # Field monitor to view the electromagnetic fields in the dipole plane
    monitor_field = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[freq0],
        name="field",
    )

    # Angular projection monitor to view radiation intensity
    monitor_projection = td.FieldProjectionAngleMonitor(
        center=[0, 0, 0],
        size=(dipole_length + pml_buffer, dipole_width + pml_buffer, dipole_width + pml_buffer),
        freqs=freqs_target,
        name="radiation",
        phi=list(phi),
        theta=list(theta),
        far_field_approx=False,
    )

    # We define a DirectivityMonitor to obatin radiation characteritics such as directivity, axial ratio and polarized far fields.
    monitor_directivity = td.DirectivityMonitor(
        center=[0, 0, 0],
        size=monitor_projection.size,
        freqs=freqs_target,
        name="directivity",
        phi=list(phi),
        theta=list(theta),
    )

    # Create the simulation object
    sim = td.Simulation(
        center=(1 * mm, 2 * mm, 1 * mm),
        medium=td.Medium(permittivity=4),
        size=[sim_x, sim_y, sim_z],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,  # The largest cell size is set to 20 cells per smallest wavelength.
            wavelength=td.C_0 / freq_stop,  # Smallest wavelength to resolve
            layer_refinement_specs=[
                td.LayerRefinementSpec(axis=2, size=[0.3 * wavelength0, 0.3 * wavelength0, 0])
            ],
            snapping_points=[(0, 0, 0), (0.25 * wavelength0, 0.3 * wavelength0, None)],
            override_structures=[
                td.MeshOverrideStructure(
                    geometry=td.Box(
                        size=[0.25 * wavelength0, 0.25 * wavelength0, 0.25 * wavelength0]
                    ),
                    dl=[1 * mm, 1 * mm, 1 * mm],
                ),
                td.Structure(
                    geometry=td.Box(
                        size=[0.5 * wavelength0, td.inf, 20 * wavelength0], center=[0, 0, 0]
                    ),
                    medium=td.Medium(permittivity=2),
                ),
            ],
        ),
        structures=[background, background_finite, arm_a, arm_b],
        monitors=[monitor_projection, monitor_field, monitor_directivity],
        run_time=100 * (wavelength0 / td.C_0),
        plot_length_units="mm",  # This option will make plots default to units of millimeters.
    )

    # Define the port impedance
    port_impedance = 50

    # Setup a LumpedPort for the TerminalComponentModeler
    port = smatrix.LumpedPort(
        center=[0, 0, 0],
        size=[dipole_gap, dipole_width, 0],
        voltage_axis=0,
        name="lumped_port",
        impedance=port_impedance,
    )

    # We integrate the base simulation with the LumpedPort using the TerminalComponentModeler.
    modeler = smatrix.TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=freqs,
        remove_dc_component=False,  # Include DC component for more accuracy at low frequencies
    )

    return modeler


def test_rectangular_array_calculator_array_make_antenna_array():
    """Test automatic antenna array creation."""
    freq0 = 10e9
    wavelength0 = td.C_0 / 10e9
    modeler = make_antenna_sim()
    sim_unit = list(modeler.sim_dict.values())[0]
    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(1, 2, 3),
        spacings=(0.5 * wavelength0, 0.6 * wavelength0, 0.4 * wavelength0),
        phase_shifts=(np.pi / 3, np.pi / 4, np.pi / 5),
    )

    # check correctness of the antenna bounds detection
    antenna_bounds = array_calculator._detect_antenna_bounds(sim_unit)
    # we don't extend along x-axis here, so the bounds should be the same as in the original simulation
    # in z-direction antenna has size zero
    assert np.allclose(
        antenna_bounds[0], [sim_unit.bounds[0][0], sim_unit.sources[-1].geometry.bounds[0][1], 0]
    )
    assert np.allclose(
        antenna_bounds[1], [sim_unit.bounds[1][0], sim_unit.sources[-1].geometry.bounds[1][1], 0]
    )

    sim_array = array_calculator.make_antenna_array(sim_unit)

    # check that it generated the right number of structures (2 arms * 6 + 2 boxes extending beoynd the simulation domain, that are automatically extended instead of duplicated)
    assert len(sim_array.structures) == 14

    # check that infinite box is automatically extended
    assert sim_array.structures[0].geometry.center == (0, 0, 0)
    assert sim_array.structures[0].geometry.size == (td.inf, td.inf, td.inf)

    # check that large finite box is automatically extended as well
    # it should go beyond the simulation domain by the same amount as in the original simulation
    old_box_bounds = np.array(sim_unit.structures[1].geometry.bounds)
    new_box_bounds = np.array(sim_array.structures[1].geometry.bounds)
    old_sim_bounds = np.array(sim_unit.bounds)
    new_sim_bounds = np.array(sim_array.bounds)
    assert np.allclose(new_box_bounds[0] - new_sim_bounds[0], old_box_bounds[0] - old_sim_bounds[0])
    assert np.allclose(new_box_bounds[1] - new_sim_bounds[1], old_box_bounds[1] - old_sim_bounds[1])

    # check that we did not duplicate any near-field monitors
    assert len(sim_array.monitors) == 2

    # check that monitors are expanded correctly
    # that is, bounds are located the same distance from simulation domain bounds
    old_monitors = [
        mnt
        for mnt in sim_unit.monitors
        if isinstance(mnt, (td.FieldProjectionAngleMonitor, td.DirectivityMonitor))
    ]
    new_monitors = [
        mnt
        for mnt in sim_array.monitors
        if isinstance(mnt, (td.FieldProjectionAngleMonitor, td.DirectivityMonitor))
    ]
    for old_mnt, new_mnt in zip(old_monitors, new_monitors):
        assert np.allclose(
            np.array(old_mnt.bounds[0]) - np.array(old_sim_bounds[0]),
            np.array(new_mnt.bounds[0]) - np.array(new_sim_bounds[0]),
        )
        assert np.allclose(
            np.array(old_mnt.bounds[1]) - np.array(old_sim_bounds[1]),
            np.array(new_mnt.bounds[1]) - np.array(new_sim_bounds[1]),
        )

    # check sources are duplicated
    assert len(sim_array.sources) == 6

    # check that override_structures are duplicated
    # assert len(sim_unit.grid_spec.override_structures) == 2
    # assert len(sim_array.grid_spec.override_structures) == 7
    assert sim_unit.grid.boundaries == modeler.base_sim.grid.boundaries

    # check that phase shifts are applied correctly
    phases_expected = array_calculator._antenna_phases
    phases_actual = np.array([s.source_time.phase for s in sim_array.sources])
    assert np.allclose(phases_actual, phases_expected)

    # check centers of the sources
    centers_expected = np.array(sim_unit.sources[0].center) + array_calculator._antenna_locations
    centers_actual = np.array([s.center for s in sim_array.sources])
    assert np.allclose(centers_actual, centers_expected)

    # check that lumped elements are duplicated
    assert len(sim_array.lumped_elements) == 6

    # test a case when we have a non-box object extending beyond the simulation domain
    # in this case we will duplicate the object
    background_sphere = td.Structure(
        geometry=td.Sphere(
            center=[0.1 * wavelength0, 0.2 * wavelength0, 0.3 * wavelength0], radius=3 * wavelength0
        ),
        medium=td.Medium(permittivity=2),
    )

    sim_unit_with_sphere = sim_unit.updated_copy(
        structures=[background_sphere, *list(sim_unit.structures)]
    )
    # check correctness of the antenna bounds detection
    antenna_bounds_with_sphere = array_calculator._detect_antenna_bounds(sim_unit_with_sphere)
    assert np.allclose(antenna_bounds_with_sphere[0], sim_unit_with_sphere.bounds[0])
    assert np.allclose(antenna_bounds_with_sphere[1], sim_unit_with_sphere.bounds[1])

    sim_array_with_sphere = array_calculator.make_antenna_array(sim_unit_with_sphere)
    assert len(sim_array_with_sphere.structures) == 20

    # check that projection monitors with zero size are not duplicated
    monitor_projection_flat = td.FieldProjectionAngleMonitor(
        center=[0, 0, 0.1 * wavelength0],
        size=(td.inf, td.inf, 0),
        freqs=[freq0],
        name="radiation",
        phi=[0],
        theta=[0],
        far_field_approx=False,
    )

    sim_unit_with_flat_monitor = sim_unit.updated_copy(monitors=[monitor_projection_flat])

    with AssertLogLevel("WARNING"):
        sim_array_with_flat_monitor = array_calculator.make_antenna_array(
            sim_unit_with_flat_monitor
        )
    assert len(sim_array_with_flat_monitor.monitors) == 0

    sim_unit_with_layer_refinement = sim_unit.updated_copy(
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,
            wavelength=td.C_0 / freq0,
            layer_refinement_specs=[
                td.LayerRefinementSpec(axis=2, size=[0.3 * wavelength0, 0.3 * wavelength0, 0]),
                td.LayerRefinementSpec(axis=0, size=[0.1 * wavelength0, td.inf, 3 * wavelength0]),
            ],
            snapping_points=[(0, 0, 0), (0.25 * wavelength0, 0.3 * wavelength0, None)],
        )
    )

    sim_array_with_layer_refinement = array_calculator.make_antenna_array(
        sim_unit_with_layer_refinement
    )
    assert len(sim_array_with_layer_refinement.grid_spec.layer_refinement_specs) == 7
    assert len(sim_array_with_layer_refinement.grid_spec.snapping_points) == 12


def test_rectangular_array_calculator_monitor_data_from_array_factor():
    """Test that we can get monitor data from array factor."""
    freq0 = 10e9
    wavelength0 = td.C_0 / freq0
    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(1, 2, 3),
        spacings=(0.5 * wavelength0, 0.6 * wavelength0, 0.4 * wavelength0),
        phase_shifts=(np.pi / 3, np.pi / 4, np.pi / 5),
    )

    monitor = td.FieldProjectionAngleMonitor(
        center=[0, 0, 0],
        size=(wavelength0, wavelength0, wavelength0),
        freqs=[freq0, 1.1 * freq0],
        name="radiation",
        phi=list(np.linspace(0, 2 * np.pi, 20)),
        theta=list(np.linspace(0, np.pi, 10)),
        far_field_approx=False,
    )

    new_monitor = td.FieldProjectionAngleMonitor(
        center=[0, 0, 0],
        size=(wavelength0, 2 * wavelength0, 3 * wavelength0),
        freqs=[freq0, 1.1 * freq0],
        name="radiation",
        phi=list(np.linspace(0, 2 * np.pi, 20)),
        theta=list(np.linspace(0, np.pi, 10)),
        far_field_approx=False,
    )

    coords = {
        "r": [monitor.proj_distance],
        "theta": list(monitor.theta),
        "phi": list(monitor.phi),
        "f": list(monitor.freqs),
    }
    values = (1 + 1j) * np.ones(
        (len(coords["r"]), len(coords["theta"]), len(coords["phi"]), len(coords["f"]))
    )
    data = td.FieldProjectionAngleDataArray(values, coords=coords)

    monitor_data = td.FieldProjectionAngleData(
        monitor=monitor,
        Er=data,
        Etheta=data,
        Ephi=data,
        Hr=data,
        Htheta=data,
        Hphi=data,
        projection_surfaces=monitor.projection_surfaces,
    )

    new_monitor_data = array_calculator.monitor_data_from_array_factor(monitor_data)
    assert new_monitor_data.monitor == monitor
    new_monitor_data = array_calculator.monitor_data_from_array_factor(monitor_data, new_monitor)
    assert new_monitor_data.monitor == new_monitor

    # check that array factor is applied to fields
    theta_grid, phi_grid = np.meshgrid(monitor.theta, monitor.phi, indexing="ij")
    array_factor = array_calculator.array_factor(
        theta=theta_grid.ravel(), phi=phi_grid.ravel(), frequency=np.array(monitor.freqs)
    )
    array_factor = np.reshape(
        array_factor, (len(monitor.theta), len(monitor.phi), len(monitor.freqs))
    )
    for _, field in new_monitor_data.field_components.items():
        # print(field.data[0])
        # print(array_factor * (1 + 1j))
        assert np.allclose(field.data[0], array_factor * (1 + 1j))

    # create a directivity monitor and data
    monitor_directivity = td.DirectivityMonitor(
        center=[0, 0, 0],
        size=monitor.size,
        freqs=monitor.freqs,
        name="directivity",
        phi=monitor.phi,
        theta=monitor.theta,
    )

    new_monitor_directivity = td.DirectivityMonitor(
        center=[0, 0, 0],
        size=new_monitor.size,
        freqs=new_monitor.freqs,
        name="directivity",
        phi=new_monitor.phi,
        theta=new_monitor.theta,
    )

    directivity_data = td.DirectivityData(
        monitor=monitor_directivity,
        flux=monitor_data.flux_from_projected_fields(),
        projection_surfaces=monitor_directivity.projection_surfaces,
        Er=data,
        Etheta=data,
        Ephi=data,
        Hr=data,
        Htheta=data,
        Hphi=data,
    )

    new_monitor_data = array_calculator.monitor_data_from_array_factor(directivity_data)
    assert new_monitor_data.monitor == monitor_directivity
    new_monitor_data = array_calculator.monitor_data_from_array_factor(
        directivity_data, new_monitor_directivity
    )
    assert new_monitor_data.monitor == new_monitor_directivity

    # check the case when we don't have enough points in theta and/or phi
    monitor_directivity_under_sampled = td.DirectivityMonitor(
        center=[0, 0, 0],
        size=(wavelength0, wavelength0, wavelength0),
        freqs=[freq0, 1.1 * freq0],
        name="radiation",
        phi=list(np.linspace(0, 2 * np.pi, 10)),
        theta=list(np.linspace(0, np.pi, 10)),
    )
    coords_under_sampled = {
        "r": [monitor_directivity_under_sampled.proj_distance],
        "theta": list(monitor_directivity_under_sampled.theta),
        "phi": list(monitor_directivity_under_sampled.phi),
        "f": list(monitor_directivity_under_sampled.freqs),
    }
    values_under_sampled = (1 + 1j) * np.random.random(
        (
            len(coords_under_sampled["r"]),
            len(coords_under_sampled["theta"]),
            len(coords_under_sampled["phi"]),
            len(coords_under_sampled["f"]),
        )
    )
    data_under_sampled = td.FieldProjectionAngleDataArray(
        values_under_sampled, coords=coords_under_sampled
    )

    directivity_data_under_sampled = td.DirectivityData(
        monitor=monitor_directivity_under_sampled,
        flux=monitor_data.flux_from_projected_fields(),
        projection_surfaces=monitor_directivity_under_sampled.projection_surfaces,
        Er=data_under_sampled,
        Etheta=data_under_sampled,
        Ephi=data_under_sampled,
        Hr=data_under_sampled,
        Htheta=data_under_sampled,
        Hphi=data_under_sampled,
    )
    with AssertLogLevel("WARNING"):
        new_monitor_data_under_sampled = array_calculator.monitor_data_from_array_factor(
            directivity_data_under_sampled
        )
    assert new_monitor_data_under_sampled is None


def test_rectangular_array_calculator_simulation_data_from_array_factor():
    """Test that we can get simulation data from array factor."""
    freq0 = 10e9
    wavelength0 = td.C_0 / freq0
    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(1, 2, 3),
        spacings=(0.5 * wavelength0, 0.6 * wavelength0, 0.4 * wavelength0),
        phase_shifts=(np.pi / 3, np.pi / 4, np.pi / 5),
    )

    modeler = make_antenna_sim()
    sim_unit = list(modeler.sim_dict.values())[0]

    monitor = sim_unit.monitors[0]
    monitor_directivity = sim_unit.monitors[2]
    coords = {
        "r": [monitor.proj_distance],
        "theta": list(monitor.theta),
        "phi": list(monitor.phi),
        "f": list(monitor.freqs),
    }
    values = (1 + 1j) * np.ones(
        (len(coords["r"]), len(coords["theta"]), len(coords["phi"]), len(coords["f"]))
    )
    data = td.FieldProjectionAngleDataArray(values, coords=coords)

    monitor_data = td.FieldProjectionAngleData(
        monitor=monitor,
        Er=data,
        Etheta=data,
        Ephi=data,
        Hr=data,
        Htheta=data,
        Hphi=data,
        projection_surfaces=monitor.projection_surfaces,
    )

    directivity_data = td.DirectivityData(
        monitor=monitor_directivity,
        flux=monitor_data.flux_from_projected_fields(),
        projection_surfaces=monitor_directivity.projection_surfaces,
        Er=data,
        Etheta=data,
        Ephi=data,
        Hr=data,
        Htheta=data,
        Hphi=data,
    )

    sim_data = td.SimulationData(
        simulation=sim_unit.updated_copy(monitors=[monitor, monitor_directivity]),
        data=[monitor_data, directivity_data],
    )

    sim_data_from_array_factor = array_calculator.simulation_data_from_array_factor(sim_data)
    assert len(sim_data_from_array_factor.data) == 2


def test_rectangular_array_calculator_array_factor_taper():
    """Test the array factor for a rectangular array."""

    n_x = 1
    n_y = 2
    n_z = 3

    d_x = 0.4
    d_y = 0.5
    d_z = 0.6

    phi_x = np.pi / 6
    phi_y = np.pi / 4
    phi_z = np.pi / 3

    with pytest.raises(ValidationError):
        # Test for type mismatch
        taper = mw.RadialTaper(window=mw.ChebWindow(attenuation=45))

    array_calculator = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        taper=None,
    )

    # Test basic array factor calculation
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()
    medium = td.Medium(permittivity=1)
    freqs = np.array([1e9, 2e9, 3e9])

    af = array_calculator.array_factor(theta_grid, phi_grid, freqs, medium)
    assert af.shape == (100, 3)

    af_exact = analytical_array_factor(
        (n_x, n_y, n_z), (d_x, d_y, d_z), (phi_x, phi_y, phi_z), theta_grid, phi_grid, freqs, medium
    )

    assert np.allclose(np.abs(af), np.abs(af_exact))

    cheb_window = mw.ChebWindow(attenuation=45)
    taper = mw.RectangularTaper.from_isotropic_window(window=cheb_window)

    # Test array factor with amplitude multipliers
    array_calculator_amps = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        taper=taper,
    )
    af_amps = array_calculator_amps.array_factor(theta_grid, phi_grid, freqs, medium)
    assert af_amps.shape == (100, 3)

    window = mw.TaylorWindow(sll=35, nbar=5)
    taper = mw.RadialTaper(window=window)

    # Test array factor with radial taper
    n_x = 5
    n_y = 8
    n_z = 9
    array_calculator_amps_nonuniform = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        taper=taper,
    )

    af_amps_nonuniform = array_calculator_amps_nonuniform.array_factor(
        theta_grid, phi_grid, freqs, medium
    )

    assert af_amps_nonuniform.shape == (100, 3)

    # test 1D Rectrangular Taper along x
    window = mw.TaylorWindow(sll=35, nbar=5)
    taper = mw.RectangularTaper(window_x=window)

    array_calculator_amps_1d = mw.RectangularAntennaArrayCalculator(
        array_size=(n_x, n_y, n_z),
        spacings=(d_x, d_y, d_z),
        phase_shifts=(phi_x, phi_y, phi_z),
        taper=taper,
    )

    af_amps_1d = array_calculator_amps_1d.array_factor(theta_grid, phi_grid, freqs, medium)

    assert af_amps_1d.shape == (100, 3)

    with pytest.raises(ValidationError):
        # assert that Rectangular Taper has at least one set window
        taper = mw.RectangularTaper()

    # Test validation
    with pytest.raises(ValueError):
        # Test mismatched theta/phi lengths
        array_calculator_amps_nonuniform.array_factor(theta, phi[:5], freqs, medium)
