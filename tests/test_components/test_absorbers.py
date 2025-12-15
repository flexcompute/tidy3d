"""Tests port absorbers."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.boundary import DEFAULT_BROADBAND_MODE_ABC_NUM_FREQS
from tidy3d.exceptions import SetupError

from ..utils import AssertLogLevel


def test_port_absorbers_alone():
    _ = td.InternalAbsorber(
        direction="+", size=(1, 1, 0), boundary_spec=td.ABCBoundary(permittivity=1)
    )

    with pytest.raises(ValidationError):
        _ = td.InternalAbsorber(
            direction="+", size=(1, 1, 1), boundary_spec=td.ABCBoundary(permittivity=1)
        )

    with pytest.raises(ValidationError):
        _ = td.InternalAbsorber(direction="+", size=(1, 1, 0), boundary_spec=td.ABCBoundary())

    absorber = td.InternalAbsorber(
        direction="-",
        size=(1, 1, 0),
        boundary_spec=td.ModeABCBoundary(plane=td.Box(size=(1, 0, 1))),
    )

    absorber.plot(x=0)
    absorber.plot(y=0, alpha=0.4)


def test_port_absorbers_simulations():
    wvl_um = 1
    freq0 = td.C_0 / wvl_um
    mode_source = td.ModeSource(
        size=(1, 1, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        direction="+",
    )

    # in Simulation
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0), direction="-", boundary_spec=td.ABCBoundary(permittivity=1)
            )
        ],
    )

    sim = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0),
                direction="-",
                boundary_spec=td.ModeABCBoundary.from_source(mode_source),
            )
        ],
    )
    sim.plot(x=0)

    # validate no fully anisotropic mediums
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[],
            medium=td.FullyAnisotropicMedium(permittivity=[[2, 0, 0], [0, 1, 0], [0, 0, 3]]),
            run_time=1e-20,
            internal_absorbers=[
                td.InternalAbsorber(
                    size=(0.4, 0.5, 0), direction="-", boundary_spec=td.ABCBoundary(permittivity=1)
                )
            ],
        )

    # disallow ABC ports in zero dimensions
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 0],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[],
            run_time=1e-20,
            internal_absorbers=[
                td.InternalAbsorber(
                    size=(0.4, 0.5, 0), direction="-", boundary_spec=td.ABCBoundary(permittivity=1)
                )
            ],
        )

    # need to define frequency for ModeABCBoundary
    # manually
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0),
                direction="-",
                boundary_spec=td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)), freq_spec=freq0),
            )
        ],
    )
    # or at least one source
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0),
                direction="-",
                boundary_spec=td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0))),
            )
        ],
    )
    # multiple sources with different central freqs is still ok, but show warning
    with AssertLogLevel(
        "WARNING", contains_str="The central frequency of the first source will be used"
    ):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[
                mode_source,
                mode_source.updated_copy(
                    source_time=td.GaussianPulse(freq0=2 * freq0, fwidth=0.2 * freq0)
                ),
            ],
            run_time=1e-20,
            internal_absorbers=[
                td.InternalAbsorber(
                    size=(0.4, 0.5, 0),
                    direction="-",
                    boundary_spec=td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0))),
                )
            ],
        )


def test_abc_boundaries_alone():
    # check basic instance
    _ = td.ABCBoundary()

    # check enforced perm (and conductivity)
    _ = td.ABCBoundary(permittivity=2)
    _ = td.ABCBoundary(permittivity=2, conductivity=0.1)

    with pytest.raises(ValidationError):
        _ = td.ABCBoundary(permittivity=0)

    with pytest.raises(ValidationError):
        _ = td.ABCBoundary(permittivity=2, conductivity=-0.1)

    with pytest.raises(ValidationError):
        _ = td.ABCBoundary(permittivity=None, conductivity=-0.1)

    # test mode abc
    wvl_um = 1
    freq0 = td.C_0 / wvl_um
    mode_abc = td.ModeABCBoundary(
        plane=td.Box(size=(1, 1, 0)),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        freq_spec=freq0,
    )

    with pytest.raises(ValidationError):
        _ = td.ModeABCBoundary(
            plane=td.Box(size=(1, 1, 0)),
            mode_spec=td.ModeSpec(num_modes=2),
            mode_index=1,
            freq_spec=-1,
        )

    with pytest.raises(ValidationError):
        _ = td.ModeABCBoundary(
            plane=td.Box(size=(1, 1, 0)),
            mode_spec=td.ModeSpec(num_modes=2),
            mode_index=-1,
            freq_spec=freq0,
        )

    with pytest.raises(ValidationError):
        _ = td.ModeABCBoundary(
            plane=td.Box(size=(1, 1, 1)),
            mode_spec=td.ModeSpec(num_modes=2),
            mode_index=0,
            freq_spec=freq0,
        )

    # from mode source
    mode_source = td.ModeSource(
        size=(1, 1, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        direction="+",
    )
    mode_abc_from_source = td.ModeABCBoundary.from_source(mode_source)
    assert mode_abc == mode_abc_from_source

    # from mode monitor
    mode_monitor = td.ModeMonitor(
        size=(1, 1, 0), mode_spec=td.ModeSpec(num_modes=2), freqs=[freq0], name="mnt"
    )
    mode_abc_from_monitor = td.ModeABCBoundary.from_monitor(
        mode_monitor, mode_index=1, freq_spec=freq0
    )
    assert mode_abc == mode_abc_from_monitor

    # in Boundary
    _ = td.Boundary(
        minus=td.ABCBoundary(permittivity=3), plus=td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)))
    )
    _ = td.Boundary.abc(permittivity=3, conductivity=1e-5)
    abc_boundary = td.Boundary.mode_abc(
        plane=td.Box(size=(1, 1, 0)),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        freq_spec=freq0,
    )
    abc_boundary_from_source = td.Boundary.mode_abc_from_source(mode_source)
    abc_boundary_from_monitor = td.Boundary.mode_abc_from_monitor(
        mode_monitor, mode_index=1, freq_spec=freq0
    )
    assert abc_boundary == abc_boundary_from_source
    assert abc_boundary == abc_boundary_from_monitor

    with pytest.raises(ValidationError):
        _ = td.Boundary(minus=td.Periodic(), plus=td.ABCBoundary())

    with pytest.raises(ValidationError):
        _ = td.Boundary(minus=td.Periodic(), plus=td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0))))


def test_abc_boundaries_simulations():
    wvl_um = 1
    freq0 = td.C_0 / wvl_um
    mode_source = td.ModeSource(
        size=(1, 1, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        direction="+",
    )

    # in Simulation
    sim = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
    )
    sim.plot(x=0)

    # validate ABC medium is not anisotorpic
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[mode_source],
            medium=td.AnisotropicMedium(xx=td.Medium(), yy=td.Medium(), zz=td.Medium()),
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
        )

    # validate homogeneous medium when permittivity=None, that is, automatic detection
    box_crossing_boundary = td.Structure(
        geometry=td.Box(size=(0.3, 0.2, td.inf)),
        medium=td.Medium(permittivity=2),
    )
    # ok if ABC boundary is not crossed
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        structures=[box_crossing_boundary],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.abc(),
            y=td.Boundary.abc(),
            z=td.Boundary.pml(),
        ),
    )
    # or if we override manually
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        structures=[box_crossing_boundary],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary(permittivity=2)),
    )
    # not ok if ABC boundary is crossed
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[mode_source],
            structures=[box_crossing_boundary],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
        )
    # edge case when a structure exactly coincides with simulation domain
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        structures=[box_crossing_boundary.updated_copy(geometry=td.Box(size=(1, 1, 1)))],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
    )

    # warning for possibly non-uniform custom medium
    with AssertLogLevel(
        "WARNING", contains_str="Nonuniform custom medium detected on an 'ABCBoundary'"
    ):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[mode_source],
            medium=td.CustomMedium(
                permittivity=td.SpatialDataArray(
                    [[[2, 3]]], coords={"x": [0], "y": [0], "z": [0, 1]}
                )
            ),
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
        )

    # disallow ABC boundaries in zero dimensions
    with AssertLogLevel("WARNING", contains_str="The simulation has zero size along the z axis"):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 0],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[mode_source],
            structures=[box_crossing_boundary],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
        )

    # need to define frequency for ModeABCBoundary
    # manually
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(
            td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)), freq_spec=freq0)
        ),
    )
    # or at least one source
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)))),
    )
    # multiple sources with different central freqs is still ok, but show warning
    with AssertLogLevel(
        "WARNING", contains_str="The central frequency of the first source will be used"
    ):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[
                mode_source,
                mode_source.updated_copy(
                    source_time=td.GaussianPulse(freq0=2 * freq0, fwidth=0.2 * freq0)
                ),
            ],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(
                td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)))
            ),
        )
    # error if no frequency
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(
                td.ModeABCBoundary(plane=td.Box(size=(1, 1, 0)))
            ),
        )
    # error if no frequency for automatic abc from interesected mediums
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary()),
        )
    # ok if abc with permittivity only
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary(permittivity=2)),
    )
    _ = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[],
        run_time=1e-20,
        boundary_spec=td.BoundarySpec.all_sides(td.ABCBoundary(permittivity=2, conductivity=0)),
    )
    # not ok if non-zerp conductivity
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=[0, 0, 0],
            size=[1, 1, 1],
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=10,
                wavelength=wvl_um,
            ),
            sources=[],
            run_time=1e-20,
            boundary_spec=td.BoundarySpec.all_sides(
                td.ABCBoundary(permittivity=2, conductivity=1e-5)
            ),
        )


def test_abc_boundaries_broadband():
    # test broadband fitter params
    fitter_params = td.BroadbandModeABCFitterParam(
        max_num_poles=10, tolerance_rms=1e-4, frequency_sampling_points=10
    )

    # test max num poles > 0
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCFitterParam(max_num_poles=0)
    # test max num poles <= 10
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCFitterParam(max_num_poles=11)

    # test tolerance rms >= 0
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCFitterParam(tolerance_rms=-1)

    # test frequency sampling points > 0
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCFitterParam(frequency_sampling_points=0)
    # test frequency sampling points <= 21
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCFitterParam(frequency_sampling_points=102)

    # test basic instance
    fmin = 100e12
    fmax = 200e12
    abc_boundary = td.BroadbandModeABCSpec(frequency_range=(fmin, fmax))
    assert np.allclose(
        abc_boundary._frequency_grid,
        np.linspace(fmin, fmax, DEFAULT_BROADBAND_MODE_ABC_NUM_FREQS),
    )

    # test max frequency > min frequency
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCSpec(frequency_range=(fmax, fmin))

    # test min frequency > 0
    with pytest.raises(ValidationError):
        _ = td.BroadbandModeABCSpec(frequency_range=(0, fmax))

    # test from_wavelength_range
    wvl_min = td.C_0 / fmax
    wvl_max = td.C_0 / fmin
    abc_boundary = td.BroadbandModeABCSpec.from_wavelength_range(
        wavelength_range=(wvl_min, wvl_max), fit_param=fitter_params
    )
    assert abc_boundary.frequency_range == (fmin, fmax)
    assert abc_boundary.fit_param == fitter_params

    with pytest.raises(SetupError):
        _ = td.BroadbandModeABCSpec.from_wavelength_range(
            wavelength_range=(wvl_max, wvl_min), fit_param=fitter_params
        )

    with pytest.raises(SetupError):
        _ = td.BroadbandModeABCSpec.from_wavelength_range(
            wavelength_range=(-1, wvl_min), fit_param=fitter_params
        )
