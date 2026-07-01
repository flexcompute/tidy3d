# test autograd setup-time validation
from __future__ import annotations

import autograd as ag
import numpy as np
import pytest

import tidy3d as td
from tidy3d.exceptions import AdjointError
from tidy3d.plugins.smatrix import ComponentModeler, Port
from tidy3d.web.api.autograd import autograd as autograd_module
from tidy3d.web.api.autograd.autograd import run_custom
from tidy3d.web.api.autograd.types import CustomVJPConfig, NumericalStructureConfig

from ...utils import custom_medium_u
from .test_autograd import FREQ0, FWIDTH, LZ, PLANE_WAVE, SIM_BASE, WVL


def _assert_setup_run_rejects(make_sim, match: str, value: float = 0.1) -> None:
    """Assert a traced setup_run objective rejects before adjoint execution."""

    def objective(param):
        autograd_module.setup_run(simulation=make_sim(param))
        return param

    with pytest.raises(AdjointError, match=match):
        ag.grad(objective)(value)


def _translated_box(x_pos):
    """Build a transformed box with a traced numeric transform entry."""
    transform = np.eye(4).tolist()
    transform[0][3] = x_pos
    geometry = td.Transformed(
        geometry=td.Box(center=(0, 0, 0), size=(0.2, 0.2, 0.2)),
    )
    return geometry.updated_copy(transform=transform, validate=False)


def test_setup_run_rejects_unsupported_traced_source_type():
    """Unsupported traced source types should fail before adjoint setup runs."""

    _assert_setup_run_rejects(
        lambda angle_theta: SIM_BASE.updated_copy(
            sources=(PLANE_WAVE.updated_copy(angle_theta=angle_theta, validate=False),),
            validate=False,
        ),
        r"source parameter 'angle_theta'.*PlaneWave",
    )


def test_is_valid_for_autograd_returns_false_for_unsupported_traced_source_type():
    """Autograd validity predicates should not raise for unsupported traced paths."""

    def objective(pol_angle):
        source = PLANE_WAVE.updated_copy(pol_angle=pol_angle, validate=False)
        sim = SIM_BASE.updated_copy(sources=(source,), validate=False)
        assert not autograd_module.is_valid_for_autograd(sim)
        assert not autograd_module.is_valid_for_autograd_async({"unsupported": sim})
        return pol_angle

    assert ag.grad(objective)(0.1) == 1.0


def test_setup_run_rejects_unsupported_traced_source_path():
    """Supported source VJP types should still reject unsupported traced paths early."""

    gaussian_source = td.GaussianBeam(
        center=(0, 0, -LZ / 2 + WVL),
        size=(0, WVL, WVL),
        direction="+",
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        waist_radius=WVL,
    )

    def make_sim(size_y):
        source = gaussian_source.updated_copy(size=(0, size_y, WVL), validate=False)
        return SIM_BASE.updated_copy(sources=(source,), validate=False)

    _assert_setup_run_rejects(
        make_sim,
        r"Unsupported traced source parameter 'size\[1\]'",
        value=WVL,
    )


@pytest.mark.parametrize(
    ("source_ctor", "dataset_key", "source_size"),
    (
        (td.CustomCurrentSource, "current_dataset", (WVL, WVL, WVL / 10)),
        (td.CustomFieldSource, "field_dataset", (WVL, WVL, 0.0)),
    ),
)
def test_setup_run_rejects_unsupported_custom_source_path(
    source_ctor,
    dataset_key,
    source_size,
):
    """Custom source route validation should reject unsupported source paths."""
    coords = {
        "x": [-source_size[0] / 2, source_size[0] / 2],
        "y": [-source_size[1] / 2, source_size[1] / 2],
        "z": [0.0] if source_size[2] == 0 else [-source_size[2] / 2, source_size[2] / 2],
        "f": [FREQ0],
    }
    scalar_field = td.ScalarFieldDataArray(
        np.ones(tuple(len(coord) for coord in coords.values())),
        coords=coords,
    )
    custom_source = source_ctor(
        center=(0, 0, 0),
        size=source_size,
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        **{dataset_key: td.FieldDataset(Ex=scalar_field)},
    )

    def make_sim(size_x):
        source = custom_source.updated_copy(
            size=(size_x, source_size[1], source_size[2]),
            validate=False,
        )
        return SIM_BASE.updated_copy(sources=(source,), validate=False)

    _assert_setup_run_rejects(
        make_sim,
        r"Unsupported traced source parameter 'size\[0\]'",
        value=source_size[0],
    )


def test_setup_run_rejects_collapsed_axis_center_trace_for_custom_source():
    """Custom source center traces on collapsed axes should fail during setup."""

    coords = {"x": [0.0], "y": [-0.5, 0.5], "z": [-0.5, 0.5], "f": [FREQ0]}
    scalar_field = td.ScalarFieldDataArray(np.ones((1, 2, 2, 1)), coords=coords)
    custom_source = td.CustomCurrentSource(
        center=(0, 0, 0),
        size=(0, WVL, WVL),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        current_dataset=td.FieldDataset(Ex=scalar_field),
    )

    def make_sim(center_x):
        source = custom_source.updated_copy(center=(center_x, 0, 0), validate=False)
        return SIM_BASE.updated_copy(sources=(source,), validate=False)

    _assert_setup_run_rejects(make_sim, r"collapsed axis 'x'.*center\[0\]", value=0.0)


def test_setup_run_rejects_invalid_center_axis_trace_for_custom_source():
    """Malformed source center indices should fail with a clear setup error."""

    coords = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [-0.5, 0.5], "f": [FREQ0]}
    scalar_field = td.ScalarFieldDataArray(np.ones((2, 2, 2, 1)), coords=coords)
    custom_source = td.CustomCurrentSource(
        center=(0, 0, 0),
        size=(WVL, WVL, WVL),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        current_dataset=td.FieldDataset(Ex=scalar_field),
    )

    def make_sim(center_5):
        source = custom_source.updated_copy(
            center=(0, 0, 0, 0, 0, center_5),
            validate=False,
        )
        return SIM_BASE.updated_copy(sources=(source,), validate=False)

    _assert_setup_run_rejects(
        make_sim,
        r"Unsupported traced source parameter 'center\[5\]'.*center\[2\]",
        value=0.0,
    )


def test_custom_source_route_rejects_missing_dataset_component():
    """Custom source dataset routes should validate against the concrete dataset."""
    coords = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [-0.5, 0.5], "f": [FREQ0]}
    scalar_field = td.ScalarFieldDataArray(np.ones((2, 2, 2, 1)), coords=coords)
    custom_source = td.CustomCurrentSource(
        center=(0, 0, 0),
        size=(WVL, WVL, WVL),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
        current_dataset=td.FieldDataset(Ex=scalar_field),
    )

    with pytest.raises(AdjointError, match=r"current_dataset\.Ey.*not present"):
        custom_source._resolve_autograd_route(("current_dataset", "Ey"))


@pytest.mark.parametrize(
    ("make_structure", "expected_message", "value"),
    [
        (
            lambda param: SIM_BASE.structures[0].updated_copy(
                geometry=td.GeometryGroup(
                    geometries=(
                        td.Box(center=(-0.3, 0, 0), size=(0.2, 0.2, 0.2)),
                        _translated_box(param),
                    )
                ),
                validate=False,
            ),
            r"geometry parameter 'geometries\[1\]\.transform\[0\]\[3\]'.*Transformed",
            0.1,
        ),
        (
            lambda param: SIM_BASE.structures[0].updated_copy(
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=param),
                ),
                validate=False,
            ),
            r"medium parameter 'optical\.permittivity'.*MultiPhysicsMedium",
            2.0,
        ),
    ],
)
def test_setup_run_rejects_unsupported_traced_structure_paths(
    make_structure,
    expected_message,
    value,
):
    """Unsupported native structure paths should fail during setup."""

    _assert_setup_run_rejects(
        lambda param: SIM_BASE.updated_copy(
            structures=(make_structure(param),),
            validate=False,
        ),
        expected_message,
        value=value,
    )


def test_setup_run_rejects_unstructured_custom_medium_trace():
    """Unstructured custom medium data should fail before adjoint postprocessing."""

    def make_sim(scale):
        traced_permittivity = custom_medium_u.permittivity.updated_copy(
            values=custom_medium_u.permittivity.values * scale,
            validate=False,
        )
        medium = custom_medium_u.updated_copy(
            permittivity=traced_permittivity,
            validate=False,
        )
        structure = SIM_BASE.structures[0].updated_copy(
            medium=medium,
            validate=False,
        )
        return SIM_BASE.updated_copy(structures=(structure,), validate=False)

    _assert_setup_run_rejects(
        make_sim,
        r"medium parameter 'permittivity'.*CustomMedium.*unstructured",
        value=1.0,
    )


def test_setup_run_allows_custom_vjp_for_unsupported_structure_path():
    """Custom VJP paths should bypass native medium/geometry support validation."""

    def custom_transform_vjp(_geometry, derivative_info):
        return dict.fromkeys(derivative_info.paths, 0.0)

    custom_vjp = CustomVJPConfig(
        structure=0,
        path_key=("geometry", "transform"),
        compute_derivatives=custom_transform_vjp,
    )

    def objective(x_pos):
        structure = SIM_BASE.structures[0].updated_copy(
            geometry=_translated_box(x_pos),
            validate=False,
        )
        sim = SIM_BASE.updated_copy(structures=(structure,), validate=False)

        setup_result = autograd_module.setup_run(
            simulation=sim,
            custom_vjp=custom_vjp,
        )
        assert ("structures", 0, "geometry", "transform", 0, 3) in setup_result.sim_fields
        return x_pos

    assert ag.grad(objective)(0.1) == 1.0


def test_setup_run_expands_type_custom_vjp_before_numerical_structures(monkeypatch):
    """Type-based custom VJPs should not match synthetic numerical structures."""
    captured_num_structures = []
    expand_custom_vjp = autograd_module.expand_custom_vjp

    def capture_expand(custom_vjp, simulation):
        captured_num_structures.append(len(simulation.structures))
        return expand_custom_vjp(custom_vjp, simulation)

    def custom_box_vjp(_geometry, derivative_info):
        return dict.fromkeys(derivative_info.paths, 0.0)

    def create_structure(parameters):
        return td.Structure(
            geometry=td.Box(size=(parameters[0], WVL, WVL)),
            medium=td.Medium(permittivity=2.0),
        )

    monkeypatch.setattr(autograd_module, "expand_custom_vjp", capture_expand)

    numerical_structure = NumericalStructureConfig(
        create=create_structure,
        compute_derivatives=lambda _parameters, derivative_info: dict.fromkeys(
            derivative_info.paths, 0.0
        ),
        parameters=np.array([WVL]),
    )
    custom_vjp = CustomVJPConfig(
        structure=td.Box,
        path_key=("geometry", "center"),
        compute_derivatives=custom_box_vjp,
    )

    setup_result = autograd_module.setup_run(
        simulation=SIM_BASE,
        numerical_structures=(numerical_structure,),
        custom_vjp=custom_vjp,
    )

    assert captured_num_structures == [len(SIM_BASE.structures)]
    assert len(setup_result.simulation.structures) == len(SIM_BASE.structures) + 1


def test_setup_run_counts_traced_numerical_structures(monkeypatch):
    """Numerical-namespace traces should count toward the traced-structure limit."""

    monkeypatch.setattr(td.config.adjoint, "max_traced_structures", 1)

    def create_structure(parameters):
        return td.Structure(
            geometry=td.Box(size=(parameters[0], WVL, WVL)),
            medium=td.Medium(permittivity=2.0),
        )

    def objective(size_x):
        numerical_structure = NumericalStructureConfig(
            create=create_structure,
            compute_derivatives=lambda _parameters, derivative_info: dict.fromkeys(
                derivative_info.paths, 0.0
            ),
            parameters=np.array([size_x]),
        )
        autograd_module.setup_run(
            simulation=SIM_BASE,
            numerical_structures=(numerical_structure, numerical_structure),
        )
        return size_x

    with pytest.raises(
        AdjointError,
        match=r"limited to 1 structures with traced fields.*Found 2 structures",
    ):
        ag.grad(objective)(WVL)


@pytest.mark.parametrize("entry_point", ["run_custom", "smatrix_run_local"])
def test_component_modeler_rejects_unsupported_traced_fields_before_submission(
    monkeypatch,
    entry_point,
):
    """ComponentModeler entry points should eagerly reject unsupported generated traces."""

    from tidy3d.plugins.smatrix import run as smatrix_run

    def reject_submission(*args, **kwargs):
        raise AssertionError("unsupported trace should fail before submission")

    base_modeler = ComponentModeler(
        simulation=SIM_BASE.updated_copy(sources=(), monitors=(), validate=False),
        ports=[
            Port(
                center=(0, 0, -LZ / 2 + WVL),
                size=(WVL, WVL, 0),
                direction="+",
                mode_spec=td.ModeSpec(num_modes=1),
                name="input_port",
            )
        ],
        freqs=[FREQ0],
    )
    sim_dict_holder = {}
    monkeypatch.setattr(
        type(base_modeler),
        "sim_dict",
        property(lambda _self: sim_dict_holder["sim_dict"]),
    )

    def objective(pol_angle):
        bad_source = PLANE_WAVE.updated_copy(pol_angle=pol_angle, validate=False)
        bad_sim = SIM_BASE.updated_copy(sources=(bad_source,), validate=False)
        plain_sim = SIM_BASE.updated_copy(sources=(), validate=False)
        sim_dict_holder["sim_dict"] = {"plain": plain_sim, "unsupported": bad_sim}

        if entry_point == "run_custom":
            monkeypatch.setattr(smatrix_run, "_run_local", reject_submission)
            run_custom(
                base_modeler,
                task_name="component_unsupported_trace",
                local_gradient=True,
            )
        else:
            monkeypatch.setattr(autograd_module, "_run_async", reject_submission)
            smatrix_run._run_local(base_modeler, local_gradient=True)
        return pol_angle

    with pytest.raises(AdjointError, match=r"source parameter 'pol_angle'.*PlaneWave"):
        ag.grad(objective)(0.1)


def test_public_run_async_uses_autograd_for_supported_mixed_batch(monkeypatch):
    """Mixed async batches should use autograd when any simulation has supported traces."""
    captured = {}

    def capture_autograd_async(**kwargs):
        captured.update(kwargs)
        return {"plain": "plain_data", "traced": "traced_data"}

    def reject_regular_async(*args, **kwargs):
        raise AssertionError("supported traced batch should use autograd async")

    monkeypatch.setattr(autograd_module, "_run_async", capture_autograd_async)
    monkeypatch.setattr(td.web.api.asynchronous, "run_async", reject_regular_async)

    def objective(size_y):
        geometry = td.Box(center=(0, 0, 0), size=(0.5, WVL, LZ / 2)).updated_copy(
            size=(0.5, size_y, LZ / 2),
            validate=False,
        )
        structure = SIM_BASE.structures[0].updated_copy(
            geometry=geometry,
            validate=False,
        )
        traced_sim = SIM_BASE.updated_copy(structures=(structure,), validate=False)
        simulations = {"plain": SIM_BASE, "traced": traced_sim}

        assert autograd_module.is_valid_for_autograd_async(simulations)
        data = td.web.run_async(simulations, verbose=False)
        assert set(data) == {"plain", "traced"}
        return size_y

    assert ag.grad(objective)(WVL) == 1.0
    assert set(captured["simulations"]) == {"plain", "traced"}
    assert captured["setup_results"]["plain"].sim_fields == {}
    assert ("structures", 0, "geometry", "size", 1) in captured["setup_results"][
        "traced"
    ].sim_fields


def test_public_run_async_rejects_traced_fdtd_mixed_with_eme_before_submission(monkeypatch):
    """Traced FDTD tasks should not fall through regular async in mixed simulation batches."""

    def reject_submission(*args, **kwargs):
        raise AssertionError("mixed traced FDTD batch should fail before submission")

    monkeypatch.setattr(autograd_module, "_run_async", reject_submission)
    monkeypatch.setattr(td.web.api.asynchronous, "run_async", reject_submission)

    eme_sim = td.EMESimulation(
        size=(WVL, WVL, WVL),
        grid_spec=td.GridSpec.auto(wavelength=WVL, min_steps_per_wvl=6),
        axis=2,
        eme_grid_spec=td.EMEUniformGrid(
            num_cells=1,
            mode_spec=td.EMEModeSpec(num_modes=1),
        ),
        freqs=[FREQ0],
    )

    def objective(size_y):
        geometry = td.Box(center=(0, 0, 0), size=(0.5, WVL, LZ / 2)).updated_copy(
            size=(0.5, size_y, LZ / 2),
            validate=False,
        )
        structure = SIM_BASE.structures[0].updated_copy(
            geometry=geometry,
            validate=False,
        )
        traced_sim = SIM_BASE.updated_copy(structures=(structure,), validate=False)
        td.web.run_async({"eme": eme_sim, "traced": traced_sim}, verbose=False)
        return size_y

    with pytest.raises(
        AdjointError,
        match=r"traced FDTD Simulation tasks cannot be mixed with other simulation types",
    ):
        ag.grad(objective)(WVL)


def test_public_run_async_rejects_mixed_plain_then_unsupported_structure_before_submission(
    monkeypatch,
):
    """Mixed async batches should validate later unsupported traced structures."""

    def reject_submission(*args, **kwargs):
        raise AssertionError("unsupported traced structure should fail before submission")

    monkeypatch.setattr(autograd_module, "_run_async", reject_submission)
    monkeypatch.setattr(td.web.api.asynchronous, "run_async", reject_submission)

    def objective(permittivity):
        bad_structure = SIM_BASE.structures[0].updated_copy(
            medium=td.MultiPhysicsMedium(
                optical=td.Medium(permittivity=permittivity),
            ),
            validate=False,
        )
        bad_sim = SIM_BASE.updated_copy(structures=(bad_structure,), validate=False)
        td.web.run_async({"plain": SIM_BASE, "unsupported_structure": bad_sim}, verbose=False)
        return permittivity

    with pytest.raises(AdjointError, match=r"medium parameter 'optical\.permittivity'"):
        ag.grad(objective)(2.0)


def test_public_run_async_rejects_mixed_plain_then_unsupported_source_before_submission(
    monkeypatch,
):
    """Mixed async batches should validate later unsupported traced sources before submission."""

    def reject_submission(*args, **kwargs):
        raise AssertionError("unsupported traced source should fail before submission")

    monkeypatch.setattr(autograd_module, "_run_async", reject_submission)
    monkeypatch.setattr(td.web.api.asynchronous, "run_async", reject_submission)

    def objective(pol_angle):
        source = PLANE_WAVE.updated_copy(pol_angle=pol_angle, validate=False)
        bad_sim = SIM_BASE.updated_copy(sources=(source,), validate=False)
        td.web.run_async({"plain": SIM_BASE, "unsupported_source": bad_sim}, verbose=False)
        return pol_angle

    with pytest.raises(AdjointError, match=r"source parameter 'pol_angle'.*PlaneWave"):
        ag.grad(objective)(0.1)
