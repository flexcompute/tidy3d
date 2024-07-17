"""Test the parameter sweep plugin."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats.qmc as qmc
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import design as tdd

from ..utils import assert_log_level, run_emulated

SWEEP_METHODS = dict(
    grid=tdd.MethodGrid(),
    monte_carlo=tdd.MethodMonteCarlo(num_points=3, rng_seed=1),
    custom=tdd.MethodRandomCustom(
        num_points=5, sampler=qmc.Halton, sampler_kwargs={"d": 3, "seed": 1}
    ),
    random=tdd.MethodRandom(num_points=3, rng_seed=1),  # TODO: remove this if not used
    bay_opt=tdd.MethodBayOpt(initial_iter=2, n_iter=2, rng_seed=1),
    gen_alg=tdd.MethodGenAlg(solutions_per_pop=4, n_generations=2, n_parents_mating=2, rng_seed=1),
    part_swarm=tdd.MethodParticleSwarm(n_particles=3, n_iter=2, rng_seed=1),
)


def emulated_batch_run(simulations, path_dir: str = None, **kwargs):
    data_dict = {task_name: run_emulated(sim) for task_name, sim in simulations.simulations.items()}
    task_ids = dict(zip(simulations.simulations.keys(), data_dict.keys()))
    task_paths = dict(zip(simulations.simulations.keys(), simulations.simulations.keys()))

    class BatchDataEmulated(web.BatchData):
        """Emulated BatchData object that just returns stored emulated data."""

        data_dict: dict
        # task_ids: dict
        # task_paths: dict

        def items(self):
            yield from self.data_dict.items()

        def __getitem__(self, task_name):
            return self.data_dict[task_name]

    return BatchDataEmulated(data_dict=data_dict, task_ids=task_ids, task_paths=task_paths)


def emulated_estimate_cost(self, verbose=True):
    # Test both value and failed None returns
    value = np.random.random()

    if value < 0.5:
        return value
    else:
        return None


@pytest.mark.parametrize("sweep_method", SWEEP_METHODS.values())
def test_sweep(sweep_method, monkeypatch):
    # Problem, simulate scattering cross section of sphere ensemble
    # 	simulation consists of `num_spheres` spheres of radius `radius`.
    #   use defines `scs` function to set up and run simulation as function of inputs.
    #   then postprocesses the data to give the SCS.

    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    monkeypatch.setattr(web.Job, "estimate_cost", emulated_estimate_cost)

    # STEP1: define your design function (inputs and outputs)

    # Non td function testing
    def float_non_td_pre(radius, num_spheres, tag):
        return radius + num_spheres * 1.1

    def float_non_td_post(res):
        return int(res)

    def float_non_td_combined(radius, num_spheres, tag):
        return int(radius + num_spheres * 1.1)

    def list_non_td_pre(radius, num_spheres, tag):
        return [radius, num_spheres, radius + num_spheres * 1.1]

    def list_non_td_post(res):
        return int(sum(res))

    def dict_non_td_pre(radius, num_spheres, tag):
        return {"rad": radius, "num": num_spheres, "sum": radius + num_spheres * 1.1}

    def dict_non_td_post(res):
        return int(sum(res.values()))

    # Functions with td elements
    def scs_pre(radius: float, num_spheres: int, tag: str) -> td.Simulation:
        """Preprocessing function (make simulation)"""

        # set up simulation
        spheres = []
        freq0 = td.C_0 / 0.5

        for _ in range(int(num_spheres)):
            spheres.append(
                td.Structure(
                    geometry=td.Sphere(radius=radius),
                    medium=td.PEC,
                )
            )

        mnt = td.FieldMonitor(
            size=(0, 0, 0),
            center=(0, 0, 0),
            freqs=[2e14],
            name="field",
        )

        return td.Simulation(
            size=(1, 1, 1),
            structures=spheres,
            sources=[
                td.PointDipole(
                    center=(0, 0, 0),
                    source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                    polarization="Ex",
                )
            ],
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            monitors=[mnt],
        )

    def scs_post(sim_data: td.SimulationData) -> float:
        """Postprocessing function (analyze simulation data)"""

        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        return np.sum(np.square(np.abs(ex_values)))

    def scs_combined(radius: float, num_spheres: int, tag: str) -> float:
        """Preprocessing function (make simulation) and run it controlled by the user"""

        # set up simulation
        spheres = []

        for _ in range(int(num_spheres)):
            spheres.append(
                td.Structure(
                    geometry=td.Sphere(radius=radius),
                    medium=td.PEC,
                )
            )

        mnt = td.FieldMonitor(
            size=(0, 0, 0),
            center=(0, 0, 0),
            freqs=[2e14],
            name="field",
        )

        sim = td.Simulation(
            size=(1, 1, 1),
            structures=spheres,
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            monitors=[mnt],
        )

        sim_data = web.run(sim, task_name="test")

        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        return np.sum(np.square(np.abs(ex_values)))

    def scs_pre_batch(radius: float, num_spheres: int, tag: str) -> float:
        sim = {"batch_test": scs_pre(radius=radius, num_spheres=num_spheres, tag=tag)}

        return web.Batch(simulations=sim)

    def scs_post_batch(batch_data) -> float:
        """Postprocessing function (analyze simulation data)"""

        sim_data = [val[1] for val in batch_data.items()][0]

        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        return np.sum(np.square(np.abs(ex_values)))

    def scs_combined_batch(radius: float, num_spheres: int, tag: str) -> float:
        sim = {
            "batch_test1": scs_pre(radius=radius, num_spheres=num_spheres, tag=tag),
            "batch_test2": scs_pre(radius=radius, num_spheres=num_spheres, tag=tag),
        }

        batch_data = web.Batch(simulations=sim).run()

        sim_data = [val[1] for val in batch_data.items()][0]

        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        return np.sum(np.square(np.abs(ex_values)))

    def scs_pre_list(radius: float, num_spheres: int, tag: str):
        sim = scs_pre(radius, num_spheres, tag)
        return [sim, sim, sim]

    def scs_post_list(sim_list):
        sim_data = [scs_post(sim) for sim in sim_list]
        return sum(sim_data)

    def scs_pre_dict(radius: float, num_spheres: int, tag: str):
        sim = scs_pre(radius, num_spheres, tag)
        return {"test1": sim, "test2": sim, "3": sim}

    def scs_post_dict(sim_dict):
        sim_data = [scs_post(sim) for sim in sim_dict.values()]
        return sum(sim_data)

    def scs_pre_list_const(radius: float, num_spheres: int, tag: str):
        sim = scs_pre(radius, num_spheres, tag)
        return [sim, sim, tag]

    def scs_post_list_const(sim_list):
        sim_data = [scs_post(sim) for sim in sim_list if isinstance(sim, td.SimulationData)]
        consts = [const for const in sim_list if not isinstance(const, td.SimulationData)]
        assert isinstance(consts[0], str)  # Included for testing
        return sum(sim_data)

    def scs_pre_dict_const(radius: float, num_spheres: int, tag: str):
        sim = scs_pre(radius, num_spheres, tag)
        return {"test1": sim, "test2": sim, "tag_const": tag}

    def scs_post_dict_const(sim_dict):
        sim_data = [
            scs_post(sim) for sim in sim_dict.values() if isinstance(sim, td.SimulationData)
        ]
        consts = sim_dict["tag_const"]
        assert isinstance(consts, str)  # Included for testing
        return sum(sim_data)

    # STEP2: define your design problem

    radius_variable = tdd.ParameterFloat(
        name="radius",
        span=(0, 1.5),
        num_points=5,  # note: only used for MethodGrid
    )

    num_spheres_variable = tdd.ParameterInt(
        name="num_spheres",
        span=(0, 3),
    )

    tag_variable = tdd.ParameterAny(name="tag", allowed_values=("tag1", "tag2", "tag3"))

    design_space = tdd.DesignSpace(
        parameters=[radius_variable, num_spheres_variable, tag_variable],
        method=sweep_method,
        name="sphere CS",
        task_name=f"{sweep_method.type}",
    )

    # Check some summaries
    design_space.summary()
    design_space.summary(fn_pre=scs_pre)

    # STEP3: Run your design problem

    # Try a basic non-td function
    # Ensure output of combined and pre-post functions is the same
    non_td_sweep1 = design_space.run(float_non_td_combined)
    non_td_sweep2 = design_space.run(float_non_td_pre, float_non_td_post)

    assert non_td_sweep1.values == non_td_sweep2.values

    # Check that estimate fails for non-td functions
    with pytest.raises(ValueError):
        estimate = design_space.estimate_cost(float_non_td_pre)

    # Ensure output of list and dict pre funcs is the same
    list_non_td_sweep = design_space.run(list_non_td_pre, list_non_td_post)
    dict_non_td_sweep = design_space.run(dict_non_td_pre, dict_non_td_post)

    assert list_non_td_sweep.values == dict_non_td_sweep.values

    # Try functions that include td objects
    # Test that estimate_cost outputs and fails for combined function output
    estimate = design_space.estimate_cost(scs_pre)

    with pytest.raises(ValueError):
        estimate = design_space.estimate_cost(scs_combined)

    # Ensure output of combined and pre-post functions is the same
    td_sweep1 = design_space.run(scs_combined)
    td_sweep2 = design_space.run(scs_pre, scs_post)

    assert td_sweep1.values == td_sweep2.values

    # Try with batch output from pre
    estimate = design_space.estimate_cost(scs_pre_batch)
    td_batch = design_space.run(scs_pre_batch, scs_post_batch)
    td_batch_run_batch = design_space.run_batch(scs_pre_batch, scs_post_batch, path_dir="")

    # # Test user specified batching works with combined function
    td_batch_combined = design_space.run(scs_combined_batch)

    assert td_batch.values == td_batch_combined.values

    # Test with list of sims
    td_sim_list = design_space.run(scs_pre_list, scs_post_list)

    assert "0_0" not in td_sim_list.task_ids[0] and "0_3" not in td_sim_list.task_ids[4]

    # Test with dict of sims
    estimate = design_space.estimate_cost(scs_pre_dict)
    td_sim_dict = design_space.run(scs_pre_dict, scs_post_dict)

    # Check naming is including dict keys
    assert "test1" in td_sim_dict.task_ids[0]

    # Test with list of sims and non-sim constant values
    ts_sim_list_const = design_space.run(scs_pre_list_const, scs_post_list_const)

    # Test with dict of sims and non-sim constant values
    ts_sim_dict_const = design_space.run(scs_pre_dict_const, scs_post_dict_const)

    sel_kwargs_0 = dict(zip(td_sweep1.dims, td_sweep1.coords[0]))
    td_sweep1.sel(**sel_kwargs_0)

    print(td_sweep1.to_dataframe().head(10))

    td_sweep1.to_dataframe().plot.hexbin(x="num_spheres", y="radius", C="output")
    td_sweep1.to_dataframe().plot.scatter(x="num_spheres", y="radius", c="output")
    plt.close()

    design_space2 = tdd.DesignSpace(
        parameters=[radius_variable, num_spheres_variable, tag_variable],
        method=tdd.MethodMonteCarlo(num_points=3),
        name="sphere CS",
    )

    sweep_results_other = design_space2.run(scs_combined)

    # test combining results
    td_sweep1.combine(sweep_results_other)
    td_sweep1 + sweep_results_other

    # STEP4: modify the sweep results

    td_sweep2 = td_sweep2.add(fn_args={"radius": 1.2, "num_spheres": 5, "tag": "tag2"}, value=1.9)

    td_sweep2 = td_sweep2.delete(fn_args={"num_spheres": 5, "tag": "tag2", "radius": 1.2})

    sweep_results_df = td_sweep2.to_dataframe()

    sweep_results_2 = tdd.Result.from_dataframe(sweep_results_df)
    sweep_results_3 = tdd.Result.from_dataframe(sweep_results_df, dims=td_sweep2.dims)

    # make sure returning a float uses the proper output column header
    float_label = tdd.Result.default_value_keys(1.0)[0]
    assert float_label in sweep_results_df, "didn't assign column header properly for float"


# Split testing sweeps
def scs_pre(radius: float, num_spheres: int, tag: str) -> td.Simulation:
    """Preprocessing function (make simulation)"""

    # set up simulation
    spheres = []

    for _ in range(int(num_spheres)):
        spheres.append(
            td.Structure(
                geometry=td.Sphere(radius=radius),
                medium=td.PEC,
            )
        )

    mnt = td.FieldMonitor(
        size=(0, 0, 0),
        center=(0, 0, 0),
        freqs=[2e14],
        name="field",
    )

    return td.Simulation(
        size=(1, 1, 1),
        structures=spheres,
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        monitors=[mnt],
    )


def scs_post_complex_return(sim_data):
    """Uses scs_pre for pre function"""
    mnt_data = sim_data["field"]
    ex_values = mnt_data.Ex.values

    return {
        "test1": [True, False],
        "test2": None,
        "test3": "a great success",
        "test4": 3.14,
        "output_val": np.sum(np.square(np.abs(ex_values))),
        "np_arr": np.zeros(shape=(1, 2)),
    }


@pytest.mark.parametrize(
    "sweep_method",
    [
        SWEEP_METHODS["grid"],
        SWEEP_METHODS["monte_carlo"],
        SWEEP_METHODS["custom"],
        SWEEP_METHODS["random"],
    ],
)
def test_sample_specific(sweep_method, monkeypatch):
    """Run tests that are only relevant to MethodSample"""

    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    def float_non_td_pre(radius, num_spheres, tag):
        return radius + num_spheres * 1.1

    def float_non_td_complex_return_post(res):
        """Uses the same float_non_td_pre as pre"""
        return ["any", "other", "data", int(res), {"any1": 2.1}, np.ones(shape=(1, 2))]

    # Setup Designspace
    radius_variable = tdd.ParameterFloat(
        name="radius",
        span=(0, 1.5),
        num_points=5,  # note: only used for MethodGrid
    )

    num_spheres_variable = tdd.ParameterInt(
        name="num_spheres",
        span=(0, 3),
    )

    tag_variable = tdd.ParameterAny(name="tag", allowed_values=("tag1", "tag2", "tag3"))

    design_space = tdd.DesignSpace(
        parameters=[radius_variable, num_spheres_variable, tag_variable],
        method=sweep_method,
        name="sphere CS",
        task_name=f"{sweep_method.type}",
    )

    # Test for complex output shape in the output for non_td run
    complex_not_td_sweep = design_space.run(float_non_td_pre, float_non_td_complex_return_post)
    complex_not_td_df = complex_not_td_sweep.to_dataframe(include_aux=True)

    assert complex_not_td_df["output_2"][0] == "data"

    # Test for complex output shape in output of td function
    ts_sim_complex = design_space.run(scs_pre, scs_post_complex_return)
    ts_sim_complex_df = ts_sim_complex.to_dataframe(include_aux=True)

    assert ts_sim_complex_df["test4"][0] == 3.14


method_module_convert = {
    "MethodBayOpt": "bayes_opt",
    "MethodGenAlg": "pygad",
    "MethodParticleSwarm": "pyswarms.single.global_best",
}


@pytest.mark.parametrize(
    "sweep_method",
    [SWEEP_METHODS["bay_opt"], SWEEP_METHODS["gen_alg"], SWEEP_METHODS["part_swarm"]],
)
def test_optimize_specific(sweep_method, monkeypatch):
    """Run tests that are only relevant to MethodOptimize"""

    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    # Setup simulations
    def float_non_td_pre(radius, num_spheres, tag):
        return radius + num_spheres * 1.1

    def float_non_td_aux_post(res):
        """Uses the same float_non_td_pre as pre"""
        return [int(res), ["any", "other", "data"]]

    def scs_pre(radius: float, num_spheres: int, tag: str) -> td.Simulation:
        """Preprocessing function (make simulation)"""

        # set up simulation
        spheres = []

        for _ in range(int(num_spheres)):
            spheres.append(
                td.Structure(
                    geometry=td.Sphere(radius=radius),
                    medium=td.PEC,
                )
            )

        mnt = td.FieldMonitor(
            size=(0, 0, 0),
            center=(0, 0, 0),
            freqs=[2e14],
            name="field",
        )

        return td.Simulation(
            size=(1, 1, 1),
            structures=spheres,
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            monitors=[mnt],
        )

    def scs_post_aux(sim_data):
        """Uses scs_pre for pre function"""
        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        return (
            np.sum(np.square(np.abs(ex_values))),
            {"test1": True, "test2": None, "test3": "a great success", "test4": 3.14},
        )

    # Setup Designspace
    radius_variable = tdd.ParameterFloat(
        name="radius",
        span=(0, 1.5),
        num_points=5,  # note: only used for MethodGrid
    )

    num_spheres_variable = tdd.ParameterInt(
        name="num_spheres",
        span=(0, 3),
    )

    tag_variable = tdd.ParameterAny(name="tag", allowed_values=("tag1", "tag2", "tag3"))

    design_space = tdd.DesignSpace(
        parameters=[radius_variable, num_spheres_variable, tag_variable],
        method=sweep_method,
        name="sphere CS",
        task_name=f"{sweep_method.type}",
    )

    # Include auxiliary data in list format in the output for non_td run
    aux_not_td_sweep = design_space.run(float_non_td_pre, float_non_td_aux_post)
    aux_not_td_df = aux_not_td_sweep.to_dataframe(include_aux=True)

    assert aux_not_td_df["aux_key_2"][0] == "data"

    # Test for auxiliary values as dict format in output of td function
    ts_sim_aux = design_space.run(scs_pre, scs_post_aux)
    ts_sim_aux_df = ts_sim_aux.to_dataframe(include_aux=True)

    assert ts_sim_aux_df["test4"][0] == 3.14

    # Test that sampler style complex input fails properly
    with pytest.raises(ValueError):
        ts_sim_complex = design_space.run(scs_pre, scs_post_complex_return)

    # Test that wrongly formatted aux outputs throw an error
    def bad_format_non_td_aux_post(res):
        """Uses the same float_non_td_pre as pre"""
        return ["not a float", int(res), ["any", "other", "data"]]

    with pytest.raises(ValueError):
        bad_format_td_aux_sweep = design_space.run(float_non_td_pre, bad_format_non_td_aux_post)

    # Test that plots have been produced and stored
    assert ts_sim_aux.opt_output is not None

    # Create an import error to test optimizer can error if relevant package is not installed
    # Placed at the end of the test so that module is loaded for other checks

    with pytest.raises(ImportError):
        sys.modules[method_module_convert[sweep_method.type]] = None
        import_fail = design_space.run(float_non_td_pre, float_non_td_aux_post)


def test_method_custom_validators():
    """Test the MethodRandomCustom validation performs as expected."""

    radius_variable = tdd.ParameterFloat(
        name="radius",
        span=(0, 1.5),
        num_points=5,  # note: only used for MethodGrid
    )

    num_spheres_variable = tdd.ParameterInt(
        name="num_spheres",
        span=(0, 3),
    )

    tag_variable = tdd.ParameterAny(name="tag", allowed_values=("tag1", "tag2", "tag3"))

    parameters = (radius_variable, num_spheres_variable, tag_variable)

    d = 3

    # expected case
    class SamplerWorks:
        def random(self, n):
            return np.random.random((n, d))

    working_sampler = tdd.MethodRandomCustom(num_points=5, sampler=SamplerWorks)
    working_sampler.get_sampler(parameters)

    # missing random method case
    class SamplerNoRandom:
        pass

    no_random = tdd.MethodRandomCustom(num_points=5, sampler=SamplerNoRandom)
    with pytest.raises(ValueError):
        no_random.get_sampler(parameters)

    # random method gives a list
    class SamplerList:
        def random(self, n):
            return np.random.random((n, d)).tolist()

    gives_list = tdd.MethodRandomCustom(num_points=5, sampler=SamplerList)
    with pytest.raises(ValueError):
        gives_list.get_sampler(parameters)

    # random method gives wrong number of dimensions
    class SamplerWrongDims:
        def random(self, n):
            return np.random.random((n, d, d))

    wrong_dim = tdd.MethodRandomCustom(num_points=5, sampler=SamplerWrongDims)
    with pytest.raises(ValueError):
        wrong_dim.get_sampler(parameters)

    # random method gives wrong first dimension length
    class SamplerWrongShape:
        def random(self, n):
            return np.random.random((n + 1, d))

    wrong_shape = tdd.MethodRandomCustom(num_points=5, sampler=SamplerWrongShape)
    with pytest.raises(ValueError):
        wrong_shape.get_sampler(parameters)

    # random method gives floats outside of range of 0, 1
    class SamplerOutOfRange:
        def random(self, n):
            return 3 * np.random.random((n, d)) - 1

    out_range = tdd.MethodRandomCustom(num_points=5, sampler=SamplerOutOfRange)
    with pytest.raises(ValueError):
        out_range.get_sampler(parameters)

    # wrong number of dims given to sampler
    d = 2

    failing_sampler = tdd.MethodRandomCustom(num_points=5, sampler=SamplerWorks)
    with pytest.raises(ValueError):
        failing_sampler.get_sampler(parameters)


@pytest.mark.parametrize(
    "monte_carlo_warning, log_level_expected",
    [(True, "WARNING"), (False, None)],
    ids=["warn_monte_carlo", "no_warn_monte_carlo"],
)
def test_method_random_warning(log_capture, monte_carlo_warning, log_level_expected):
    """Test that method random validation / warning works as expected."""

    tdd.MethodRandom(num_points=10, monte_carlo_warning=monte_carlo_warning)
    assert_log_level(log_capture, log_level_expected)


@pytest.mark.parametrize(
    "parameter",
    [
        tdd.ParameterAny(name="test", allowed_values=("a", "b", "c")),
        tdd.ParameterFloat(name="test", span=(0, 1)),
        tdd.ParameterInt(name="test", span=(1, 5)),
    ],
    ids=["any", "float", "int"],
)
def test_random_sampling(parameter):
    """just make sure sample_random still works in case we need it."""
    parameter.sample_random(10)


# Test bad parameter setup
def test_bad_parameters():
    # span min bigger than max
    with pytest.raises(
        ValueError
    ):  # Is raised as a pd.ValidationError but pytest catches it as a ValueError
        bad_span = tdd.ParameterInt(name="bad_span", span=(3, 1))

    # not unique values in Parameter
    with pytest.raises(ValueError):
        not_unique1 = tdd.ParameterFloat(name="not_unique1", values=(1.2, 2.2, 1.2))

    # not unique allowed_values in ParameterAny
    with pytest.raises(ValueError):
        not_unique2 = tdd.ParameterAny(name="not_unique2", allowed_values=("a", "b", "a"))

    # no values
    with pytest.raises(ValueError):
        no_values = tdd.ParameterAny(name="no_values", allowed_values=())


# Test bad result setup
def test_bad_result():
    # Too many coords, not enough dims
    with pytest.raises(ValueError):
        bad_dims = tdd.result.Result(
            dims=("test_param",),
            values=[1, 2],
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_ids=None,
            aux_values=None,
        )

    # Too many coords, not enough values
    with pytest.raises(ValueError):
        bad_values = tdd.result.Result(
            dims=("test_param", "test_param2"),
            values=(1,),
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_ids=None,
            aux_values=None,
        )

    # Combine tests
    # Different sources
    with pytest.raises(ValueError):
        source_1 = tdd.result.Result(
            dims=("test_param", "test_param2"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="Uh",
            task_ids=None,
            aux_values=None,
        )

        source_2 = tdd.result.Result(
            dims=("test_param", "test_param2"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="Oh",
            task_ids=None,
            aux_values=None,
        )

        source_1.combine(source_2)

    # Different output names
    with pytest.raises(ValueError):
        dim_1 = tdd.result.Result(
            dims=("test_paramX", "test_paramY"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_ids=None,
            aux_values=None,
        )

        dim_2 = tdd.result.Result(
            dims=("test_param1", "test_param2"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_ids=None,
            aux_values=None,
        )

        dim_1.combine(dim_2)
