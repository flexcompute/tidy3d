"""Test the parameter sweep plugin."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins import design as tdd

from ..utils import run_emulated

SWEEP_METHODS = dict(
    grid=tdd.MethodGrid(),
    monte_carlo=tdd.MethodMonteCarlo(num_points=5, seed=1),
    bay_opt=tdd.MethodBayOpt(initial_iter=5, n_iter=2, seed=1),
    gen_alg=tdd.MethodGenAlg(
        solutions_per_pop=6,
        n_generations=2,
        n_parents_mating=4,
        seed=1,
        mutation_prob=0,
        keep_parents=0,
    ),
    part_swarm=tdd.MethodParticleSwarm(n_particles=3, n_iter=2, seed=1),
)

# Task names that should be produced for the different methods
expected_task_names = {
    "MethodGrid": ["MethodGrid_10_10", "MethodGrid_41_41"],
    "MethodMonteCarlo": ["MethodMonteCarlo_0_0", "MethodMonteCarlo_3_3"],
    "MethodBayOpt": ["MethodBayOpt_2_2", "MethodBayOpt_0_5"],
    "MethodGenAlg": ["MethodGenAlg_3_3", "MethodGenAlg_0_8"],
    "MethodParticleSwarm": ["MethodParticleSwarm_2_2", "MethodParticleSwarm_1_4"],
}


def emulated_batch_run(simulations, path_dir: str = None, **kwargs):
    data_dict = {task_name: run_emulated(sim) for task_name, sim in simulations.simulations.items()}
    task_ids = dict(zip(simulations.simulations.keys(), data_dict.keys()))
    task_paths = {key: f"/path/to/{key}" for key in simulations.simulations.keys()}

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


# Pre and post functions
# Non td function testing
def float_non_td_pre(radius, num_spheres, tag):
    return radius + num_spheres * 1.1


def float_non_td_post(res):
    return int(res)


def float_non_td_tuple_pre(radius, num_spheres, tag):
    return radius + num_spheres * 1.1, tag


def float_non_td_aux_post(res):
    """Uses the same float_non_td_pre as pre"""
    return [int(res), ["any", "other", "data"]]


def float_non_td_complex_return_post(res):
    """Uses the same float_non_td_pre as pre"""
    return ["any", "other", "data", int(res), {"any1": 2.1}, np.ones(shape=(1, 2))]


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


def non_td_pre_multi_return(radius: float, num_spheres: int, tag: str):
    if tag == "tag1":
        return {"rad": radius, "num": num_spheres, "sum": radius + num_spheres * 1.1}
    else:
        return [radius + num_spheres - 1, tag]


def non_td_post_multi_return(res):
    if isinstance(res, dict):
        return [res["sum"], (res["rad"], res["num"])]
    else:
        return res[0]


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

    sim = td.Simulation(
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
    batch = web.Batch(simulations={"a": sim, "b": sim})
    return {"test1": sim, "test2": sim, "batch1": batch}


def scs_post_dict(sim_dict):
    sim_data = [sim_dict["test1"], sim_dict["test2"]]
    batched_data = [sim for _, sim in sim_dict["batch1"].items()]
    sim_data.extend(batched_data)
    post_sim_data = [scs_post(sim) for sim in sim_data]
    return sum(post_sim_data)


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
    sim_data = [scs_post(sim) for sim in sim_dict.values() if isinstance(sim, td.SimulationData)]
    consts = sim_dict["tag_const"]
    assert isinstance(consts, str)  # Included for testing
    return sum(sim_data)


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


def scs_post_aux(sim_data):
    """Uses scs_pre for pre function"""
    mnt_data = sim_data["field"]
    ex_values = mnt_data.Ex.values

    return (
        np.sum(np.square(np.abs(ex_values))),
        {"test1": True, "test2": None, "test3": "a great success", "test4": 3.14},
    )


def init_design_space(sweep_method):
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

    return design_space


@pytest.mark.parametrize("sweep_method", SWEEP_METHODS.values())
def test_sweep(sweep_method, monkeypatch):
    # Problem, simulate scattering cross section of sphere ensemble
    # 	simulation consists of `num_spheres` spheres of radius `radius`.
    #   use defines `scs` function to set up and run simulation as function of inputs.
    #   then postprocesses the data to give the SCS.

    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    # Initialize designspace and parameters
    design_space = init_design_space(sweep_method=sweep_method)

    # Check some summaries
    design_space.summarize()

    # Try a basic non-td function
    # Ensure output of combined and pre-post functions is the same
    non_td_sweep1 = design_space.run(float_non_td_combined)
    non_td_sweep2 = design_space.run(float_non_td_pre, float_non_td_post)

    assert non_td_sweep1.values == non_td_sweep2.values

    # Test that tuple output from pre is not accepted
    with pytest.raises(ValueError):
        non_td_tuple_failure = design_space.run(float_non_td_tuple_pre, float_non_td_post)

    # Ensure output of list and dict pre funcs is the same
    list_non_td_sweep = design_space.run(list_non_td_pre, list_non_td_post)
    dict_non_td_sweep = design_space.run(dict_non_td_pre, dict_non_td_post, verbose=False)

    assert list_non_td_sweep.values == dict_non_td_sweep.values

    # Test multiple return statements for pre and post
    # Currently not supporting this and intercepting attempts within _extract_output
    # Note this requires a "tag1" tag to be in the prediction space
    with pytest.raises(ValueError):
        non_td_multi_return = design_space.run(non_td_pre_multi_return, non_td_post_multi_return)

    # Try functions that include td objects
    # Ensure output of combined and pre-post functions is the same
    td_sweep1 = design_space.run(scs_combined, verbose=False)
    td_sweep2 = design_space.run(scs_pre, scs_post)

    assert np.allclose(td_sweep1.values, td_sweep2.values)

    # Check names are what they should be
    assert (
        expected_task_names[sweep_method.type][0] in td_sweep2.task_names
        and expected_task_names[sweep_method.type][0] in td_sweep2.task_names
    )

    # Try with batch output from pre
    td_batch = design_space.run(scs_pre_batch, scs_post_batch)
    td_batch_run_batch = design_space.run_batch(
        scs_pre_batch, scs_post_batch, path_dir="", batch_kwargs={"fake_kwarg": None}
    )

    # Test user specified batching works with combined function
    td_batch_combined = design_space.run(scs_combined_batch)

    assert np.allclose(td_batch.values, td_batch_combined.values)

    # Test with list of sims
    td_sim_list = design_space.run(scs_pre_list, scs_post_list, verbose=False)

    assert "0_0" not in td_sim_list.task_names[0] and "0_3" not in td_sim_list.task_names[4]

    # Collect the sim names and check there are no repeats
    total_sim_names = [sim_name for sim_list in td_sim_list.task_names for sim_name in sim_list]
    unique_sim_names = set(total_sim_names)

    assert len(total_sim_names) == len(unique_sim_names)

    # Test with dict of sims
    td_sim_dict = design_space.run(scs_pre_dict, scs_post_dict)

    # Check naming is including dict keys
    assert "test1" in td_sim_dict.task_names[0]

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

    design_space2 = init_design_space(sweep_method=sweep_method)

    sweep_results_other = design_space2.run(scs_combined)

    # test combining results
    td_sweep1.combine(sweep_results_other)
    td_sweep1 + sweep_results_other

    # Modify the sweep results

    td_sweep2 = td_sweep2.add(fn_args={"radius": 1.2, "num_spheres": 5, "tag": "tag2"}, value=1.9)

    td_sweep2 = td_sweep2.delete(fn_args={"num_spheres": 5, "tag": "tag2", "radius": 1.2})

    sweep_results_df = td_sweep2.to_dataframe()

    sweep_results_2 = tdd.Result.from_dataframe(sweep_results_df)
    sweep_results_3 = tdd.Result.from_dataframe(sweep_results_df, dims=td_sweep2.dims)

    # make sure returning a float uses the proper output column header
    float_label = tdd.Result.default_value_keys(1.0)[0]
    assert float_label in sweep_results_df, "didn't assign column header properly for float"


def emulated_estimate_cost_return(self, verbose=True):
    return 0.5


def emulated_estimate_cost_none(self, verbose=True):
    return None


@pytest.mark.parametrize(
    "est_cost_func",
    (emulated_estimate_cost_return, emulated_estimate_cost_none),
)
@pytest.mark.parametrize("sweep_method", SWEEP_METHODS.values())
def test_estimate_cost(est_cost_func, sweep_method, monkeypatch):
    # Mock job.delete to remove any calls to server during the testing
    def emulated_job_delete(self):
        pass

    # Still need to emulate the run as tests may call tidy3d
    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    monkeypatch.setattr(web.Job, "delete", emulated_job_delete)

    monkeypatch.setattr(web.Job, "estimate_cost", est_cost_func)

    # Initialize designspace and parameters
    design_space = init_design_space(sweep_method=sweep_method)

    # Summary with pre calls estimate_cost so included here
    design_space.summarize(fn_pre=scs_pre)

    # Check that estimate fails for non-td functions
    with pytest.raises(ValueError):
        estimate = design_space.estimate_cost(float_non_td_pre)

    # Test that estimate_cost outputs and fails for combined function output
    estimate = design_space.estimate_cost(scs_pre)

    with pytest.raises(ValueError):
        estimate = design_space.estimate_cost(scs_combined)

    # Test with batched pre function
    estimate = design_space.estimate_cost(scs_pre_batch)

    # Test with dict pre function
    estimate = design_space.estimate_cost(scs_pre_dict)

    if est_cost_func == emulated_estimate_cost_return:
        assert isinstance(estimate, float)
    else:
        assert estimate is None


@pytest.mark.parametrize(
    "sweep_method",
    [
        SWEEP_METHODS["grid"],
        SWEEP_METHODS["monte_carlo"],
    ],
)
def test_sample_specific(sweep_method, monkeypatch):
    """Run tests that are only relevant to MethodSample"""

    monkeypatch.setattr(web, "run", run_emulated)

    monkeypatch.setattr(web.Batch, "run", emulated_batch_run)

    # Setup Designspace
    design_space = init_design_space(sweep_method=sweep_method)

    # Test for complex output shape in the output for non_td run
    complex_not_td_sweep = design_space.run(float_non_td_pre, float_non_td_complex_return_post)
    complex_not_td_df = complex_not_td_sweep.to_dataframe()

    assert complex_not_td_df["output_2"][0] == "data"

    # Test for complex output shape in output of td function
    ts_sim_complex = design_space.run(scs_pre, scs_post_complex_return)
    ts_sim_complex_df = ts_sim_complex.to_dataframe()

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

    # Setup Designspace
    design_space = init_design_space(sweep_method=sweep_method)

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
    assert ts_sim_aux.optimizer is not None

    # Create an import error to test optimizer can error if relevant package is not installed
    # Placed at the end of the test so that module is loaded for other checks

    with pytest.raises(ImportError):
        monkeypatch.setitem(sys.modules, method_module_convert[sweep_method.type], None)
        import_fail = design_space.run(float_non_td_pre, float_non_td_aux_post)


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
            task_names=None,
            aux_values=None,
        )

    # Too many coords, not enough values
    with pytest.raises(ValueError):
        bad_values = tdd.result.Result(
            dims=("test_param", "test_param2"),
            values=(1,),
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_names=None,
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
            task_names=None,
            aux_values=None,
        )

        source_2 = tdd.result.Result(
            dims=("test_param", "test_param2"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="Oh",
            task_names=None,
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
            task_names=None,
            aux_values=None,
        )

        dim_2 = tdd.result.Result(
            dims=("test_param1", "test_param2"),
            values=(1, 2),
            coords=[(1, 2), (3, 4)],
            fn_source="",
            task_names=None,
            aux_values=None,
        )

        dim_1.combine(dim_2)


def test_genalg_early_stop():
    gen_alg_pass = tdd.MethodGenAlg(
        solutions_per_pop=4,
        n_generations=2,
        n_parents_mating=2,
        stop_criteria_type="reach",
        stop_criteria_number=1,
        seed=1,
    )
    design_space_pass = init_design_space(gen_alg_pass)

    non_td_sweep_stop = design_space_pass.run(float_non_td_pre, float_non_td_post)

    # Check that the GA has early stopped
    # 8 as it will include the initial_population and Generation 1
    assert len(non_td_sweep_stop.coords) == 8

    gen_alg_fail = tdd.MethodGenAlg(
        solutions_per_pop=4,
        n_generations=2,
        n_parents_mating=2,
        stop_criteria_type=None,
        stop_criteria_number=1,
        seed=1,
    )

    design_space_fail = init_design_space(gen_alg_fail)

    # Check that stop criteria can't be made successfully
    with pytest.raises(ValueError):
        non_td_sweep_stop = design_space_fail.run(float_non_td_pre, float_non_td_post)


def test_genalg_run_count():
    """Huge GA with fast fitness func to test the GA doesn't fail after many iterations."""

    gen_alg = tdd.MethodGenAlg(
        solutions_per_pop=100,
        n_generations=100,
        n_parents_mating=6,
        parent_selection_type="rws",
        keep_parents=2,
        mutation_prob=0.4,
        mutation_type="scramble",
        crossover_prob=0.6,
        crossover_type="uniform",
        seed=1,
        save_solution=True,
    )
    design_space = init_design_space(gen_alg)

    # Using list as counter as the pre function burried in the GA can append to a list
    # However it cannot update an int counter variable
    count = []

    def count_float_non_td_pre(radius, num_spheres, tag):
        print(f"Pre ran: {len(count)}")  # To see when pre ran
        count.append(1)
        return len(count)

    def count_float_non_td_post(res):
        return [res, [res]]

    non_td_sweep_count = design_space.run(count_float_non_td_pre, count_float_non_td_post)
    df = non_td_sweep_count.to_dataframe(include_aux=True)

    coord_strings = [str(coord) for coord in non_td_sweep_count.coords]
    coord_set = set(coord_strings)

    # Checks that every time the pre_func is run the GA is storing the result
    assert len(count) == len(coord_set)

    # Check aux is the same length as coords
    assert len(non_td_sweep_count.coords) == len(non_td_sweep_count.aux_values)

    # Check that the result and the aux are the same value
    assert all(df["output"] == df["aux_key_0"])


@pytest.mark.parametrize(
    "dims, coords, values, multi_val",
    [
        (("a", "b", "c"), ((1, 0, 1), (2, 3, 4), (5, 1, 2)), (1, 2, 3), False),
        (("a", "b", "c"), ((5, 2, 0), (1, 2, 3), (3, 4, 1)), ((3, 4), (4, 6), (6, 9)), True),
        (("x", "y", "z", "t"), ((1, 2, 3, 4), (5, 6, 7, 8)), (5, 3), False),
        (tuple("a"), (tuple("b"), tuple("c")), ("d", "e"), False),
    ],
)
def test_result_accessor_and_len(dims, coords, values, multi_val):
    """Test the accessor for Result lines up with regular indexing into the coordinates and values
    and that the length matches that of the coordinates and values. Finally, since we have defined
    accessors and length, ensure we can cast the Result class to a list and tuple"""

    result = tdd.Result(dims=dims, values=values, coords=coords)

    for idx in range(0, len(values)):
        coord, value = result[idx]

        assert all(coord == coords[idx])

        if multi_val:
            assert all(value == values[idx])
        else:
            assert value == values[idx]

    assert len(result) == len(coords)
    assert len(result) == len(values)

    result_list = list(result)
    result_tuple = tuple(result)


def test_dim_coord_alignment():
    """Test that the dims coming out of the optimization method align with the ordering of the coordinates.
    The ordering of these dimensions from the optimization can differ from the initial ordering of dimensions
    input into the DesignSpace."""
    bay_opt = tdd.MethodBayOpt(initial_iter=5, n_iter=2, seed=1)

    parameters_sweep = [
        [
            tdd.ParameterFloat(name="radius", span=(1, 5)),
            tdd.ParameterFloat(name="alpha", span=(-5, -1)),
        ],
        [
            tdd.ParameterFloat(name="alpha", span=(-5, -1)),
            tdd.ParameterFloat(name="radius", span=(1, 5)),
        ],
    ]

    for parameters in parameters_sweep:
        design_space = tdd.DesignSpace(
            parameters=parameters, method=bay_opt, name="bay opt", task_name="bayesian"
        )

        def opt_fn(radius, alpha):
            return radius**2 - alpha * radius + alpha**2 - 2 * radius + 2 * alpha

        result = design_space.run(opt_fn)

        # get the location of these parameters according to the result
        radius_parameter_location = result.dims.index("radius")
        alpha_parameter_location = result.dims.index("alpha")

        # make sure each of the coordinates (as given by their dimension) make sense with their allowed spans
        assert all(coord[radius_parameter_location] > 0 for coord in result.coords)
        assert all(coord[alpha_parameter_location] < 0 for coord in result.coords)
