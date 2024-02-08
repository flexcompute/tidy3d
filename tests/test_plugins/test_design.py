"""Test the parameter sweep plugin."""
import pytest
import numpy as np
import tidy3d as td
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc

import tidy3d.web as web
from tidy3d.components.base import Tidy3dBaseModel

from tidy3d.plugins import design as tdd
from tidy3d.plugins.design.method import MethodIndependent

from ..utils import run_emulated, log_capture, assert_log_level


SWEEP_METHODS = dict(
    grid=tdd.MethodGrid(),
    monte_carlo=tdd.MethodMonteCarlo(num_points=5),
    custom=tdd.MethodRandomCustom(num_points=5, sampler=qmc.Halton(d=3)),
    random=tdd.MethodRandom(num_points=5),  # TODO: remove this if not used
)


@pytest.mark.parametrize("sweep_method", SWEEP_METHODS.values(), ids=SWEEP_METHODS.keys())
def test_sweep(sweep_method, monkeypatch, ids=[]):
    # Problem, simulate scattering cross section of sphere ensemble
    # 	simulation consists of `num_spheres` spheres of radius `radius`.
    #   use defines `scs` function to set up and run simulation as function of inputs.
    #   then postprocesses the data to give the SCS.

    monkeypatch.setattr(web, "run", run_emulated)

    def emulated_batch_run(self, simulations, path_dir: str = None, **kwargs):
        data_dict = {task_name: run_emulated(sim) for task_name, sim in simulations.items()}
        task_ids = dict(zip(simulations.keys(), data_dict.keys()))

        class BatchDataEmulated(Tidy3dBaseModel):
            """Emulated BatchData object that just returns stored emulated data."""

            data_dict: dict
            task_ids: dict

            def items(self):
                for task_name, sim_data in self.data_dict.items():
                    yield task_name, sim_data

            def __getitem__(self, task_name):
                return self.data_dict[task_name]

        return BatchDataEmulated(data_dict=data_dict, task_ids=task_ids)

    monkeypatch.setattr(MethodIndependent, "_run_batch", emulated_batch_run)

    # STEP1: define your design function (inputs and outputs)

    def scs_pre(radius: float, num_spheres: int, tag: str) -> td.Simulation:
        """Preprocessing function (make simulation)"""

        # set up simulation
        spheres = []

        for i in range(int(num_spheres)):
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

    def scs_post(sim_data: td.SimulationData) -> float:
        """Postprocessing function (analyze simulation data)"""

        mnt_data = sim_data["field"]
        ex_values = mnt_data.Ex.values

        # generate a random number to add some variance to data
        np.random.seed(hash(sim_data) % 1000)

        return np.sum(np.square(np.abs(ex_values))) + np.random.random()

    def scs_pre_multi(*args, **kwargs):
        sim = scs_pre(*args, **kwargs)

        return [sim, sim, sim]

    def scs_post_multi(*sim_datas):
        vals = [scs_post(sim_data) for sim_data in sim_datas]
        return np.mean(vals)

    def scs_pre_dict(*args, **kwargs):
        sims = scs_pre_multi(*args, **kwargs)
        keys = "abc"
        return dict(zip(keys, sims))

    def scs_post_dict(a=None, b=None, c=None):
        sims = [a, b, c]
        return scs_post_multi(*sims)

    def scs(radius: float, num_spheres: int, tag: str) -> float:
        """End to end function."""

        sim = scs_pre(radius=radius, num_spheres=num_spheres, tag=tag)

        # run simulation
        sim_data = run_emulated(sim, task_name=f"SWEEP_{tag}")

        # postprocess
        return scs_post(sim_data=sim_data)

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
    )

    # STEP3: Run your design problem

    # either supply generic function and run one by one
    sweep_results = design_space.run(scs)

    # or supply function factored into pre and post and run in batch
    sweep_results2 = design_space.run_batch(scs_pre, scs_post)

    sweep_results3 = design_space.run_batch(scs_pre_multi, scs_post_multi)

    sweep_results4 = design_space.run_batch(scs_pre_dict, scs_post_dict)

    # multiprocessing (built in)
    sweep_results5 = design_space.run_multiprocess(scs, num_processes=2)

    # multiprocessing (user supplied map function)
    from multiprocess import Pool

    with Pool() as p:
        sweep_results6 = design_space.run(scs, map_fn=p.map)

    sel_kwargs_0 = dict(zip(sweep_results.dims, sweep_results.coords[0]))
    sweep_results.sel(**sel_kwargs_0)

    print(sweep_results.to_dataframe().head(10))

    im = sweep_results.to_dataframe().plot.hexbin(x="num_spheres", y="radius", C="output")
    im = sweep_results.to_dataframe().plot.scatter(x="num_spheres", y="radius", c="output")
    plt.close()

    design_space2 = tdd.DesignSpace(
        parameters=[radius_variable, num_spheres_variable, tag_variable],
        method=tdd.MethodMonteCarlo(num_points=3),
        name="sphere CS",
    )

    sweep_results_other = design_space2.run(scs)

    results_combined = sweep_results.combine(sweep_results_other)
    results_combined = sweep_results + sweep_results_other

    # STEP4: modify the sweep results

    sweep_results = sweep_results.add(
        fn_args={"radius": 1.2, "num_spheres": 5, "tag": "tag2"}, value=1.9
    )

    sweep_results = sweep_results.delete(fn_args={"num_spheres": 5, "tag": "tag2", "radius": 1.2})

    sweep_results_df = sweep_results.to_dataframe()

    sweep_results_2 = tdd.Result.from_dataframe(sweep_results_df)
    sweep_results_3 = tdd.Result.from_dataframe(sweep_results_df, dims=sweep_results.dims)

    assert sweep_results == sweep_results_2 == sweep_results_3

    # VALIDATE PROPER DATAFRAME HEADERS AND DATA STORAGE

    # make sure returning a float uses the proper output column header
    float_label = tdd.Result.default_value_keys(1.0)[0]
    assert float_label in sweep_results_df, "didn't assign column header properly for float"

    # make sure returning a dict uses the keys as output column headers

    labels = ["label1", "label2"]

    def scs_dict(*args, **kwargs):
        output = scs(*args, **kwargs)
        return dict(zip(labels, len(labels) * [output]))

    df = design_space.run(scs_dict).to_dataframe()
    for label in labels:
        assert label in df, "dict key not parsed properly as column header"
        for value in df[label]:
            assert not isinstance(value, dict), "dict saved instead of value"

    # make sure returning a list assigns column labels properly

    num_outputs = 3
    label_keys = tdd.Result.default_value_keys(num_outputs * [0.0])

    def scs_list(*args, **kwargs):
        output = scs(*args, **kwargs)
        return num_outputs * [output]

    df = design_space.run(scs_list).to_dataframe()
    for label in label_keys:
        assert label in df, "dict key not parsed properly as column header"
        for value in df[label]:
            assert not isinstance(value, (tuple, list)), "dict saved instead of value"


def test_method_custom_validators():
    """Test the MethodRandomCustom validation performs as expected."""

    d = 3

    # expected case
    class SamplerWorks:
        def random(self, n):
            return np.random.random((n, d))

    tdd.MethodRandomCustom(num_points=5, sampler=SamplerWorks()),

    # missing random method case
    class SamplerNoRandom:
        pass

    with pytest.raises(ValueError):
        tdd.MethodRandomCustom(num_points=5, sampler=SamplerNoRandom()),

    # random method gives a list
    class SamplerList:
        def random(self, n):
            return np.random.random((n, d)).tolist()

    with pytest.raises(ValueError):
        tdd.MethodRandomCustom(num_points=5, sampler=SamplerList()),

    # random method gives wrong number of dimensions
    class SamplerWrongDims:
        def random(self, n):
            return np.random.random((n, d, d))

    with pytest.raises(ValueError):
        tdd.MethodRandomCustom(num_points=5, sampler=SamplerWrongDims()),

    # random method gives wrong first dimension length
    class SamplerWrongShape:
        def random(self, n):
            return np.random.random((n + 1, d))

    with pytest.raises(ValueError):
        tdd.MethodRandomCustom(num_points=5, sampler=SamplerWrongShape()),

    # random method gives floats outside of range of 0, 1
    class SamplerOutOfRange:
        def random(self, n):
            return 3 * np.random.random((n, d)) - 1

    with pytest.raises(ValueError):
        tdd.MethodRandomCustom(num_points=5, sampler=SamplerOutOfRange()),


@pytest.mark.parametrize(
    "monte_carlo_warning, log_level_expected",
    [(True, "WARNING"), (False, None)],
    ids=["warn_monte_carlo", "no_warn_monte_carlo"],
)
def test_method_random_warning(log_capture, monte_carlo_warning, log_level_expected):
    """Test that method random validation / warning works as expected."""

    method = tdd.MethodRandom(num_points=10, monte_carlo_warning=monte_carlo_warning)
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
