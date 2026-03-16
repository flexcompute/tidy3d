"""Tests for standalone Adam optimizer and optimize() function."""

from __future__ import annotations

import numpy as np
import pytest

from tidy3d.plugins.autograd.optimizers import Adam, adam, apply_updates, optimize


@pytest.fixture
def opt():
    return adam(learning_rate=0.01)


@pytest.fixture
def params():
    return np.array([1.0, 2.0, 3.0])


class TestAdamInit:
    def test_shapes(self, opt, params):
        state = opt.init(params)
        assert state["m"].shape == params.shape
        assert state["v"].shape == params.shape

    def test_zeros(self, opt, params):
        state = opt.init(params)
        np.testing.assert_array_equal(state["m"], 0.0)
        np.testing.assert_array_equal(state["v"], 0.0)
        assert state["t"] == 0

    def test_2d_params(self, opt):
        params_2d = np.zeros((3, 4))
        state = opt.init(params_2d)
        assert state["m"].shape == (3, 4)

    def test_m_v_not_aliased(self, opt, params):
        state = opt.init(params)
        assert state["m"] is not state["v"]


class TestAdamFactory:
    def test_returns_adam_instance(self):
        opt = adam(learning_rate=0.1)
        assert isinstance(opt, Adam)

    def test_hyperparams(self):
        opt = adam(learning_rate=0.05, beta1=0.8, beta2=0.99, eps=1e-6)
        assert opt.learning_rate == 0.05
        assert opt.beta1 == 0.8
        assert opt.beta2 == 0.99
        assert opt.eps == 1e-6


class TestApplyUpdates:
    def test_array(self):
        params = np.array([1.0, 2.0])
        updates = np.array([-0.1, 0.2])
        result = apply_updates(params, updates)
        np.testing.assert_allclose(result, [0.9, 2.2])

    def test_dict(self):
        params = {"a": np.array([1.0]), "b": np.array([2.0])}
        updates = {"a": np.array([-0.1]), "b": np.array([0.2])}
        result = apply_updates(params, updates)
        np.testing.assert_allclose(result["a"], [0.9])
        np.testing.assert_allclose(result["b"], [2.2])


class TestAdamUpdate:
    def test_returns_updates_and_state(self, opt, params):
        state = opt.init(params)
        grad = np.array([1.0, 0.0, -1.0])
        updates, new_state = opt.update(grad, state, params)
        assert updates.shape == params.shape
        assert "m" in new_state and "v" in new_state and "t" in new_state

    def test_updates_oppose_gradient(self, opt, params):
        state = opt.init(params)
        grad = np.array([1.0, 0.0, -1.0])
        updates, _ = opt.update(grad, state, params)
        # Where gradient is positive, update should be negative
        assert updates[0] < 0
        # Where gradient is negative, update should be positive
        assert updates[2] > 0

    def test_apply_updates_moves_params(self, opt, params):
        state = opt.init(params)
        grad = np.array([1.0, 0.0, -1.0])
        updates, _ = opt.update(grad, state, params)
        new_params = apply_updates(params, updates)
        assert new_params[0] < params[0]
        assert new_params[2] > params[2]

    def test_state_accumulates(self, opt, params):
        state = opt.init(params)
        grad = np.array([1.0, 2.0, 3.0])
        _, state1 = opt.update(grad, state, params)
        assert state1["t"] == 1
        assert not np.allclose(state1["m"], 0.0)
        assert not np.allclose(state1["v"], 0.0)

        _, state2 = opt.update(grad, state1, params)
        assert state2["t"] == 2

    def test_numerical_correctness(self):
        """Verify one Adam step against hand-computed values."""
        lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
        opt = adam(learning_rate=lr, beta1=b1, beta2=b2, eps=eps)
        params = np.array([5.0])
        grad = np.array([2.0])
        state = opt.init(params)

        updates, new_state = opt.update(grad, state, params)

        # Hand compute
        m = b1 * 0 + (1 - b1) * 2.0  # 0.2
        v = b2 * 0 + (1 - b2) * 4.0  # 0.004
        m_hat = m / (1 - b1)  # 0.2 / 0.1 = 2.0
        v_hat = v / (1 - b2)  # 0.004 / 0.001 = 4.0
        expected_update = -lr * m_hat / (np.sqrt(v_hat) + eps)

        np.testing.assert_allclose(updates, [expected_update])
        np.testing.assert_allclose(new_state["m"], [m])
        np.testing.assert_allclose(new_state["v"], [v])
        assert new_state["t"] == 1

        # Verify apply_updates gives correct new params
        new_params = apply_updates(params, updates)
        np.testing.assert_allclose(new_params, [5.0 + expected_update])


class TestOptimize:
    def test_runs_and_returns_correct_shapes(self):
        """optimize() runs and returns params, state, history with correct structure."""

        def quadratic(x):
            return np.sum(x**2)

        params0 = np.array([3.0, -4.0])
        opt = adam(learning_rate=0.1)
        num_steps = 10

        params, state, history = optimize(quadratic, params0, opt, num_steps)

        assert params.shape == params0.shape
        assert "m" in state and "v" in state and "t" in state
        assert len(history["objective_fn_val"]) == num_steps
        assert len(history["grad_norm"]) == num_steps

    def test_objective_decreases(self):
        """Objective should generally decrease for a simple convex problem."""

        def quadratic(x):
            return np.sum(x**2)

        params0 = np.array([5.0, 5.0])
        opt = adam(learning_rate=0.1)

        _, _, history = optimize(quadratic, params0, opt, num_steps=50)

        assert history["objective_fn_val"][-1] < history["objective_fn_val"][0]

    def test_objective_increases_when_maximizing(self):
        """Objective should generally increase when direction='max'."""

        target = np.array([2.0, -3.0])

        def concave_quadratic(x):
            return -np.sum((x - target) ** 2)

        params0 = np.array([0.0, 0.0])
        opt = adam(learning_rate=0.1)

        params, _, history = optimize(
            concave_quadratic, params0, opt, num_steps=50, direction="max"
        )

        assert history["objective_fn_val"][-1] > history["objective_fn_val"][0]
        assert np.linalg.norm(params - target) < np.linalg.norm(params0 - target)

    def test_callback_called_each_step(self):
        """Callback should be called once per step with correct arguments."""
        call_log = []

        def cb(params, grad, state, step_index, objective_val):
            call_log.append(
                {
                    "step": step_index,
                    "params_shape": params.shape,
                    "grad_shape": grad.shape,
                    "val": objective_val,
                }
            )

        def quadratic(x):
            return np.sum(x**2)

        num_steps = 5
        optimize(quadratic, np.array([1.0]), adam(learning_rate=0.01), num_steps, callback=cb)

        assert len(call_log) == num_steps
        assert call_log[0]["step"] == 0
        assert call_log[-1]["step"] == num_steps - 1

    def test_multidimensional_params(self):
        """optimize() works with 3D array params (e.g. shape (nx, ny, nz))."""
        shape = (4, 5, 6)
        target = np.ones(shape) * 2.0

        def objective(x):
            return np.sum((x - target) ** 2)

        params0 = np.zeros(shape)
        opt = adam(learning_rate=0.05)

        params, state, history = optimize(objective, params0, opt, num_steps=20)

        assert params.shape == shape
        assert state["m"].shape == shape
        assert state["v"].shape == shape
        # Should be converging toward target
        assert history["objective_fn_val"][-1] < history["objective_fn_val"][0]

    def test_dict_params(self):
        """optimize() works when params is a dict of arrays and scalars."""

        target = {"weights": np.array([1.0, 2.0, 3.0]), "bias": np.array([0.5])}

        def objective(p):
            return np.sum((p["weights"] - target["weights"]) ** 2) + np.sum(
                (p["bias"] - target["bias"]) ** 2
            )

        params0 = {"weights": np.zeros(3), "bias": np.zeros(1)}
        opt = adam(learning_rate=0.05)

        params, state, history = optimize(objective, params0, opt, num_steps=30)

        assert set(params.keys()) == {"weights", "bias"}
        assert params["weights"].shape == (3,)
        assert state["m"]["weights"].shape == (3,)
        assert history["objective_fn_val"][-1] < history["objective_fn_val"][0]
        # grad_norm should be a positive float
        assert all(isinstance(g, float) and g >= 0 for g in history["grad_norm"])

    def test_bounds_array(self):
        """Array params are clipped to (lo, hi) after each step."""

        def objective(x):
            # Minimum at x = -10, but we bound to [0, 5]
            return np.sum((x + 10.0) ** 2)

        params0 = np.array([3.0, 4.0])
        opt = adam(learning_rate=0.5)

        params, _, _ = optimize(objective, params0, opt, num_steps=50, bounds=(0.0, 5.0))

        np.testing.assert_array_less(-1e-12, params)  # params >= 0
        np.testing.assert_array_less(params, 5.0 + 1e-12)  # params <= 5

    def test_bounds_array_none_side(self):
        """None on one side means unbounded on that side."""

        def objective(x):
            return np.sum((x - 100.0) ** 2)

        params0 = np.array([0.0])
        opt = adam(learning_rate=0.5)

        params, _, _ = optimize(objective, params0, opt, num_steps=50, bounds=(None, 2.0))

        assert params[0] <= 2.0 + 1e-12

    def test_bounds_dict(self):
        """Dict params are clipped per-key; missing keys are unclipped."""

        def objective(p):
            return np.sum((p["a"] - 10.0) ** 2) + np.sum((p["b"] + 10.0) ** 2)

        params0 = {"a": np.array([0.0]), "b": np.array([0.0])}
        opt = adam(learning_rate=0.5)

        params, _, _ = optimize(
            objective,
            params0,
            opt,
            num_steps=50,
            bounds={"a": (0.0, 3.0)},  # "b" left unclipped
        )

        assert params["a"][0] <= 3.0 + 1e-12
        assert params["a"][0] >= 0.0 - 1e-12
        # "b" should have moved freely toward -10
        assert params["b"][0] < -1.0

    def test_history_keys(self):
        def quadratic(x):
            return np.sum(x**2)

        _, _, history = optimize(quadratic, np.array([1.0]), adam(learning_rate=0.01), num_steps=3)

        assert set(history.keys()) == {"objective_fn_val", "grad_norm"}
        for v in history["objective_fn_val"]:
            assert isinstance(v, float)
        for v in history["grad_norm"]:
            assert isinstance(v, float)
            assert v >= 0

    def test_invalid_direction(self):
        def quadratic(x):
            return np.sum(x**2)

        with pytest.raises(ValueError, match="'direction' must be one of"):
            optimize(
                quadratic,
                np.array([1.0]),
                adam(learning_rate=0.01),
                num_steps=3,
                direction="sideways",
            )
