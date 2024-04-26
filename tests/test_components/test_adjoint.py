# tests adjoint parts of components

import tidy3d as td
import jax
import jax.numpy as jnp

x0 = 1.0


def make_box(x: float) -> td.Box:
    """Make a box storing (x,x,x) in its size."""
    return td.Box(
        center=(0, 0, 0),
        size=(x, x, x),
    )


def make_structure(x: float) -> td.Structure:
    """Make a structure with a geometry of a Box storing (x,x,x) in its size."""
    b = make_box(x)
    return td.Structure(geometry=b, medium=td.Medium(permittivity=2.0))


def make_sim(x: float) -> td.Simulation:
    """Make a sim with a structure with a geometry of a Box storing (x,x,x) in its size."""
    s = make_structure(x)
    return td.Simulation(
        size=(5, 5, 5),
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=[s],
    )


def sum_size(b: td.Box) -> float:
    """Sum the size of a box."""
    return jnp.sum(jnp.array(b.jax_info["size"]))


def test_jax_field_box():
    """Test storage of jax fields in tidy3d objects."""

    def f(x):
        b = make_box(x)
        return sum_size(b) ** 2

    val = f(x0)
    grad = jax.grad(f)(x0)

    assert val == (3 * x0) ** 2
    assert grad == 2 * 3 * (3 * x0)


def test_jax_field_structure():
    """Test storage of jax fields in nested tidy3d object."""

    def f(x):
        s = make_structure(x)
        return sum_size(s.geometry) ** 2

    val = f(x0)
    grad = jax.grad(f)(x0)

    assert val == (3 * x0) ** 2
    assert grad == 2 * 3 * (3 * x0)


def test_jax_field_sim():
    """Test storage of jax fields in a simulation."""

    def f(x):
        sim = make_sim(x)
        return sum_size(sim.structures[0].geometry) ** 2

    val = f(x0)
    grad = jax.grad(f)(x0)

    assert val == (3 * x0) ** 2
    assert grad == 2 * 3 * (3 * x0)


def test_passing_objects():
    """Test passing tidy3d objects into jax functions."""

    def g(s: td.Structure) -> float:
        return sum_size(s.structures[0].geometry) ** 2

    def f(x: float) -> float:
        s = make_sim(x)
        return g(s)

    val = f(x0)
    grad = jax.grad(f)(x0)

    assert val == (3 * x0) ** 2
    assert grad == 2 * 3 * (3 * x0)


def test_flatten():
    """Test flatten / unflatten operations on tidy3d objects."""

    x = jnp.array(x0)
    s1 = make_structure(x)

    leaves, treedef = jax.tree_util.tree_flatten(s1)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)

    assert s1 == s2
