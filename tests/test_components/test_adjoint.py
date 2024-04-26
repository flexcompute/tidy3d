# tests adjoint parts of components

import tidy3d as td
import jax
import jax.numpy as jnp


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


def test_jax_field():
    """Test storage of jax fields in tidy3d objects."""

    def f(x):
        b = make_box(x)
        return jnp.sum(jnp.array(b.jax_info["size"]))

    val = f(1.0)
    grad = jax.grad(f)(1.0)

    assert val >= 0.0
    assert abs(grad) >= 0.0


def test_jax_field_nested():
    """Test storage of jax fields in nested tidy3d objects."""

    def f(x):
        s = make_structure(x)
        return jnp.sum(jnp.array(s.geometry.jax_info["size"]))

    val = f(1.0)
    grad = jax.grad(f)(1.0)

    assert val >= 0.0
    assert abs(grad) >= 0.0


def test_passing_objects():
    """Test passing tidy3d objects into jax functions."""

    def g(s: td.Structure) -> float:
        return jnp.sum(jnp.array(s.geometry.jax_info["size"])) ** 2

    def f(x: float) -> float:
        s = make_structure(x)
        return g(s)

    x0 = 1.0

    val = f(x0)
    grad = jax.grad(f)(x0)

    assert val == (3 * x0) ** 2
    assert grad == 2 * 3 * (3 * x0)


def test_flatten():
    """Test flatten / unflatten operations on tidy3d objects."""

    x = jnp.array(1.0)
    s1 = make_structure(x)

    leaves, treedef = jax.tree_util.tree_flatten(s1)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)

    assert s1 == s2
