# tests adjoint parts of components

import tidy3d as td
import jax
import jax.numpy as jnp

def test_jax_field():


	def f(x):
		b = td.Box(
			center=(0,0,0),
			size=(x, x, x),
		)

		return jnp.sum(jnp.array(b.jax_info["size"]))

	val = f(1.0)
	grad = jax.grad(f)(1.0)

	assert val >= 0.0
	assert abs(grad) >= 0.0

