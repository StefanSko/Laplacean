import jax.numpy as jnp
from jax import lax



def assert_allclose(a, b, rtol=1e-7, atol=1e-7):
    diff = jnp.abs(a - b)
    max_abs = jnp.maximum(jnp.abs(a), jnp.abs(b))
    tol = atol + rtol * max_abs
    assert jnp.all(lax.le(diff, tol)), f"Arrays not close: {a} vs {b}"
