

import jax.numpy as jnp

from methods.potential_energy import LaplaceanPotentialEnergy, GaussianLogDensity, ConstantLogDensity


def test_potential_energy():
    potential_energy = LaplaceanPotentialEnergy(log_prior=GaussianLogDensity(mean=jnp.array([0.]), var=jnp.array([1.])), log_likelihood=ConstantLogDensity())
    assert potential_energy(jnp.array([0.])) == 0.0

def test_gradient(potential_energy):
    assert potential_energy.gradient(jnp.array([0.])) == 0.0
