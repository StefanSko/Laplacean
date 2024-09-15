

import jax.numpy as jnp

from methods.potential_energy import normal_log_density, BayesianModel
from tests.test_utils import assert_allclose


def test_potential_energy():
    prior = normal_log_density(mean=jnp.array(0.0), std=jnp.array(0.0))

    model = BayesianModel((prior,))

    result = model.potential_energy(jnp.array([0.]))
    assert isinstance(result, jnp.ndarray)

def test_gradient():
    prior = normal_log_density(mean=jnp.array(0.0), std=jnp.array(0.0))

    model = BayesianModel((prior,))
    result = model.gradient(jnp.array([0.]))
    assert isinstance(result, jnp.ndarray)
    assert_allclose(result, jnp.array(0.0), rtol=1e-7, atol=1e-7)


def test_gaussian_potential_energy():
    mean = jnp.array([1.0, -1.0])
    var = jnp.array([0.5, 2.0])
    prior = normal_log_density(mean=mean, std=var)

    # Test at mean
    result_at_mean = potential_energy(mean)
    assert_allclose(result_at_mean, jnp.array(0.0), rtol=1e-7, atol=1e-7)
    
    # Test away from mean
    q = jnp.array([0.0, 0.0])
    result = potential_energy(q)
    # Directional checks
    assert result < 0, "Potential energy should be negative away from mean"
    
    # Check that moving further from mean increases potential energy
    q_further = jnp.array([-1.0, 2.0])
    result_further = potential_energy(q_further)
    assert result_further < result, "Potential energy should decrease as we move further from mean"

def test_gaussian_gradient():
    mean = jnp.array([1.0, -1.0])
    var = jnp.array([0.5, 2.0])
    potential_energy = LaplaceanPotentialEnergy(
        log_prior=GaussianLogDensity(mean=mean, var=var),
        log_likelihood=ConstantLogDensity()
    )
    
    # Test at mean
    gradient_at_mean = potential_energy.gradient(mean)
    assert_allclose(gradient_at_mean, jnp.zeros_like(mean), rtol=1e-7, atol=1e-7)
    
    # Test away from mean
    q = jnp.array([0.0, 0.0])
    gradient = potential_energy.gradient(q)
    
    # Directional checks
    assert gradient[0] > 0, "Gradient should be positive in first dimension (pointing towards mean)"
    assert gradient[1] < 0, "Gradient should be negative in second dimension (pointing towards mean)"
    
    # Check that gradient magnitude increases as we move further from mean
    q_further = jnp.array([-1.0, 2.0])
    gradient_further = potential_energy.gradient(q_further)
    assert jnp.linalg.norm(gradient_further) > jnp.linalg.norm(gradient), "Gradient magnitude should increase as we move further from mean"
    
    # Check that gradient points towards the mean
    assert jnp.dot(gradient, mean - q) > 0, "Gradient should point towards the mean"



def test_custom_log_likelihood():
    class CustomLogLikelihood(ConstantLogDensity):
        def __call__(self, q: jnp.ndarray) -> jnp.ndarray:
            return -jnp.sum(q**2)  # Simple quadratic function
    
    potential_energy = LaplaceanPotentialEnergy(
        log_prior=GaussianLogDensity(mean=jnp.array([0.0]), var=jnp.array([1.0])),
        log_likelihood=CustomLogLikelihood()
    )
    
    q = jnp.array([2.0])
    expected_energy = -0.5 * 2.0**2 - 2.0**2  # Prior + likelihood
    result = potential_energy(q)
    assert_allclose(result, expected_energy, rtol=1e-7, atol=1e-7)
    
    expected_gradient = -2.0 - 4.0  # Gradient of prior + likelihood
    result_gradient = potential_energy.gradient(q)
    assert_allclose(result_gradient, expected_gradient, rtol=1e-7, atol=1e-7)

