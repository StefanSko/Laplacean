

import jax.numpy as jnp

from methods.potential_energy import normal_log_density, BayesianModel
from tests.test_utils import assert_allclose


def test_potential_energy():
    prior = normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0))

    model = BayesianModel((prior,))

    result = model.potential_energy(jnp.array([0.]))
    assert isinstance(result, jnp.ndarray)

def test_gradient():
    prior = normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0))

    model = BayesianModel((prior,))
    result = model.gradient(jnp.array([0.]))
    assert isinstance(result, jnp.ndarray)
    assert_allclose(result, jnp.array(0.0), rtol=1e-7, atol=1e-7)


def test_gaussian_potential_energy():
    mean = jnp.array([1.0, -1.0])
    std = jnp.array([0.5, 2.0])
    prior = normal_log_density(mean=mean, std=std)

    model = BayesianModel((prior,))

    # Test at mean
    result_at_mean = model.potential_energy(mean)
    assert_allclose(result_at_mean, 0.0, rtol=1e-7, atol=1e-7)
    
    # Test away from mean
    q = jnp.array([0.0, 0.0])
    result = model.potential_energy(q)
    assert result > 0, "Potential energy should be positive away from mean"
    
    # Further away from mean
    q_further = jnp.array([-1.0, 2.0])
    result_further = model.potential_energy(q_further)
    assert result_further > result, "Potential energy should increase as we move further from mean"

def test_gaussian_gradient():
    mean = jnp.array([1.0, -1.0])
    std = jnp.array([0.5, 2.0])
    prior = normal_log_density(mean=mean, std=std)
    model = BayesianModel((prior,))
    
    # Test at mean
    gradient_at_mean = model.gradient(mean)
    assert_allclose(gradient_at_mean, jnp.zeros_like(mean), rtol=1e-7, atol=1e-7)
    
    # Test away from mean
    q = jnp.array([0.0, 0.0])
    gradient = model.gradient(q)
    
    # Directional checks
    assert gradient[0] < 0, "Gradient should be negative in first dimension (pointing away from mean)"
    assert gradient[1] > 0, "Gradient should be positive in second dimension (pointing away from mean)"
    
    # Check that gradient magnitude increases as we move further from mean
    q_further = jnp.array([-1.0, 2.0])
    gradient_further = model.gradient(q_further)
    assert jnp.linalg.norm(gradient_further) > jnp.linalg.norm(gradient), "Gradient magnitude should increase as we move further from mean"
    
    # Check that gradient points away from the mean
    assert jnp.dot(gradient, mean - q) < 0, "Gradient should point towards the mean"


def test_gradient_accuracy():
    # Define the model
    model = BayesianModel((normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0)),))

    # Sample a point
    q = jnp.array([1.0, 2.0, 3.0])

    # Compute gradients
    grad_U = model.gradient(q)

    # Expected gradient for U = 0.5 * sum(q^2) is grad_U = q
    expected_grad_U = q

    assert jnp.allclose(grad_U, expected_grad_U, atol=1e-5), f"Gradients do not match: {grad_U} vs {expected_grad_U}"

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

