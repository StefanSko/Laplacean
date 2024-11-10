import jax.numpy as jnp
from jax.scipy.stats import norm

from base.data import RandomVar
from methods.bayesian_network_v2 import Distribution


def test_normal_with_constant_parameters():
    # Create a mock random variable
    mock_rv = RandomVar("test", (1,), lambda: jnp.array([1.0]))

    # Create normal distribution with constant parameters
    dist = Distribution.normal(loc=0.0, scale=1.0)

    # Compute log probability
    log_prob = dist.log_prob(mock_rv)

    # Compare with expected value
    expected = -1.418939  # log(1/√(2π)) - 0.5
    assert jnp.abs(log_prob - expected) < 1e-5

def test_normal_with_random_parameters():
    # Create random variables for value, location, and scale
    value_rv = RandomVar("value", (1,), lambda: jnp.array([1.0]))
    loc_rv = RandomVar("loc", (1,), lambda: jnp.array([0.0]))
    scale_rv = RandomVar("scale", (1,), lambda: jnp.array([1.0]))

    # Create normal distribution with random variable parameters
    dist = Distribution.normal(loc=loc_rv, scale=scale_rv)

    # Compute log probability
    log_prob = dist.log_prob(value_rv)

    # Compare with expected value
    expected = -1.418939  # log(1/√(2π)) - 0.5
    assert jnp.abs(log_prob - expected) < 1e-5

def test_normal_with_vector_input():
    # Test with vector-valued random variable
    vector_rv = RandomVar("vector", (3,), lambda: jnp.array([1.0, 2.0, 3.0]))

    dist = Distribution.normal(loc=0.0, scale=1.0)
    log_prob = dist.log_prob(vector_rv)

    # Sum of individual log probabilities
    log_probs = norm.logpdf(jnp.array([1.0, 2.0, 3.0]), loc=0.0, scale=1.0)
    expected = jnp.sum(log_probs)
    assert jnp.abs(log_prob - expected) < 1e-5

def test_exponential_distribution():
    rv = RandomVar("test", (1,), lambda: jnp.array([1.0]))
    dist = Distribution.exponential(rate=1.0)

    log_prob = dist.log_prob(rv)

    # For exponential distribution with rate λ=1:
    # log(p(x)) = -x + log(λ) = -1 + log(1) = -1
    expected = -1.0
    assert jnp.abs(log_prob - expected) < 1e-5

def test_exponential_with_vector_input():
    vector_rv = RandomVar("vector", (3,), lambda: jnp.array([1.0, 2.0, 3.0]))
    dist = Distribution.exponential(rate=1.0)

    log_prob = dist.log_prob(vector_rv)

    # Sum of log probabilities: Σ(-x_i) = -(1 + 2 + 3) = -6
    expected = -(1.0 + 2.0 + 3.0)
    assert jnp.abs(log_prob - expected) < 1e-5