import jax.numpy as jnp
from jax.scipy.stats import norm

from base.data import RandomVar, ObservedVariable, Parameter
from methods.bayesian_network_v2 import Distribution, Edge


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

def test_edge_normal_relationship():
    # Test edge representing y ~ Normal(μ, σ)
    y = ObservedVariable("y", (1,), lambda: jnp.array([2.0]))
    mu = Parameter("mu", (1,), lambda: jnp.array([1.0]))
    sigma = Parameter("sigma", (1,), lambda: jnp.array([1.0]))

    dist = Distribution.normal(loc=mu, scale=sigma)
    edge = Edge(child=y, distribution=dist, name="y_normal")

    # Verify log probability computation
    log_prob = edge.log_prob()
    expected = norm.logpdf(2.0, loc=1.0, scale=1.0)
    assert jnp.abs(log_prob - expected) < 1e-5

    # Verify string representation
    assert str(edge) == "Edge(y_normal)"

def test_edge_hierarchical_relationship():
    """Test a hierarchical relationship where one parameter depends on another.
    
    Model:
    x ~ Normal(0, 1)     # Prior on x
    y|x ~ Normal(x, 1)   # y conditional on x
    
    Test values:
    x = 1.0
    y = 2.0
    """
    # Create the parameters
    x = Parameter("x", (1,), lambda: jnp.array([1.0]))
    y = Parameter("y", (1,), lambda: jnp.array([2.0]))
    
    # Create two edges representing the full hierarchical relationship
    prior_edge = Edge(
        child=x,
        distribution=Distribution.normal(loc=0.0, scale=1.0),
        name="x_prior"
    )
    
    conditional_edge = Edge(
        child=y,
        distribution=Distribution.normal(loc=x, scale=1.0),
        name="y_given_x"
    )
    
    # Test the prior: p(x)
    prior_log_prob = prior_edge.log_prob()
    expected_prior = norm.logpdf(1.0, loc=0.0, scale=1.0)
    assert jnp.abs(prior_log_prob - expected_prior) < 1e-5
    
    # Test the conditional: p(y|x)
    conditional_log_prob = conditional_edge.log_prob()
    expected_conditional = norm.logpdf(2.0, loc=1.0, scale=1.0)
    assert jnp.abs(conditional_log_prob - expected_conditional) < 1e-5

def test_edge_vector_relationship():
    # Test edge with vector-valued random variables
    # y ~ Normal(μ, σ) where y and μ are vectors
    y = ObservedVariable("y", (3,), lambda: jnp.array([1.0, 2.0, 3.0]))
    mu = Parameter("mu", (3,), lambda: jnp.array([0.0, 1.0, 2.0]))

    edge = Edge(
        child=y,
        distribution=Distribution.normal(loc=mu, scale=1.0),
        name="vector_normal"
    )

    # Each component follows its own normal distribution
    log_prob = edge.log_prob()
    expected = jnp.sum(norm.logpdf(
        jnp.array([1.0, 2.0, 3.0]),
        loc=jnp.array([0.0, 1.0, 2.0]),
        scale=1.0
    ))
    assert jnp.abs(log_prob - expected) < 1e-5