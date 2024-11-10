import jax.numpy as jnp
from jax.scipy.stats import norm, expon

from base.data import RandomVar, ObservedVariable, Parameter
from methods.bayesian_network_v2 import Distribution, Edge, BayesianNetwork


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

def test_simple_normal_model():
    """Test simplest possible model: x ~ Normal(0, 1)"""
    # Create parameter and edge
    x = Parameter("x", (1,), lambda: jnp.array([1.0]))
    edge = Edge(
        child=x,
        distribution=Distribution.normal(loc=0.0, scale=1.0),
        name="x_prior"
    )

    # Create network
    model = BayesianNetwork(
        variables={"x": x},
        edges=[edge],
        param_size=1
    )

    # Test structure
    assert len(model.variables) == 1
    assert len(model.edges) == 1
    assert model.param_size == 1

    # Test log probability computation
    log_prob = model.log_prob()
    expected = norm.logpdf(1.0, loc=0.0, scale=1.0)
    assert jnp.abs(log_prob - expected) < 1e-5

def test_linear_regression_model():
    """Test linear regression: y ~ Normal(β₀ + β₁x, σ)

    With specific values:
    x = [1, 2, 3]
    y = [2.1, 3.9, 6.2]
    β₀ = 2.0
    β₁ = 1.0
    σ = 0.5
    """
    # Create parameters and data
    beta0 = Parameter("beta0", (1,), lambda: jnp.array([2.0]))
    beta1 = Parameter("beta1", (1,), lambda: jnp.array([1.0]))
    sigma = Parameter("sigma", (1,), lambda: jnp.array([0.5]))
    x = ObservedVariable("x", (3,), lambda: jnp.array([1.0, 2.0, 3.0]))
    y = ObservedVariable("y", (3,), lambda: jnp.array([2.1, 3.9, 6.2]))

    # Create edges for priors
    prior_edges = [
        Edge(beta0, Distribution.normal(loc=0.0, scale=10.0), "beta0_prior"),
        Edge(beta1, Distribution.normal(loc=0.0, scale=10.0), "beta1_prior"),
        Edge(sigma, Distribution.normal(loc=0.0, scale=10.0), "sigma_prior")
    ]

    # Create network
    model = BayesianNetwork(
        variables={"beta0": beta0, "beta1": beta1, "sigma": sigma, "x": x, "y": y},
        edges=prior_edges,
        param_size=3
    )

    # Test log probability
    log_prob = model.log_prob()

    # Manual calculation of expected log prob (just priors in this case)
    expected = (
            norm.logpdf(2.0, 0.0, 10.0) +  # beta0 prior
            norm.logpdf(1.0, 0.0, 10.0) +  # beta1 prior
            norm.logpdf(0.5, 0.0, 10.0)  # sigma prior
    )

    assert jnp.abs(log_prob - expected) < 1e-5

def test_hierarchical_model():
    """Test hierarchical model:
    μ ~ Normal(0, 10)
    σ ~ Exponential(1)
    x[i] ~ Normal(μ, σ) for i in 1..3

    With specific values:
    μ = 1.0
    σ = 1.0
    x = [1.1, 0.9, 1.2]
    """
    # Create parameters and data
    mu = Parameter("mu", (1,), lambda: jnp.array([1.0]))
    sigma = Parameter("sigma", (1,), lambda: jnp.array([1.0]))
    x = ObservedVariable("x", (3,), lambda: jnp.array([1.1, 0.9, 1.2]))

    # Create edges
    edges = [
        Edge(mu, Distribution.normal(loc=0.0, scale=1.0), "mu_prior"),
        Edge(sigma, Distribution.exponential(rate=1.0), "sigma_prior"),
        Edge(x, Distribution.normal(loc=mu, scale=sigma), "x_likelihood")
    ]

    # Create network
    model = BayesianNetwork(
        variables={"mu": mu, "sigma": sigma, "x": x},
        edges=edges,
        param_size=2
    )

    # Test log probability
    log_prob = model.log_prob()

    # Manual calculation of expected log prob
    expected = (
            norm.logpdf(1.0, 0.0, 1.0) +  # mu prior
            expon.logpdf(1.0, scale=1.0) +  # sigma prior
            jnp.sum(norm.logpdf(jnp.array([1.1, 0.9, 1.2]), loc=1.0, scale=1.0))  # likelihood
    )

    assert jnp.abs(log_prob - expected) < 1e-5

def test_potential_energy():
    """Test that potential energy is negative log probability"""
    # Simple model: x ~ Normal(0, 1)
    x = Parameter("x", (1,), lambda: jnp.array([1.0]))
    edge = Edge(
        child=x,
        distribution=Distribution.normal(loc=0.0, scale=1.0),
        name="x_prior"
    )

    model = BayesianNetwork(
        variables={"x": x},
        edges=[edge],
        param_size=1
    )

    log_prob = model.log_prob()
    potential = model.potential_energy()

    assert jnp.abs(potential + log_prob) < 1e-5


def test_hamiltonian_energy_conservation():
    """Test gradient computation in context of Hamiltonian Monte Carlo.

    In HMC, we have:
    - Potential energy U(q) = -log(p(q))  [our probability model]
    - Kinetic energy K(p) = p^T p / 2     [momentum term]
    - Total energy H(p,q) = U(q) + K(p)   [should be conserved]

    The gradient of U(q) should lead to correct momentum updates:
    dp/dt = -dU/dq
    dq/dt = p
    """
    # Create simple normal model: q ~ Normal(0, 1)
    q = Parameter("q", (1,), lambda: jnp.array([1.0]))
    edge = Edge(
        child=q,
        distribution=Distribution.normal(loc=0.0, scale=1.0),
        name="normal_prior"
    )

    model = BayesianNetwork(
        variables={"q": q},
        edges=[edge],
        param_size=1
    )

    # Initial state
    q0 = jnp.array([1.0])  # position
    p0 = jnp.array([0.5])  # momentum

    # Compute initial energies
    U0 = model.potential_energy()  # -log(p(q))
    K0 = jnp.sum(p0 ** 2) / 2  # p^T p / 2
    H0 = U0 + K0  # total energy

    # Single leapfrog step
    epsilon = 0.1
    #TODO: Fix Gradient
    grad = model.gradient()
    p1 = p0 - epsilon * grad / 2  # half step in momentum
    q1 = q0 + epsilon * p1  # full step in position

    # Update model with new position
    q = Parameter("q", (1,), lambda: q1)
    edge = Edge(
        child=q,
        distribution=Distribution.normal(loc=0.0, scale=1.0),
        name="normal_prior"
    )
    model = BayesianNetwork(
        variables={"q": q},
        edges=[edge],
        param_size=1
    )

    grad = model.gradient()
    p1 = p1 - epsilon * grad / 2  # half step in momentum

    # Compute final energies
    U1 = model.potential_energy()
    K1 = jnp.sum(p1 ** 2) / 2
    H1 = U1 + K1

    # Total energy should be approximately conserved
    assert jnp.abs(H1 - H0) < 1e-3
