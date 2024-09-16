import pytest
import jax.numpy as jnp
import jax.random as random

from methods.hmc import step
from methods.potential_energy import BayesianModel, normal_log_density, constant_log_density
from sampler.sampling import Sampler
from base.data import JaxHMCData


def mock_step(model: BayesianModel, data: JaxHMCData) -> JaxHMCData:
    # Simple mock step function that just adds 1 to current_q
    new_q = data.current_q + 1
    new_key = random.split(data.key)[0]
    return JaxHMCData(epsilon=data.epsilon, L=data.L, current_q=new_q, key=new_key)

@pytest.fixture
def sampler():
    return Sampler()

@pytest.fixture
def energy():
    return BayesianModel((constant_log_density(),))

@pytest.fixture
def init_data():
    return JaxHMCData(epsilon=0.1, L=10, current_q=jnp.array([0.0]), key=random.PRNGKey(0))

def test_sampler_immediate(sampler, energy, init_data):
    # Run the sampler with our mock step function
    samples = sampler(mock_step, init_data, energy, num_warmup=5, num_samples=10)

    # Check the shape of the output
    assert samples.shape == (10, 1), f"Expected shape (10, 1), got {samples.shape}"

    expected_samples = jnp.arange(6, 16).reshape(-1, 1)
    assert jnp.allclose(samples, expected_samples), f"Expected {expected_samples}, got {samples}"

def test_sampler_different_warmup_and_samples(sampler, energy, init_data):
    # Test with different warmup and sample numbers
    samples = sampler(mock_step, init_data, energy, num_warmup=3, num_samples=7)

    assert samples.shape == (7, 1), f"Expected shape (7, 1), got {samples.shape}"
    expected_samples = jnp.arange(4, 11).reshape(-1, 1)
    assert jnp.allclose(samples, expected_samples), f"Expected {expected_samples}, got {samples}"

def test_sampler_multidimensional(sampler, energy):
    # Test with multidimensional input
    init_data_multi = JaxHMCData(epsilon=0.1, L=10, current_q=jnp.array([0.0, 0.0]), key=random.PRNGKey(0))
    samples = sampler(mock_step, init_data_multi, energy, num_warmup=2, num_samples=5)

    assert samples.shape == (5, 2), f"Expected shape (5, 2), got {samples.shape}"


def test_gaussian_sampling():
    # Set up a simple Gaussian distribution
    
    mean = 0.0
    var = 1.0
    
    prior = normal_log_density(mean=jnp.array(mean), std=jnp.array(var))
    model = BayesianModel((prior,))

    initial_q = jnp.array([0.0])
    
    epsilon = 0.05
    L = 11
    
    input_data = JaxHMCData(epsilon=epsilon, L=L, current_q=initial_q, key=random.PRNGKey(0))

    sampler = Sampler()
    samples = sampler(step, input_data, model, num_warmup=100, num_samples=1000)

    # Check that the mean and variance are close to the true values
    assert jnp.abs(jnp.mean(samples) - mean) < 0.3
    assert jnp.abs(jnp.var(samples) - var) < 0.3