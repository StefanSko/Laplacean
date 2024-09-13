import pytest
import jax.numpy as jnp
import jax.random as random

from logging_utils import logger
from methods.hmc import step
from sampler.sampling import Sampler
from base.data import JaxHMCData
from methods.potential_energy import LaplaceanPotentialEnergy, ConstantLogDensity, GaussianLogDensity


def mock_step(energy: LaplaceanPotentialEnergy, data: JaxHMCData) -> JaxHMCData:
    # Simple mock step function that just adds 1 to current_q
    new_q = data.current_q + 1
    new_key = random.split(data.key)[0]
    return JaxHMCData(epsilon=data.epsilon, L=data.L, current_q=new_q, key=new_key)

@pytest.fixture
def sampler():
    return Sampler()

@pytest.fixture
def energy():
    return LaplaceanPotentialEnergy(
        log_prior=ConstantLogDensity(),
        log_likelihood=ConstantLogDensity()
    )

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
    mean = jnp.array([0.0])
    var = jnp.array([1.0])
    potential_energy = LaplaceanPotentialEnergy(
        log_prior=GaussianLogDensity(mean=mean, var=var),
        log_likelihood=ConstantLogDensity()
    )

    initial_q = jnp.array([1.0])
    
    # Try different epsilon and L values
    for epsilon in [0.01, 0.05, 0.1]:
        for L in [5, 10, 20]:
            input_data = JaxHMCData(epsilon=epsilon, L=L, current_q=initial_q, key=random.PRNGKey(0))

            sampler = Sampler()
            samples = sampler(step, input_data, potential_energy, num_warmup=100, num_samples=1000)

            # Log the results
            logger.info(f"Epsilon: {epsilon}, L: {L}")
            logger.info(f"Mean: {jnp.mean(samples)}, Std: {jnp.std(samples)}")
            logger.info(f"Min: {jnp.min(samples)}, Max: {jnp.max(samples)}")

    # Check that the mean and variance are close to the true values
    assert jnp.abs(jnp.mean(samples) - mean[0]) < 0.1
    assert jnp.abs(jnp.var(samples) - var[0]) < 0.1

    # Check that samples are not all the same
    assert jnp.std(samples) > 0.1