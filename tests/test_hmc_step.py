import pytest
import jax.numpy as jnp
from jax import random
from methods.hmc import (
    leapfrog_step,
    leapfrog_integrate,
    compute_hamiltonian,
    metropolis_accept,
    step
)
from methods.potential_energy import LaplaceanPotentialEnergy, LogDensity
from base.data import JaxHMCData

# Mock potential energy function for testing
class MockLogPrior(LogDensity):
    def __call__(self, q):
        return -jnp.sum(q**2) / 2

class MockLogLikelihood(LogDensity):
    def __call__(self, q):
        return 0.0


@pytest.fixture
def mock_potential():
    return LaplaceanPotentialEnergy(
        log_prior=MockLogPrior(),
        log_likelihood=MockLogLikelihood()
    )

def test_leapfrog_step_energy_conservation(mock_potential):
    q = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.1, 0.2, 0.3])
    epsilon = 0.1
    
    initial_energy = compute_hamiltonian(q, p, mock_potential)
    q_new, p_new = leapfrog_step(q, p, epsilon, mock_potential)
    final_energy = compute_hamiltonian(q_new, p_new, mock_potential)
    
    assert jnp.abs(final_energy - initial_energy) < 1e-3

def test_leapfrog_integrate_reversibility(mock_potential):
    q = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.1, 0.2, 0.3])
    epsilon = 0.1
    L = 10
    
    q_forward, p_forward = leapfrog_integrate(q, p, epsilon, L, mock_potential)
    q_reverse, p_reverse = leapfrog_integrate(q_forward, -p_forward, epsilon, L, mock_potential)
    
    assert jnp.allclose(q, q_reverse, atol=1e-6)
    assert jnp.allclose(p, -p_reverse, atol=1e-6)

def test_compute_hamiltonian_separability(mock_potential):
    q = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.1, 0.2, 0.3])
    
    total_energy = compute_hamiltonian(q, p, mock_potential)
    kinetic_energy = jnp.sum(p**2) / 2
    potential_energy = mock_potential(q)
    
    assert jnp.isclose(total_energy, kinetic_energy + potential_energy)

def test_metropolis_accept_detailed_balance():
    key = random.PRNGKey(0)
    n_samples = 100000
    current_h = 1.0
    proposed_h = 1.5
    
    accepts = []
    for _ in range(n_samples):
        accept, key = metropolis_accept(key, current_h, proposed_h)
        accepts.append(accept)
    
    acceptance_rate = jnp.mean(jnp.array(accepts))
    expected_rate = jnp.exp(current_h - proposed_h)
    
    assert jnp.abs(acceptance_rate - expected_rate) < 0.01

def test_step_dimensionality_preservation(mock_potential):
    key = random.PRNGKey(0)
    q = jnp.array([1.0, 2.0, 3.0])
    epsilon = 0.1
    L = 10
    input_data = JaxHMCData(epsilon=epsilon, L=L, current_q=q, key=key)

    
    output_data = step(mock_potential, input_data)
    
    assert output_data.current_q.shape == input_data.current_q.shape
    assert output_data.epsilon == input_data.epsilon
    assert output_data.L == input_data.L

def test_step_energy_conservation(mock_potential):
    key = random.PRNGKey(0)
    q = jnp.array([1.0, 2.0, 3.0])
    epsilon = 0.01  # Small step size for better energy conservation
    L = 100
    input_data = JaxHMCData(epsilon=epsilon, L=L, current_q=q, key=key)
    
    initial_energy = compute_hamiltonian(q, jnp.zeros_like(q), mock_potential)
    
    n_steps = 1000
    current_q = q
    for _ in range(n_steps):
        output_data = step(mock_potential, input_data)
        current_q = output_data.current_q
        input_data = output_data
    
    final_energy = compute_hamiltonian(current_q, jnp.zeros_like(current_q), mock_potential)
    
    assert jnp.abs(final_energy - initial_energy) / initial_energy < 0.05  # Allow for 5% variation