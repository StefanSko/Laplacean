from typing import Optional

import pytest
import jax.numpy as jnp
from jax import random, Array
from jaxtyping import Float

from methods.hmc import (
    leapfrog_step,
    leapfrog_integrate,
    compute_hamiltonian,
    metropolis_accept,
    step
)
from methods.potential_energy import BayesianModel, LogDensity, constant_log_density, normal_log_density
from base.data import JaxHMCData

# Mock potential energy function for testing
def mock_log_prior() -> LogDensity:
    def log_prob(q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        return -jnp.sum(q**2) / 2
    return LogDensity(log_prob)



@pytest.fixture
def mock_potential():
    return BayesianModel((mock_log_prior(), constant_log_density()))

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
    epsilon = 0.01
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
    potential_energy = mock_potential.potential_energy(q)
    
    assert jnp.isclose(total_energy, kinetic_energy + potential_energy)

def test_metropolis_accept_detailed_balance():
    key = random.PRNGKey(0)
    n_samples = 10000
    current_h = 1.0
    proposed_h = 1.5
    
    accepts = []
    for _ in range(n_samples):
        accept, key = metropolis_accept(key, jnp.array(current_h), jnp.array(proposed_h))
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


def test_leapfrog_energy_conservation():
    # Define the model with a simple normal log density
    model = BayesianModel((normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0)),))
    
    # Initialize positions and momenta
    q = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.1, 0.2, 0.3])
    
    # Set integrator parameters
    epsilon = 0.01  # Smaller step size for better accuracy
    L = 10  # Number of leapfrog steps
    
    # Compute initial Hamiltonian
    H_initial = compute_hamiltonian(q, p, model)
    
    # Perform leapfrog integration
    q_new, p_new = leapfrog_integrate(q, p, epsilon, L, model)
    
    # Compute final Hamiltonian
    H_final = compute_hamiltonian(q_new, p_new, model)
    
    # Check energy conservation
    energy_diff = jnp.abs(H_final - H_initial)
    assert energy_diff < 1e-3, f"Energy difference {energy_diff} exceeds threshold"