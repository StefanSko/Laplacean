from typing import Callable, Tuple
from jax import random, lax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Bool, Array, PRNGKeyArray

from methods.potential_energy import LaplaceanPotentialEnergy
from base.data import JaxHMCData

StepFunc = Callable[[LaplaceanPotentialEnergy, JaxHMCData], JaxHMCData]

def leapfrog_step(q: jnp.ndarray, p: jnp.ndarray, epsilon: float, U: LaplaceanPotentialEnergy) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform a single leapfrog step."""
    p = p - epsilon * U.gradient(q) / 2  # Half step for momentum
    q = q + epsilon * p  # Full step for position
    p = p - epsilon * U.gradient(q) / 2  # Half step for momentum
    return q, p

def leapfrog_integrate(q: jnp.ndarray, p: jnp.ndarray, epsilon: float, L: int, U: LaplaceanPotentialEnergy) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform L steps of leapfrog integration."""
    def body_fun(i, carry):
        q, p = carry
        return leapfrog_step(q, p, epsilon, U)
    return lax.fori_loop(0, L, body_fun, (q, p))

def compute_hamiltonian(q: jnp.ndarray, p: jnp.ndarray, U: LaplaceanPotentialEnergy) -> Float[Array, ""]:
    """Compute the Hamiltonian (total energy) of the system."""
    return U(q) + jnp.sum(p ** 2) / 2

def metropolis_accept(key: PRNGKeyArray, current_h: Float[Array, ""], proposed_h: Float[Array, ""]) -> Tuple[Bool[Array, ""], PRNGKeyArray]:
    """Perform Metropolis acceptance step."""
    log_accept_prob = current_h - proposed_h
    log_accept_prob = jnp.where(jnp.isfinite(log_accept_prob), log_accept_prob, -jnp.inf)
    key, subkey = random.split(key)
    accept = jnp.log(random.uniform(subkey)) < log_accept_prob
    return accept, key

def step(U: LaplaceanPotentialEnergy, input: JaxHMCData) -> JaxHMCData:
    q = input.current_q
    key, subkey = random.split(input.key)
    p = random.normal(subkey, q.shape)
    
    # Perform leapfrog integration
    q_new, p_new = leapfrog_integrate(q, p, input.epsilon, input.L, U)
    
    # Negate momentum for symmetry
    p_new = -p_new
    
    # Compute Hamiltonians
    current_h = compute_hamiltonian(q, p, U)
    proposed_h = compute_hamiltonian(q_new, p_new, U)
    
    # Metropolis acceptance step
    accept, key = metropolis_accept(key, current_h, proposed_h)
    
    # Update position based on acceptance
    q_new = lax.cond(accept, lambda _: q_new, lambda _: q, operand=None)
    
    return JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=q_new, key=key)

# Convert step to an Equinox filter function
step_filter = eqx.filter_jit(step)