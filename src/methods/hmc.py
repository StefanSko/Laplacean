from typing import Callable, Tuple
from jax import random, lax, debug
import jax.numpy as jnp
from jaxtyping import Float, Bool, Array, PRNGKeyArray

from logging_utils import hamiltonians_print, acceptance_print, step_output_print, \
    after_leapfrog_print, generated_momentum_print, step_input_print
from base.data import JaxHMCData
from methods.bayesian_execution_network import BayesianExecutionModel
from util import conditional_jit

StepFunc = Callable[[BayesianExecutionModel, JaxHMCData], JaxHMCData]

def leapfrog_step(q: jnp.ndarray, p: jnp.ndarray, epsilon: float, U: BayesianExecutionModel) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform a single leapfrog step."""
    p = p - epsilon * U.gradient(q) / 2  # Half step for momentum
    q = q + epsilon * p  # Full step for position
    p = p - epsilon * U.gradient(q) / 2  # Half step for momentum
    return q, p

def leapfrog_integrate(q: jnp.ndarray, p: jnp.ndarray, epsilon: float, L: int, U: BayesianExecutionModel) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform L steps of leapfrog integration."""
    def body_fun(i, carry):
        q, p = carry
        return leapfrog_step(q, p, epsilon, U)
    return lax.fori_loop(0, L, body_fun, (q, p))

def compute_hamiltonian(q: jnp.ndarray, p: jnp.ndarray, U: BayesianExecutionModel) -> Float[Array, ""]:
    """Compute the Hamiltonian (total energy) of the system."""
    return U.potential_energy(q) + jnp.sum(p ** 2) / 2

def metropolis_accept(key: PRNGKeyArray, current_h: Float[Array, ""], proposed_h: Float[Array, ""]) -> Tuple[Bool[Array, ""], PRNGKeyArray]:
    """Perform Metropolis acceptance step."""
    log_accept_prob = current_h - proposed_h
    log_accept_prob = jnp.where(jnp.isfinite(log_accept_prob), log_accept_prob, -jnp.inf)
    key, subkey = random.split(key)
    accept = jnp.log(random.uniform(subkey)) < log_accept_prob
    return accept, key

@conditional_jit(use_jit=True)
def step(U: BayesianExecutionModel, input: JaxHMCData) -> JaxHMCData:
    q = input.current_q
    key, subkey = random.split(input.key)
    p = random.normal(subkey, q.shape)

    debug.callback(step_input_print, q=q, key=input.key)
    debug.callback(generated_momentum_print, p=p)

    # Perform leapfrog integration
    q_new, p_new = leapfrog_integrate(q, p, input.epsilon, input.L, U)

    debug.callback(after_leapfrog_print, q_new=q_new, p_new=p_new)

    # Negate momentum for symmetry
    p_new = -p_new

    # Compute Hamiltonians
    current_h = compute_hamiltonian(q, p, U)
    proposed_h = compute_hamiltonian(q_new, p_new, U)

    debug.callback(hamiltonians_print, current=current_h, proposed=proposed_h)

    # Metropolis acceptance step
    accept, key = metropolis_accept(key, current_h, proposed_h)

    debug.callback(acceptance_print, accept=accept)

    # Update position based on acceptance
    q_new = jnp.where(accept, q_new, q)

    debug.callback(step_output_print, q_new=q_new, key=key)

    return JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=q_new, key=key)