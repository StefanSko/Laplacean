from typing import Callable

from jax import random, lax, jit
import jax.numpy as jnp


from methods.potential_energy import PotentialEnergy
from base.data import JaxHMCData


StepFunc = Callable[[PotentialEnergy, JaxHMCData], JaxHMCData]

@jit
def step(U: PotentialEnergy, input: JaxHMCData) -> JaxHMCData:
    q = input.current_q
    key, subkey = random.split(input.key)
    p = random.normal(subkey, q.shape)
    epsilon = input.epsilon
    L = input.L
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - epsilon * U.gradient(q) / 2

    def loop_body(_, carry):
        q, p = carry
        # Make a full step for the position
        q = q + epsilon * p
        # Make a full step for the momentum
        p = p - epsilon * U.gradient(q)
        return q, p

    q, p = lax.fori_loop(0, L- 1, loop_body, (q, p))

    # Make a half step for momentum at the end
    q = q + epsilon * p
    p = p - epsilon * U.gradient(q) / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(input.current_q)
    current_K = jnp.sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p ** 2) / 2
    # Accept or reject the state at the end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    # Compute log acceptance probability
    log_accept_prob = current_U - proposed_U + current_K - proposed_K
    log_accept_prob = jnp.where(jnp.isfinite(log_accept_prob), log_accept_prob,
                                -jnp.inf)  # handle non-finite values
    # Accept or reject the state at the end of trajectory
    key, subkey = random.split(key)
    accept = jnp.log(random.uniform(subkey)) < log_accept_prob
    q_new = lax.cond(accept, lambda _: q, lambda _: input.current_q, operand=None)
    return JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=q_new, key=key)