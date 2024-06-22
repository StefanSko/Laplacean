

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import jax.random as random
from laplacean.backend.base import HMC, PotentialFn, GradientFn


class JaxHMC(HMC):

    def __init__(self, random_key: jax.random.PRNGKey) -> None:
        super().__init__()
        self.key = random_key

    def hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, current_q: Array) -> Array:
        q = jnp.array(current_q)
        self.key, subkey = random.split(self.key)
        p = random.normal(subkey, q.shape)

        current_p = p
        # Make a half step for momentum at the beginning
        p = p - epsilon * grad_U(q) / 2

        for i in range(L):
            # Make a full step for the position
            q = q + epsilon * p
            # Make a full step for the momentum, except at the end of trajectory
            if i != L - 1:
                p = p - epsilon * grad_U(q)
        
        # Make a half step for momentum at the end
        p = p - epsilon * grad_U(q) / 2
        # Negate momentum at end of trajectory to make the proposal symmetric
        p = -p
        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = U(current_q)
        current_K = jnp.sum(current_p ** 2) / 2
        proposed_U = U(q)
        proposed_K = jnp.sum(p ** 2) / 2
        # Accept or reject the state at the end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)
        key, subkey = random.split(key)
        accept = random.uniform(subkey) < accept_prob
        q_new = jax.lax.cond(accept, lambda _: q, lambda _: current_q, operand=None)
        return q_new
    

    def run_hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, initial_q: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        def body_fun(carry):
            q = carry
            q = self.hmc(U, grad_U, epsilon, L, q)
            return q, q
        _, samples = jax.lax.scan(body_fun, initial_q, jnp.arange(num_samples))
        return samples
