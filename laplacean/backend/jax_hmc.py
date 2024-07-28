from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array
import jax_dataclasses as jdc


import jax.random as random

# Define type annotations for clarity
PotentialFn = Callable[[Array], float]
GradientFn = Callable[[Array], Array]

@jdc.pytree_dataclass
class JaxHMCData:
    epsilon: float
    L: int
    current_q: Array
    key: Array


class HMCProtocol:

    def hmc(self, input: JaxHMCData) -> JaxHMCData:  # type: ignore
        ...

    def run_hmc(self, input: JaxHMCData, num_samples: int, num_warmup: int) -> Array:  # type: ignore
        ...

class JaxHMC(HMCProtocol):

    def __init__(self, U: PotentialFn, grad_U: GradientFn):
        self.U = U
        self.grad_U = grad_U

    def hmc_step(self, input: JaxHMCData) -> JaxHMCData:
        q = input.current_q
        key, subkey = random.split(input.key)
        p = random.normal(subkey, q.shape)
        epsilon = input.epsilon
        L = input.L
        current_p = p
        # Make a half step for momentum at the beginning
        p = p - epsilon * self.grad_U(q) / 2

        def loop_body(_, carry):
            q, p = carry
            # Make a full step for the position
            q = q + epsilon * p
            # Make a full step for the momentum
            p = p - epsilon * self.grad_U(q)
            return q, p

        q, p = jax.lax.fori_loop(0, L-1, loop_body, (q, p))
        
        # Make a half step for momentum at the end
        q = q + epsilon * p
        p = p - epsilon * self.grad_U(q) / 2
        # Negate momentum at end of trajectory to make the proposal symmetric
        p = -p
        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = self.U(input.current_q)
        current_K = jnp.sum(current_p ** 2) / 2
        proposed_U = self.U(q)
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
        q_new = jax.lax.cond(accept, lambda _: q, lambda _: input.current_q, operand=None)
        return JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=q_new, key=key)

    def run_hmc(self, input: JaxHMCData, num_samples: int, num_warmup: int) -> Array:
        # Warm-up phase
        def warmup_body(carry, _):
            input, key = carry
            output = self.hmc_step(JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.current_q

        (input, _), _ = jax.lax.scan(warmup_body, (input, input.key), jnp.zeros(num_warmup))

        # Sampling phase
        def sampling_body(carry, _):
            input, key = carry
            output = self.hmc_step(JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.current_q

        (_, key), samples = jax.lax.scan(sampling_body, (input, input.key), jnp.zeros(num_samples))

        return samples
