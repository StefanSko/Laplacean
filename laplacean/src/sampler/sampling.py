import jax
import jax.numpy as jnp
from jaxtyping import Array

from methods.hmc import StepFunc
from methods.potential_energy import PotentialEnergy
from base.data import JaxHMCData


class Sampler:

    def __call__(self, step: StepFunc, init: JaxHMCData, energy: PotentialEnergy, 
                 num_warmup: int = 500, num_samples: int = 1000) -> Array:
        # Warm-up phase
        def warmup_body(carry, _):
            input, key = carry
            output = step(energy, JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.current_q

        (input, _), _ = jax.lax.scan(warmup_body, (init, init.key), jnp.zeros(num_warmup))

        # Sampling phase
        def sampling_body(carry, _):
            input, key = carry
            output = step(energy, JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.current_q

        (_, key), samples = jax.lax.scan(sampling_body, (input, input.key), jnp.zeros(num_samples))

        return samples
