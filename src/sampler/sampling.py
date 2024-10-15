from typing import Callable
import jax
import jax.numpy as jnp
from jax import debug
from jaxtyping import Array

from base.data import JaxHMCData

import equinox as eqx

from logging_utils import logger, sample_step_print, warmup_step_print
from methods.bayesian_execution_network import BayesianExecutionModel
from util import conditional_jit


class Sampler(eqx.Module):

    @conditional_jit(use_jit=True)
    def __call__(self, step: Callable, init: JaxHMCData, energy: BayesianExecutionModel,
                 num_warmup: int = 500, num_samples: int = 1000) -> Array:
        logger.info(f"Starting sampling with num_warmup={num_warmup}, num_samples={num_samples}")

        # Warm-up phase
        def warmup():
            def warmup_body(carry, i):
                input = carry
                output = step(energy, input)
                debug.callback(warmup_step_print, i=i, q=output.current_q, key=output.key)
                return output, output.current_q

            final_state, _ = jax.lax.scan(warmup_body, init, jnp.arange(num_warmup))
            return final_state

        input = warmup()

        # Sampling phase
        def sampling():
            def sampling_body(carry, i):
                input = carry
                output = step(energy, input)
                debug.callback(sample_step_print, i=i, q=output.current_q, key=output.key)
                return output, output.current_q

            _, samples = jax.lax.scan(sampling_body, input, jnp.arange(num_samples))
            return samples

        samples = sampling()
        logger.debug(f"Final samples shape: {samples.shape}")
        logger.debug(f"First 5 samples: {samples[:5]}")
        logger.debug(f"Last 5 samples: {samples[-5:]}")

        logger.info("Sampling completed")
        
        return samples
