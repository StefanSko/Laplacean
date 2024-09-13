import os
import logging
from functools import partial
from typing import Any

from jax import numpy as jnp

# Set up logger
logger = logging.getLogger('hmc_sampler')
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() == 'true' else logging.INFO)

# Create file handler
file_handler = logging.FileHandler('hmc_sampler.log')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_jax_array(array):
    """Convert JAX array to a loggable format."""
    if isinstance(array, jnp.ndarray):
        return f"JAX array with shape {array.shape} and dtype {array.dtype}"
    return array

def custom_jax_print(msg: str, **kwargs: Any) -> None:
    """Custom print function for JAX debugging."""
    formatted_kwargs = {k: log_jax_array(v) for k, v in kwargs.items()}
    formatted_msg = msg.format(**formatted_kwargs)
    logger.debug(formatted_msg)
    return None  # Return None to be compatible with JAX transformations

# Partial functions for different log messages
step_input_print = partial(custom_jax_print, "Step input: q = {q}, key = {key}")
generated_momentum_print = partial(custom_jax_print, "Generated momentum: p = {p}")
after_leapfrog_print = partial(custom_jax_print, "After leapfrog: q_new = {q_new}, p_new = {p_new}")
hamiltonians_print = partial(custom_jax_print, "Hamiltonians: current = {current}, proposed = {proposed}")
acceptance_print = partial(custom_jax_print, "Acceptance: {accept}")
step_output_print = partial(custom_jax_print, "Step output: q_new = {q_new}, key = {key}")
warmup_step_print = partial(custom_jax_print, "Warmup step {i}: q = {q}, key = {key}")
sample_step_print = partial(custom_jax_print, "Sample step {i}: q = {q}, key = {key}")