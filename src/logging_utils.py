import os
import logging
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
        return np.array(array)
    return array


def jax_debug_print(msg, **kwargs):
    """Log JAX debug messages."""

    def format_value(v):
        if isinstance(v, jnp.ndarray):
            return f"JAX array with shape {v.shape} and dtype {v.dtype}"
        return str(v)

    formatted_kwargs = {k: format_value(v) for k, v in kwargs.items()}
    logger.debug(msg.format(**formatted_kwargs))


# Custom JAX-friendly print function
def custom_jax_print(msg, **kwargs):
    formatted_msg = msg.format(**kwargs)
    logger.debug(formatted_msg)
    return None  # Return None to be compatible with JAX transformations