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
    return array.tolist() if isinstance(array, jnp.ndarray) else array

def jax_debug_print(msg, **kwargs):
    """Log JAX debug messages."""
    formatted_kwargs = {k: log_jax_array(v) for k, v in kwargs.items()}
    logger.debug(msg.format(**formatted_kwargs))