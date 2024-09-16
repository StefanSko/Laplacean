
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Callable, Tuple, Optional

from util import conditional_jit

class LogDensity(eqx.Module):
    log_prob: Callable[[Array, Optional[dict]], Float[Array, ""]]

    def __init__(self, log_prob: Callable[[Array, Optional[dict]], Float[Array, ""]]):
        self.log_prob = log_prob
        
    def __call__(self, q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        return self.log_prob(q, data)

def constant_log_density() -> LogDensity:
    def log_prob(q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        return jnp.array(0.0)
    return LogDensity(log_prob) 

def normal_log_density(mean: Array, std: Array) -> LogDensity:
    def log_prob(q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        return jnp.sum(-0.5 * ((q - mean) / std) ** 2)

    return LogDensity(log_prob)

def exponential_log_density(rate: float = 1.0) -> LogDensity:
    def log_prob(q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        return jnp.sum(jnp.log(rate) - rate * q)
    return LogDensity(log_prob)

def parameterized_normal_log_density(
    mean: Callable[[dict], Array],
    std: Callable[[dict], Array]
) -> LogDensity:
    def log_prob(q: Array, data: Optional[dict] = None) -> Float[Array, ""]:
        if data is None:
            return jnp.array(0.0)  # Return 0 log probability as an array when no data is provided
        mean_value = mean(data)
        std_value = std(data)
        return jnp.sum(-0.5 * ((q - mean_value) / std_value) ** 2 - jnp.log(std_value) - 0.5 * jnp.log(2 * jnp.pi))
    return LogDensity(log_prob)

class BayesianModel(eqx.Module):
    log_densities: Tuple[LogDensity, ...]
    data: Optional[dict]

    def __init__(self, log_densities: Tuple[LogDensity, ...], data: Optional[dict] = None):
        self.log_densities = log_densities
        self.data = data

    @conditional_jit(use_jit=True)
    def log_joint(self, q: Array) -> Float[Array, ""]:
        result = sum(d.log_prob(q, self.data) for d in self.log_densities)
        return jnp.array(result)

    @conditional_jit(use_jit=True)
    def potential_energy(self, q: Array) -> Float[Array, ""]:
        return -self.log_joint(q)

    @conditional_jit(use_jit=True)
    def gradient(self, q: Array) -> Array:
        return jax.grad(self.potential_energy)(q)

def bind_data(model: BayesianModel, data: dict) -> BayesianModel:
    return BayesianModel(model.log_densities, data)

def sample_prior(model: BayesianModel, key, shape):
    # This is a placeholder. Actual implementation would depend on the specific priors.
    pass