
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Callable, Tuple, NewType

from util import conditional_jit

class LogDensity(eqx.Module):
    log_prob: Callable[[Array], Float[Array, ""]]

    def __init__(self, log_prob: Callable[[Array], Float[Array, ""]]):
        self.log_prob = log_prob
        
    def __call__(self, q: Array) -> Float[Array, ""]:
        return self.log_prob(q)

def constant_log_density() -> LogDensity:
    def log_prob(q: Array) -> Float[Array, ""]:
        return jnp.array(0.0)
    return LogDensity(log_prob) 

def normal_log_density(
    mean: Callable[[Array], Array],
    std: Callable[[Array], Array]) -> LogDensity:
    def log_prob(q: Array) -> Float[Array, ""]:
        return jnp.sum(-0.5 * ((q - mean(q)) / std(q)) ** 2)

    return LogDensity(log_prob)

def exponential_log_density(rate: Array = jnp.array(1.0)) -> LogDensity:
    def log_prob(q: Array) -> Float[Array, ""]:
        return jnp.sum(jnp.log(rate) - rate * q)
    return LogDensity(log_prob)

def parameterized_normal_log_density(
    mean: Callable[[Array], Array],
    std: Callable[[Array], Array]
) -> LogDensity:
    def log_prob(q: Array) -> Float[Array, ""]:
        y_pred = mean(q)
        std_value = std(q)
        return jnp.sum(-0.5 * ((q - y_pred) / std_value) ** 2 - jnp.log(std_value) - 0.5 * jnp.log(2 * jnp.pi))
    return LogDensity(log_prob)



class ObservableProvider:
    def __init__(self, data_fn: Callable[[], Array]):
        self.data_fn = data_fn

    def __call__(self) -> Array:
        return self.data_fn()

NeutralObservableProvider = NewType('NeutralObservableProvider', ObservableProvider)
neutral_observable_provider = NeutralObservableProvider(ObservableProvider(lambda: jnp.array(0.0)))

ZeroLogDensity = NewType('ZeroLogDensity', LogDensity)
zero_log_density = ZeroLogDensity(LogDensity(lambda q: jnp.array(0.0)))

class Likeihood(LogDensity):
    def __init__(self, log_density: LogDensity, observable_provider: ObservableProvider):
        self.log_density = log_density
        self.observable_provider = observable_provider
        super().__init__(self.log_prob)
    
    def log_prob(self, q: Array) -> Float[Array, ""]:
        match self.log_density:
            case ZeroLogDensity():
                return jnp.array(0.0)
            case _:
                return self.log_density.log_prob(q)


class LikelihoodBuilder


class NormalLogDensity(LogDensity):

    mean: Callable[[Array], Array]
    std: Callable[[Array], Array]
    observable_provider: ObservableProvider

    def __init__(self, mean: Callable[[Array], Array], std: Callable[[Array], Array], observable_provider: ObservableProvider):
        self.mean = mean
        self.std = std  
        self.observable_provider = observable_provider
        super().__init__(self.log_prob)
    
    @classmethod
    def normal_likelihood(cls, mean: Callable[[Array], Array], std: Callable[[Array], Array]) -> 'NormalLogDensity':
        instance = cls(mean, std, neutral_observable_provider)
        return instance

    def log_prob(self, q: Array) -> Float[Array, ""]:
        y_pred = self.mean(q)
        std_value = self.std(q)
        return jnp.sum(-0.5 * ((q - y_pred) / std_value) ** 2 - jnp.log(std_value) - 0.5 * jnp.log(2 * jnp.pi)) 

class BayesianModel(eqx.Module):
    log_densities: Tuple[LogDensity, ...]
    def __init__(self, log_densities: Tuple[LogDensity, ...]):
        self.log_densities = log_densities

    @conditional_jit(use_jit=True)
    def log_joint(self, q: Array) -> Float[Array, ""]:
        result = sum(d.log_prob(q) for d in self.log_densities)
        return jnp.array(result)

    @conditional_jit(use_jit=True)
    def potential_energy(self, q: Array) -> Float[Array, ""]:
        return -self.log_joint(q)

    @conditional_jit(use_jit=True)
    def gradient(self, q: Array) -> Array:
        return jax.grad(self.potential_energy)(q)

def bind_data(model: BayesianModel, data: dict) -> BayesianModel:
    return BayesianModel(model.log_densities)

def sample_prior(model: BayesianModel, key, shape):
    # This is a placeholder. Actual implementation would depend on the specific priors.
    pass