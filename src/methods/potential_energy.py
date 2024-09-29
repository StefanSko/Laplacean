from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Callable, Generic, NewType, Tuple, TypeVar

from util import conditional_jit

#we have unobserved Variables being the parameters and observed variables (data). Priors are beliefs about the parameters, 
# while The likelihood is the probability of the observed data given the parameters. 
# In Bayesian inference, the likelihood connects the observed variables to the unobserved parameters.


Parameter = NewType('Parameter', Array)

LikelihoodId = NewType('LikelihoodId', str)
LogProbability = NewType('LogProbability', Float[Array, ""])

class ObservableProvider(ABC):
    @abstractmethod
    def __call__(self) -> Array:
        pass


Variable = Parameter | ObservableProvider

T = TypeVar('T', bound=Parameter)
U = TypeVar('U', bound=ObservableProvider)

class IdentityObservableProvider(ObservableProvider):
    def __call__(self) -> Array:
        return jnp.array(0.0)

class DataObservableProvider(ObservableProvider):
    def __init__(self, data_fn: Callable[[], Array]):
        self._data_fn = data_fn

    def __call__(self) -> Array:
        return self._data_fn()

class LogDensity(Generic[T, U]):
    log_prob: Callable[[T,U], LogProbability]

    def __init__(self, log_prob: Callable[[T,U], LogProbability]):
        self.log_prob = log_prob

    def __call__(self, params: T, observable_provider: U) -> LogProbability:
        return self.log_prob(params, observable_provider)

class PriorLogDensity(LogDensity[T, IdentityObservableProvider]):
    def __init__(self, log_prob: Callable[[T, IdentityObservableProvider], LogProbability]):
        super().__init__(log_prob)

class LikelihoodLogDensity(LogDensity[T, U]):
    id: LikelihoodId

    def __init__(self, log_prob: Callable[[T, U], LogProbability], id: LikelihoodId):
        super().__init__(log_prob)
        self.id = id

def constant_log_density() -> PriorLogDensity[Parameter]:
    def log_prob(params: Parameter, _: IdentityObservableProvider) -> LogProbability:
        return LogProbability(jnp.array(0.0))
    return PriorLogDensity(log_prob)

def normal_log_density(
    mean: Callable[[Parameter], Array],
    std: Callable[[Parameter], Array]) -> PriorLogDensity[Parameter]:
    def log_prob(params: Parameter, _: IdentityObservableProvider) -> LogProbability:
        return LogProbability(jnp.sum(-0.5 * ((params - mean(params)) / std(params)) ** 2))
    return PriorLogDensity(log_prob)

def exponential_log_density(rate: Array = jnp.array(1.0)) -> PriorLogDensity[Parameter]:
    def log_prob(params: Parameter, _: IdentityObservableProvider) -> LogProbability:
        return LogProbability(jnp.sum(jnp.log(rate) - rate * params))
    return PriorLogDensity(log_prob)

def create_likelihood_log_density(
    distribution_log_prob: Callable[[Parameter, ObservableProvider], LogProbability]
) -> Callable[[Parameter, ObservableProvider], LogProbability]:
    def log_prob(params: Parameter, observable_provider: ObservableProvider) -> LogProbability:
        match observable_provider:
            case IdentityObservableProvider():
                return LogProbability(jnp.array(0.0))
            case _:
                return distribution_log_prob(params, observable_provider)
    return log_prob

def bind_data_to_likelihood(
    likelihood: LikelihoodLogDensity[Parameter, IdentityObservableProvider],
    data_provider: DataObservableProvider
) -> LogDensity[Parameter, DataObservableProvider]:
    def bound_log_prob(params: Parameter, _: DataObservableProvider) -> LogProbability:
        return likelihood.log_prob(params, data_provider)
    
    return LogDensity(bound_log_prob)

def likelihood_normal_log_density(
    mean: Callable[[Parameter, ObservableProvider], Array],
    std: Callable[[Parameter, ObservableProvider], Array],
    id: LikelihoodId
) -> LikelihoodLogDensity[Parameter, ObservableProvider]:
    def normal_log_prob(params: Parameter, observable_provider: ObservableProvider) -> LogProbability:
        y_pred = mean(params, observable_provider)
        std_value = std(params, observable_provider)
        return LogProbability(jnp.sum(-0.5 * ((observable_provider() - y_pred) / std_value) ** 2 - jnp.log(std_value) - 0.5 * jnp.log(2 * jnp.pi)))
    
    return LikelihoodLogDensity(create_likelihood_log_density(normal_log_prob), id)


#TODO: Fix bind; think about lazy evals and some execution plan 


class BayesianModel(eqx.Module, LogDensity[Parameter, ObservableProvider]):
    
    log_densities: Tuple[LogDensity, ...]
    log_prob: Callable[[Parameter, ObservableProvider], LogProbability]
    
    def __init__(self, log_densities: Tuple[LogDensity, ...]):
        self.log_densities = log_densities
        super().__init__(self._log_joint)

    @conditional_jit(use_jit=True)
    def _log_joint(self, params: Parameter, observable_provider: ObservableProvider) -> LogProbability:
        result = sum(d.log_prob(params, observable_provider) for d in self.log_densities)
        return LogProbability(jnp.array(result))

    @conditional_jit(use_jit=True)
    def potential_energy(self, params: Parameter, observable_provider: ObservableProvider) -> LogProbability:
        return LogProbability(-self._log_joint(params, observable_provider))

    @conditional_jit(use_jit=True)
    def gradient(self, params: Parameter, observable_provider: ObservableProvider) -> Array:
        return jax.grad(self.potential_energy)(params, observable_provider)
    
    def bind_data(self, data: dict[LikelihoodId, DataObservableProvider]) -> 'BayesianModel':
        new_log_densities = []
        for ld in self.log_densities:
            if isinstance(ld, LikelihoodLogDensity) and ld.id in data:
                new_ld = bind_data_to_likelihood(ld, data[ld.id])
                new_log_densities.append(new_ld)
            else:
                new_log_densities.append(ld)
        return BayesianModel(tuple(new_log_densities))





def sample_prior(model: BayesianModel, key, shape):
    # This is a placeholder. Actual implementation would depend on the specific priors.
    pass