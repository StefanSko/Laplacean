from typing import Callable, Generic, Union, Protocol
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.stats import norm
import jax

from typing import TypeVar

from base.data import RandomVar, Index

from dataclasses import dataclass

LogProb = Float[Array, ""]

class Var:
    """Base class for variables in the Bayesian network"""
    value: Array | float | None = None
    
    def __init__(self, value: Array | float | None = None) -> None:
        self.value = value

class Parameter(Var):
    """Parameter variable type"""
    pass

class Data(Var):
    """Data variable type"""
    pass

V = TypeVar('V', bound=Var)

class Distribution(Generic[V]):
    log_prob: Callable[[V], LogProb]

    def __init__(self, log_prob: Callable[[V], LogProb]):
        self.log_prob = log_prob
    
    def __call__(self, var: V) -> LogProb:
        return self.log_prob(var)

class Prior(Distribution[Parameter]):
    """A distribution over parameters (prior distribution)"""
    def __init__(self, log_prob: Callable[[Parameter], LogProb]):
        super().__init__(log_prob)

class Likelihood:
    """A function that produces a distribution over data given parameters"""
    def __init__(self, dist_fn: Callable[[Parameter], Distribution[Data]]):
        self._dist_fn = dist_fn
    
    def __call__(self, param: Parameter) -> Distribution[Data]:
        return self._dist_fn(param)

class DistSupplier(Protocol):
    """Protocol defining how distributions are supplied to edges"""
    def log_prob(self, var: RandomVar, params: Array) -> LogProb:
        ...

class PriorSupplier(DistSupplier):
    """Supplies prior distributions for parameters"""
    distribution: Prior
    
    def __init__(self, distribution: Prior):
        self.distribution = distribution

    def log_prob(self, var: RandomVar, params: Array) -> LogProb:
        param_value = Parameter(var.get_value(params))
        return self.distribution(param_value)

class LikelihoodSupplier(DistSupplier):
    """Supplies likelihood distributions for data"""
    likelihood: Likelihood
    data: Data
    
    def __init__(self, likelihood: Likelihood, data: Data):
        self.likelihood = likelihood
        self.data = data
        
    def log_prob(self, var: RandomVar, params: Array) -> LogProb:
        parameter = Parameter(var.get_value(params))
        return self.likelihood(parameter).log_prob(self.data)
        

def normal(
        loc: Parameter, 
        scale: Parameter
) -> Distribution[Var]:
    def log_prob(var: Var) -> LogProb:
        loc_value = jnp.array(loc.value)
        scale_value = jnp.array(scale.value)
        
        if var.value is None:
            return jnp.array(0.0)
        return jnp.sum(norm.logpdf(var.value, loc_value, scale_value))
            
    return Distribution(log_prob)



@dataclass(frozen=True)
class Edge:
    """Probabilistic relationship between variables"""
    child: RandomVar
    dist_supplier: DistSupplier
    name: str = "unnamed_edge"

    def log_prob(self, param_values: Array) -> LogProb:
        """
        Compute log probability using parameter values
        
        Args:
            param_values: Dictionary mapping parameter names to their values
        """
        return self.dist_supplier.log_prob(self.child, param_values)
    
    def __repr__(self) -> str:
        return f"Edge({self.name})"



class BayesianNetwork:
    """Complete probabilistic graphical model"""
    variables: dict[str, RandomVar]
    edges: list[Edge]
    param_size: int
    param_index: Index

    def __init__(self, variables: dict, edges: list[Edge], param_size: int, param_index: Index):
        self.variables = variables
        self.edges = edges
        self.param_size = param_size
        self.param_index = param_index

    def log_prob(self, params: Array) -> LogProb:
        """Compute total log probability of model"""
        return jnp.sum(jnp.array([edge.log_prob(params) for edge in self.edges]))

    def potential_energy(self, params: Array) -> LogProb:
        """Compute potential energy (negative log probability)"""
        return -self.log_prob(params)

    def gradient(self, params: Array) -> Array:
        """
        Compute gradient of potential energy with respect to parameters.
        Uses JAX automatic differentiation to trace through the edges.
        """
        return jax.grad(self.potential_energy)(params)


class ModelBuilder:
    """Stan-like model builder interface"""

    variables: dict[str, RandomVar]
    edges: list[Edge]
    _size_vars: dict[str, int]
    _current_param_idx: int

    def __init__(self):
        self.variables = {}
        self.edges = []
        self._size_vars = {}
        self._current_param_idx = 0

    def data(self) -> 'DataBlockBuilder':
        return DataBlockBuilder(self)

    def parameters(self) -> 'ParameterBlockBuilder':
        return ParameterBlockBuilder(self)

    def model(self) -> 'ModelBlockBuilder':
        return ModelBlockBuilder(self)

    def build(self) -> BayesianNetwork:
        return BayesianNetwork(
            variables=self.variables,
            edges=self.edges,
            param_size=self._current_param_idx
        )


class DataBlockBuilder:
    """Builder for data block declarations"""

    def __init__(self, model_builder: ModelBuilder):
        self.model_builder = model_builder

    def int_scalar(self, name: str, value: Array) -> 'DataBlockBuilder':
        """Declare an integer scalar (usually for sizes)"""
        self.model_builder._size_vars[name] = int(value)
        self.model_builder.variables[name] = ObservedVariable(
            name=name,
            shape=(),
            provider=lambda _: value
        )
        return self

    def vector(self, name: str, size: str) -> 'DataBlockBuilder':
        """Declare a vector of observations"""
        vector_size = self.model_builder._size_vars[size]

        self.model_builder.variables[name] = ObservedVariable(
            name=name,
            shape=(vector_size,),
            provider=lambda x: x  # This will be replaced when data is bound
        )
        return self

    def done(self) -> ModelBuilder:
        return self.model_builder


class ParameterBlockBuilder:
    """Builder for parameter declarations"""

    def __init__(self, model_builder: ModelBuilder):
        self.model_builder = model_builder

    def real(self, name: str) -> 'ParameterBlockBuilder':
        """Declare a real-valued scalar parameter"""
        param_index = Index.single(self.model_builder._current_param_idx)
        self.model_builder._current_param_idx += 1

        self.model_builder.variables[name] = Parameter(
            name=name,
            shape=(),
            provider=param_index
        )
        return self

    def vector(self, name: str, size: str) -> 'ParameterBlockBuilder':
        """Declare a vector of parameters"""
        vector_size = self.model_builder._size_vars[size]
        param_index = Index.vector(
            self.model_builder._current_param_idx,
            self.model_builder._current_param_idx + vector_size
        )
        self.model_builder._current_param_idx += vector_size

        self.model_builder.variables[name] = Parameter(
            name=name,
            shape=(vector_size,),
            provider=param_index
        )
        return self

    def done(self) -> ModelBuilder:
        return self.model_builder


class ModelBlockBuilder:
    """Builder for model relationships"""

    def __init__(self, model_builder: ModelBuilder):
        self.model_builder = model_builder

    def normal(
            self,
            target: str,
            loc: Union[str, float],
            scale: Union[str, float]
    ) -> 'ModelBlockBuilder':
        """Add normal distribution relationship"""
        target_var = self.model_builder.variables[target]

        # Handle location parameter
        if isinstance(loc, str):
            loc_var = self.model_builder.variables[loc]
        else:
            loc_var = Parameter(
                name=f"{target}_loc",
                shape=(),
                provider=lambda _: jnp.array(loc)
            )

        # Handle scale parameter
        if isinstance(scale, str):
            scale_var = self.model_builder.variables[scale]
        else:
            scale_var = Parameter(
                name=f"{target}_scale",
                shape=(),
                provider=lambda _: jnp.array(scale)
            )

        self.model_builder.edges.append(Edge(
            child=target_var,
            distribution=Distribution.normal(loc_var, scale_var),
            name=f"normal_{target}"
        ))
        return self

    def exponential(
            self,
            target: str,
            rate: float
    ) -> 'ModelBlockBuilder':
        """Add exponential distribution relationship"""
        target_var = self.model_builder.variables[target]

        self.model_builder.edges.append(Edge(
            child=target_var,
            distribution=Distribution.exponential(rate),
            name=f"exponential_{target}"
        ))
        return self

    def done(self) -> BayesianNetwork:
        """Finalize the model and return the BayesianNetwork"""
        return self.model_builder.build()


def bind_data(
        model: BayesianNetwork,
        data: dict[str, Array]
) -> BayesianNetwork:
    """
    Bind data to model, handling missing values transparently.
    Returns a new model instance with updated variables.
    """
    new_variables = dict(model.variables)

    for name, value in data.items():
        if name in new_variables:
            var = new_variables[name]
            new_variables[name] = ObservedVariable(
                name=var.name,
                shape=var.shape,
                provider=lambda _: value
            )

    return BayesianNetwork(
        variables=new_variables,
        edges=model.edges,
        param_size=model.param_size
    )


#TODO: FIXME!! Parameters should not trasnfer to data part if not explicitly desired
def get_initial_params(
        model: BayesianNetwork,
        random_key: jax.random.PRNGKey,
) -> Array:
    """
    Generate initial parameters for the model.
    Handles both regular parameters and missing value parameters.
    """
    # Split key for different random number generations
    key1, key2 = jax.random.split(random_key)

    # Initialize parameters array
    params = jnp.zeros(model.param_size)

    # Helper function to initialize a variable's parameters
    def init_variable_params(
            var: RandomVar,
            key: jax.random.PRNGKey
    ) -> Array:
        if var.param_index is None:
            return params

        # Get the slice for this variable's parameters
        idx = var.param_index.indices[0]
        size = idx.stop - idx.start if idx.stop else 1

        # Generate random initial values
        if var.has_missing():
            # For missing values, initialize near observed data mean/std
            observed = var.observed_values
            observed_mask = ~jnp.isnan(observed)
            if jnp.any(observed_mask):
                mean = jnp.mean(observed[observed_mask])
                std = jnp.std(observed[observed_mask]) + 1e-6
            else:
                mean, std = 0., 1.
            values = mean + std * jax.random.normal(key, (size,))
        else:
            # For parameters, use standard normal initialization
            values = jax.random.normal(key, (size,))

        return params.at[idx].set(values)

    # Initialize all variables
    for var in model.variables.values():
        if var.param_index is not None:
            key1, key2 = jax.random.split(key1)
            params = init_variable_params(var, key2)

    return params
