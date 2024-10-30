from typing import Callable, Generic, TypeVar, Union, Optional
import equinox as eqx
import jax_dataclasses as jdc
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.stats import norm, expon
import jax

# Core type definitions
RandomVar = TypeVar('RandomVar')
LogDensity = Float[Array, ""]

@jdc.pytree_dataclass
class ParamIndex:
    """Unified parameter indexing"""
    indices: tuple[slice, ...] = (slice(None),)
    
    @staticmethod
    def single(i: int) -> 'ParamIndex':
        return ParamIndex((slice(i, i+1),))
    
    @staticmethod
    def vector(start: int, end: int | None = None) -> 'ParamIndex':
        return ParamIndex((slice(start, end),))
    
    def select(self, params: Array) -> Array:
        if len(self.indices) == 1:
            return params[self.indices[0]]
        return params[self.indices]

class RandomVariable(eqx.Module, Generic[RandomVar]):
    """Unified representation of random variables (both parameters and data)"""
    name: str
    shape: tuple[int, ...]
    observed_values: Optional[Array] = None
    param_index: Optional[ParamIndex] = None

    def get_value(self, params: Array) -> Array:
        """Get current value, combining observed and parameter values"""
        # Handle case where we have no parameters (fully observed)
        if self.param_index is None:
            return self.observed_values
        
        # Get parameter values
        param_values = self.param_index.select(params)
        
        # If no observed values, return parameters
        if self.observed_values is None:
            return param_values
        
        # Handle partially observed values using JAX operations
        mask = ~jnp.isnan(self.observed_values)
        return jnp.where(mask, self.observed_values, param_values)

    def is_observed(self) -> bool:
        return self.observed_values is not None

    def has_missing(self) -> bool:
        return (self.observed_values is not None and 
                jnp.any(jnp.isnan(self.observed_values)))

class Distribution(eqx.Module, Generic[RandomVar]):
    """Probability distribution over random variables"""
    log_prob: Callable[[RandomVariable[RandomVar], Array], LogDensity]

    @staticmethod
    def normal(
        loc: Union[RandomVariable[RandomVar], float],
        scale: Union[RandomVariable[RandomVar], float]
    ) -> 'Distribution[RandomVar]':
        def log_prob(rv: RandomVariable[RandomVar], params: Array) -> LogDensity:
            value = rv.get_value(params)
            loc_value = (loc.get_value(params) if isinstance(loc, RandomVariable) 
                        else jnp.array(loc))
            scale_value = (scale.get_value(params) if isinstance(scale, RandomVariable) 
                          else jnp.array(scale))
            return jnp.sum(norm.logpdf(value, loc_value, scale_value))
        return Distribution(log_prob)

    @staticmethod
    def exponential(rate: float) -> 'Distribution[RandomVar]':
        def log_prob(rv: RandomVariable[RandomVar], params: Array) -> LogDensity:
            value = rv.get_value(params)
            return jnp.sum(expon.logpdf(value, scale=1/rate))
        return Distribution(log_prob)

class Edge(eqx.Module, Generic[RandomVar]):
    """Probabilistic relationship between variables"""
    child: RandomVariable[RandomVar]
    distribution: Distribution[RandomVar]

    def log_prob(self, params: Array) -> LogDensity:
        return self.distribution.log_prob(self.child, params)

class BayesianNetwork(eqx.Module, Generic[RandomVar]):
    """Complete probabilistic graphical model"""
    variables: dict[str, RandomVariable[RandomVar]]
    edges: list[Edge[RandomVar]]
    param_size: int

    def log_prob(self, params: Array) -> LogDensity:
        """Compute total log probability of model"""
        return jnp.sum(jnp.array([edge.log_prob(params) for edge in self.edges]))

    def potential_energy(self, params: Array) -> LogDensity:
        """Compute potential energy (negative log probability)"""
        return -self.log_prob(params)

    def gradient(self, params: Array) -> Array:
        """
        Compute gradient of potential energy with respect to parameters.
        Uses JAX automatic differentiation.
        """
        return jax.grad(self.potential_energy)(params)

class ModelBuilder:
    """Stan-like model builder interface"""
    def __init__(self):
        self.variables: dict[str, RandomVariable[RandomVar]] = {}
        self.edges: list[Edge[RandomVar]] = []
        self._size_vars: dict[str, int] = {}
        self._current_param_idx: int = 0

    def data(self) -> 'DataBlockBuilder':
        return DataBlockBuilder(self)

    def parameters(self) -> 'ParameterBlockBuilder':
        return ParameterBlockBuilder(self)

    def model(self) -> 'ModelBlockBuilder':
        return ModelBlockBuilder(self)

    def build(self) -> BayesianNetwork[RandomVar]:
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
        self.model_builder.variables[name] = RandomVariable(
            name=name,
            shape=(),
            observed_values=value,
            param_index=None  # Fully observed, no parameters needed
        )
        return self
    
    def vector(self, name: str, size: str) -> 'DataBlockBuilder':
        """Declare a vector of observations"""
        vector_size = self.model_builder._size_vars[size]
        # Initially no observed values - will be bound later
        # But we allocate parameter space for potential missing values
        param_index = ParamIndex.vector(
            self.model_builder._current_param_idx,
            self.model_builder._current_param_idx + vector_size
        )
        self.model_builder._current_param_idx += vector_size
        
        self.model_builder.variables[name] = RandomVariable(
            name=name,
            shape=(vector_size,),
            observed_values=None,  # Will be set when data is bound
            param_index=param_index  # For potential missing values
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
        param_index = ParamIndex.single(self.model_builder._current_param_idx)
        self.model_builder._current_param_idx += 1
        
        self.model_builder.variables[name] = RandomVariable(
            name=name,
            shape=(),
            observed_values=None,  # Unobserved (parameter)
            param_index=param_index
        )
        return self
    
    def vector(self, name: str, size: str) -> 'ParameterBlockBuilder':
        """Declare a vector of parameters"""
        vector_size = self.model_builder._size_vars[size]
        param_index = ParamIndex.vector(
            self.model_builder._current_param_idx,
            self.model_builder._current_param_idx + vector_size
        )
        self.model_builder._current_param_idx += vector_size
        
        self.model_builder.variables[name] = RandomVariable(
            name=name,
            shape=(vector_size,),
            observed_values=None,  # Unobserved (parameter)
            param_index=param_index
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
            loc_var = RandomVariable(
                name=f"{target}_loc",
                shape=(),
                observed_values=jnp.array(loc),
                param_index=None
            )
            
        # Handle scale parameter
        if isinstance(scale, str):
            scale_var = self.model_builder.variables[scale]
        else:
            scale_var = RandomVariable(
                name=f"{target}_scale",
                shape=(),
                observed_values=jnp.array(scale),
                param_index=None
            )
        
        self.model_builder.edges.append(Edge(
            child=target_var,
            distribution=Distribution.normal(loc_var, scale_var)
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
            distribution=Distribution.exponential(rate)
        ))
        return self
    
    def done(self) -> BayesianNetwork[RandomVar]:
        """Finalize the model and return the BayesianNetwork"""
        return self.model_builder.build()


def bind_data(
    model: BayesianNetwork[RandomVar], 
    data: dict[str, Array]
) -> BayesianNetwork[RandomVar]:
    """
    Bind data to model, handling missing values transparently.
    Returns a new model instance with updated variables.
    """
    new_variables = dict(model.variables)
    
    for name, value in data.items():
        if name in new_variables:
            var = new_variables[name]
            # Create new variable with observed values but keep other attributes
            new_variables[name] = RandomVariable(
                name=var.name,
                shape=var.shape,
                observed_values=value,
                param_index=var.param_index
            )
    
    return BayesianNetwork(
        variables=new_variables,
        edges=model.edges,
        param_size=model.param_size
    )

def get_initial_params(
    model: BayesianNetwork[RandomVar],
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
        var: RandomVariable[RandomVar], 
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





