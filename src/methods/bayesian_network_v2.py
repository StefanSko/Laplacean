from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Any, cast
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.stats import norm, expon

# Type definitions
Parameter = Array
Data = Array
LogDensity = Float[Array, ""]

Variable = Parameter | Data

U = TypeVar('U', bound=Parameter)
V = TypeVar('V', bound=Data)
T = TypeVar('T', bound=Variable)


@dataclass(frozen=True)
class ParamIndex:
    """Unified parameter indexing"""
    indices: tuple[slice, ...] = (slice(None),)
    
    @staticmethod
    def single(i: int) -> 'ParamIndex':
        return ParamIndex((slice(i, i+1),))
    
    @staticmethod
    def vector(start: int, end: int | None = None) -> 'ParamIndex':
        return ParamIndex((slice(start, end),))
    
    def select(self, params: U) -> U:
        if len(self.indices) == 1:
            return cast(U, params[self.indices[0]])
        return cast(U, params[self.indices])


class BayesVar(eqx.Module, Generic[T]):
    """Represents a value that may or may not be present"""
    value: T

    def __init__(self, value: T):
        self.value = value

    def get(self) -> Any:
        return self.value

    def map(self, f: Callable[[T], Any]) -> Any:
        return jax.lax.cond(
            jnp.any(jnp.isnan(self.value)),
            lambda _: jnp.array(0.0),
            lambda x: f(x),
            self.value
        )

    @classmethod
    def empty(cls) -> 'BayesVar[T]':
        return cls(cast(T, jnp.array(float('nan'))))

    @classmethod
    def just(cls, value: T) -> 'BayesVar[T]':
        return cls(value)


@dataclass
class DataSpec:
    """Specification for data variables"""
    name: str
    shape: tuple[int, ...]
    value: BayesVar[Array] = BayesVar.empty()

    @classmethod
    def vector(cls, name: str, size: int) -> 'DataSpec':
        return cls(name, (size,))
    
    @classmethod
    def scalar(cls, name: str) -> 'DataSpec':
        return cls(name, ())


@dataclass
class ParameterSpec:
    """Specification for parameter variables"""
    name: str
    shape: tuple[int, ...]
    index: ParamIndex

    @classmethod
    def vector(cls, name: str, size: int, start_idx: int) -> 'ParameterSpec':
        return cls(name, (size,), ParamIndex.vector(start_idx, start_idx + size))
    
    @classmethod
    def scalar(cls, name: str, idx: int) -> 'ParameterSpec':
        return cls(name, (), ParamIndex.single(idx))


class Distribution(Generic[U, V]):
    """Represents a probability distribution with log probability function"""
    def __init__(self, log_prob: Callable[[U, BayesVar[V]], LogDensity]):
        self.log_prob = log_prob
    
    @staticmethod
    def normal(
        mean: Callable[[U], Array] | float,
        std: Callable[[U], Array] | float
    ) -> 'Distribution[U, V]':
        """Create a normal distribution that works for both priors and likelihoods"""
        # Convert constants to callables for unified handling
        mean_fn = mean if callable(mean) else (lambda _: jnp.array(mean))
        std_fn = std if callable(std) else (lambda _: jnp.array(std))
        
        return Distribution(
            lambda params, data: data.map(
                lambda d: jnp.sum(norm.logpdf(d, mean_fn(params), std_fn(params)))
            ) if not data.map(jnp.any) else jnp.sum(norm.logpdf(params, mean_fn(params), std_fn(params)))
        )
    
    @staticmethod
    def exponential(
        rate: Callable[[U], Array] | float
    ) -> 'Distribution[U, V]':
        """Create an exponential distribution that works for both priors and likelihoods"""
        rate_fn = rate if callable(rate) else (lambda _: jnp.array(rate))
        
        return Distribution(
            lambda params, data: data.map(
                lambda d: jnp.sum(expon.logpdf(d, scale=1/rate_fn(params)))
            ) if not data.map(jnp.any) else jnp.sum(expon.logpdf(params, scale=1/rate_fn(params)))
        )
    
    def map(self, f: Callable[[U], U]) -> 'Distribution[U, V]':
        return Distribution(lambda x, d: self.log_prob(f(x), d))


class BayesNode(eqx.Module, Generic[U, V]):
    """Unified node structure for both prior and likelihood nodes"""
    node_id: int
    param_index: ParamIndex
    log_density: Callable[[U, BayesVar[V]], LogDensity]
    data: BayesVar[V]

    def __init__(
        self, 
        node_id: int,
        param_index: ParamIndex,
        log_density: Callable[[U, BayesVar[V]], LogDensity],
        data: V | None = None
    ):
        self.node_id = node_id
        self.param_index = param_index
        self.log_density = log_density
        self.data = BayesVar.just(data) if data is not None else BayesVar.empty()

    def evaluate(self, params: U) -> LogDensity:
        selected_params = self.param_index.select(params)
        log_pdf = self.log_density(selected_params, self.data)
        jax.debug.print("node_id -> {}; logpdf -> {}", self.node_id, log_pdf)
        return log_pdf


class Model(eqx.Module, Generic[U, V]):
    """Main model class"""
    nodes: list[BayesNode[U, V]]
    data_vars: dict[str, DataSpec]
    param_vars: dict[str, ParameterSpec]
    
    def __init__(
        self, 
        nodes: list[BayesNode[U, V]], 
        data_vars: dict[str, DataSpec],
        param_vars: dict[str, ParameterSpec]
    ):
        self.nodes = nodes
        self.data_vars = data_vars
        self.param_vars = param_vars
    
    def __call__(self, params: U) -> LogDensity:
        return jnp.sum(jnp.array([node.evaluate(params) for node in self.nodes]))
    
    def potential_energy(self, params: U) -> LogDensity:
        return -self(params)
    
    def gradient(self, params: U) -> Array:
        return jax.grad(self.potential_energy)(params)
    
    def get_param_size(self) -> int:
        """Returns the total size of parameters needed"""
        return max(
            spec.index.indices[0].stop 
            for spec in self.param_vars.values()
            if spec.index.indices[0].stop is not None
        )


class ModelBuilder(Generic[U, V]):
    """Stan-like model builder"""
    def __init__(self) -> None:
        #TODO: FIX data vars spec
        self.data_vars: dict[str, DataSpec] = {}
        self.param_vars: dict[str, ParameterSpec] = {}
        self.nodes: list[BayesNode[U, V]] = []
        self._current_param_idx: int = 0
    
    def data(self) -> 'DataBlockBuilder[U, V]':
        """Start data block definition"""
        return DataBlockBuilder(self)
    
    def parameters(self) -> 'ParameterBlockBuilder[U, V]':
        """Start parameter block definition"""
        return ParameterBlockBuilder(self)
    
    def model(self) -> 'ModelBlockBuilder[U, V]':
        """Start model block definition"""
        return ModelBlockBuilder(self)
    
    def build(self) -> Model[U, V]:
        return Model(self.nodes, self.data_vars, self.param_vars)


class DataBlockBuilder(Generic[U, V]):
    """Builder for data block"""
    def __init__(self, model_builder: ModelBuilder[U, V]):
        self.model_builder = model_builder
    
    def int_scalar(self, name: str) -> 'DataBlockBuilder[U, V]':
        self.model_builder.data_vars[name] = DataSpec.scalar(name)
        return self
    
    def vector(self, name: str, size: str | int) -> 'DataBlockBuilder[U, V]':
        if isinstance(size, str):
            size = self.model_builder.data_vars[size].value.get()
        self.model_builder.data_vars[name] = DataSpec.vector(name, cast(int, size))
        return self
    
    def done(self) -> ModelBuilder[U, V]:
        return self.model_builder


class ParameterBlockBuilder(Generic[U, V]):
    """Builder for parameter block"""
    def __init__(self, model_builder: ModelBuilder[U, V]):
        self.model_builder = model_builder
    
    def real(self, name: str) -> 'ParameterBlockBuilder[U, V]':
        spec = ParameterSpec.scalar(name, self.model_builder._current_param_idx)
        self.model_builder.param_vars[name] = spec
        self.model_builder._current_param_idx += 1
        return self
    
    def vector(self, name: str, size: str | int) -> 'ParameterBlockBuilder[U, V]':
        if isinstance(size, str):
            size = self.model_builder.data_vars[size].value.get()
        spec = ParameterSpec.vector(name, cast(int, size), self.model_builder._current_param_idx)
        self.model_builder.param_vars[name] = spec
        self.model_builder._current_param_idx += cast(int, size)
        return self
    
    def done(self) -> ModelBuilder[U, V]:
        return self.model_builder


class ModelBlockBuilder(Generic[U, V]):
    """Builder for model block"""
    def __init__(self, model_builder: ModelBuilder[U, V]):
        self.model_builder = model_builder
    
    def _get_value_fn(self, v: float | str) -> Callable[[U], Array]:
        """Convert a value or parameter name into a function"""
        if isinstance(v, (int, float)):
            return lambda _: jnp.array(v)
        # Check if it's a parameter reference
        if v in self.model_builder.param_vars:
            return lambda p: p[self.model_builder.param_vars[v].index.select(p)]
        # Check if it's a data reference
        if v in self.model_builder.data_vars:
            return lambda _: self.model_builder.data_vars[v].value.get()
        raise ValueError(f"Unknown reference: {v}")
    
    def _add_distribution(
        self, 
        target: str, 
        dist_type: str,
        params: dict[str, float | str]
    ) -> 'ModelBlockBuilder[U, V]':
        """Generic method to add any distribution"""
        # Get target specification (could be either parameter or data)
        if target in self.model_builder.param_vars:
            target_spec = self.model_builder.param_vars[target]
            param_index = target_spec.index
        elif target in self.model_builder.data_vars:
            target_spec = self.model_builder.data_vars[target]
            # For data targets, we need to create an appropriate index
            param_index = ParamIndex.vector(0, target_spec.shape[0]) if len(target_spec.shape) > 0 else ParamIndex.single(0)
        else:
            raise ValueError(f"Unknown target: {target}")

        # Convert all parameter references to functions
        param_fns = {
            name: self._get_value_fn(value) 
            for name, value in params.items()
        }
        
        # Create the appropriate distribution
        if dist_type == "normal":
            distribution = Distribution.normal(
                mean=param_fns["mean"],
                std=param_fns["std"]
            )
        elif dist_type == "exponential":
            distribution = Distribution.exponential(
                rate=param_fns["rate"]
            )
        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")
        
        node = BayesNode(
            node_id=len(self.model_builder.nodes),
            param_index=param_index,
            log_density=distribution.log_prob
        )
        self.model_builder.nodes.append(node)
        return self
    
    def normal(
        self, 
        target: str, 
        mean: float | str, 
        std: float | str
    ) -> 'ModelBlockBuilder[U, V]':
        """Add normal distribution"""
        return self._add_distribution(
            target=target,
            dist_type="normal",
            params={"mean": mean, "std": std}
        )
    
    def exponential(
        self, 
        target: str, 
        rate: float | str
    ) -> 'ModelBlockBuilder[U, V]':
        """Add exponential distribution"""
        return self._add_distribution(
            target=target,
            dist_type="exponential",
            params={"rate": rate}
        )
    
    def done(self) -> ModelBuilder[U, V]:
        return self.model_builder


# Utility functions for working with models
def get_initial_params(model: Model[U, V], random_key: Array) -> U:
    """Generate initial parameters for the model"""
    size = model.get_param_size()
    return cast(U, jax.random.normal(random_key, (size,)))


def validate_data(model: Model[U, V], data_dict: dict[str, Array]) -> None:
    """Validate that provided data matches model specifications"""
    for name, spec in model.data_vars.items():
        if name not in data_dict:
            raise ValueError(f"Missing data for variable: {name}")
        data = data_dict[name]
        if data.shape != spec.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {spec.shape}, got {data.shape}"
            )


def bind_data(model: Model[U, V], data_dict: dict[str, Array]) -> Model[U, V]:
    """Bind data to model variables.
    
    Args:
        model: The model to bind data to
        data_dict: Dictionary mapping variable names to their values
        
    Returns:
        A new model with the data bound to it
    """
    # Validate data first
    validate_data(model, data_dict)
    
    # Update data specifications
    new_data_vars = {}
    for name, spec in model.data_vars.items():
        if name in data_dict:
            new_spec = DataSpec(
                name=spec.name,
                shape=spec.shape,
                value=BayesVar.just(data_dict[name])
            )
            new_data_vars[name] = new_spec
        else:
            new_data_vars[name] = spec
    
    # Create mapping from node_id to data
    node_data_map: dict[int, Array] = {}
    for node in model.nodes:
        # Check if this node is a likelihood node (has data dependency)
        for name, data in data_dict.items():
            if name in model.data_vars:
                # If the node's parameter index matches any data variable's shape,
                # assume it's the corresponding likelihood node
                if (isinstance(node.param_index.indices[0].stop, int) and 
                    node.param_index.indices[0].stop - node.param_index.indices[0].start == data.shape[0]):
                    node_data_map[node.node_id] = data
    
    # Create new nodes with bound data
    new_nodes = [
        BayesNode(
            node_id=node.node_id,
            param_index=node.param_index,
            log_density=node.log_density,
            data=node_data_map.get(node.node_id)
        )
        for node in model.nodes
    ]
    
    return Model(new_nodes, new_data_vars, model.param_vars)


