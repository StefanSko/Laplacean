import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Generic, NamedTuple, Optional, TypeVar, cast, Any
from jaxtyping import Array, Float
from jax.scipy.stats import expon, norm

Parameter = Array
Data = Array
LogDensity = Float[Array, ""]

Variable = Parameter | Data

T = TypeVar('T', bound=Variable)
U = TypeVar('U', bound=Parameter)
V = TypeVar('V', bound=Data)

NONE: LogDensity = jnp.array(0.0)

class BayesVar(eqx.Module, Generic[T]):
    value: T

    def __init__(self, value: T):
        self.value = value

    def get(self) -> Any:
        return self.value

    def map(self, f: Callable[[T], Any]) -> Any:
        return jax.lax.cond(
            jnp.any(jnp.isnan(self.value)),
            lambda _: NONE,
            lambda x: f(x),
            self.value
        )

    @classmethod
    def empty(cls) -> 'BayesVar[T]':
        return cls(cast(T, jnp.array(float('nan'))))

    @classmethod
    def just(cls, value: T) -> 'BayesVar[T]':
        return cls(value)

#TODO: Think about long term node indexing for likelihood
class LikelihoodState(eqx.Module, Generic[U, V]):
    log_likelihood: Callable[[U, V], LogDensity]
    data: BayesVar[V]

    def __init__(self, log_likelihood: Callable[[U, V], LogDensity], data: Optional[V] = None):
        self.log_likelihood = log_likelihood
        self.data = BayesVar.just(data) if data is not None else BayesVar.empty()

class IdentityParam(NamedTuple):
    pass

class SingleParam(NamedTuple):
    index: int

class ParamVector(NamedTuple):
    start: int = 0
    end: int | None = None

class ParamMatrix(NamedTuple):
    row: int | ParamVector = ParamVector()
    col: int | ParamVector = ParamVector()

ParamIndex = IdentityParam | SingleParam | ParamVector | ParamMatrix

class PriorNode(Generic[U], eqx.Module):
    node_id: int
    param_index: ParamIndex
    log_density: Callable[[U], LogDensity]
    
    def __init__(self, node_id: int, param_index: ParamIndex, log_density: Callable[[U], LogDensity]):
        self.node_id = node_id
        self.param_index = param_index
        self.log_density = log_density

    def evaluate(self, params: U) -> LogDensity:
        selected_params = _select_params(params, self.param_index)
        log_pdf = self.log_density(selected_params)
        jax.debug.print("node_id -> {}; logpdf -> {}", self.node_id, log_pdf)
        return log_pdf

class LikelihoodNode(Generic[U, V], eqx.Module):
    node_id: int
    param_index: ParamIndex
    state: LikelihoodState[U, V]
    
    def __init__(self, node_id: int, param_index: ParamIndex, log_likelihood: Callable[[U, V], LogDensity], data: V = None):
        self.node_id = node_id
        self.param_index = param_index
        self.state = LikelihoodState(log_likelihood, data)

    def evaluate(self, params: U) -> LogDensity:
        selected_params = _select_params(params, self.param_index)
        log_pdf = self.state.data.map(lambda d: self.state.log_likelihood(selected_params, d))
        jax.debug.print("node_id -> {}; logpdf -> {}", self.node_id, log_pdf)
        return log_pdf



    @classmethod
    def bind_data(cls, node: 'LikelihoodNode[U, V]', data: V) -> 'LikelihoodNode[U, V]':
        new_state = LikelihoodState(node.state.log_likelihood, data)
        return cls(node.node_id, node.param_index, new_state.log_likelihood, data)

class SubModelNode(Generic[U, V], eqx.Module):
    
    node_id: int
    sub_model: 'QueryPlan[U, V]'
    
    def __init__(self, node_id: int, sub_model: 'QueryPlan[U, V]'):
        self.node_id = node_id
        self.sub_model = sub_model

    def evaluate(self, params: U) -> LogDensity:
        return execute_query_plan(self.sub_model, params)

NodeType = PriorNode[U] | LikelihoodNode[U, V] | SubModelNode[U, V]

class QueryPlan(Generic[U, V]):
    def __init__(self, nodes: list[NodeType]):
        self.nodes = nodes

def normal_prior(
    mean: Callable[[U], Array],
    std: Callable[[U], Array],
    index: ParamIndex = IdentityParam()
) -> Callable[[U], LogDensity]:
    def log_prob(params: U) -> LogDensity:
        selected_params_mean = _select_params(params, index)
        selected_params_std = _select_params(params, index)
        return jnp.sum(-0.5 * ((params - mean(selected_params_mean)) / std(selected_params_std)) ** 2)
    return log_prob

def exponential_prior(
    rate: Callable[[U], Array]
) -> Callable[[U], LogDensity]:
    def log_prob(params: U) -> LogDensity:
        return expon.logpdf(params, scale=1/rate(params))
    return log_prob

class ParamFunction(NamedTuple):
    func: Callable[[U, V], Array]
    param_index: ParamIndex

def normal_likelihood(
    mean: ParamFunction,
    std: ParamFunction
) -> Callable[[U, V], LogDensity]:
    def log_likelihood(params: U, data: V) -> LogDensity:
        selected_params_mean = _select_params(params, mean.param_index)
        selected_params_std = _select_params(params, std.param_index)
        mean_value = mean.func(selected_params_mean, data)
        std_value = std.func(selected_params_std, data)
        return jnp.sum(norm.logpdf(data, mean_value, std_value))
    return log_likelihood

def _select_params(params: U, index: ParamIndex) -> U:
    match index:
        case IdentityParam():
            return cast(U,params)
        case SingleParam(i):
            return cast(U, params[i])
        case ParamVector(start, end):
            return cast(U, params[start:end])
        case ParamMatrix(row, col):
            return cast(U, params[_get_slice(row), _get_slice(col)])
        case _:
            raise ValueError(f"Unsupported param_index type: {type(index)}")

def _get_slice(index: int | ParamVector) -> slice:
    match index:
        case int():
            return slice(index, index + 1)
        case ParamVector(start, end):
            return slice(start, end)

def create_prior_node(node_id: int, param_index: ParamIndex, log_density: Callable[[U], LogDensity]) -> PriorNode[U]:
    return PriorNode(node_id, param_index, log_density)

def create_likelihood_node(node_id: int, param_index: ParamIndex, log_likelihood: Callable[[U, V], LogDensity]) -> LikelihoodNode[U, V]:
    return LikelihoodNode(node_id, param_index, log_likelihood)

def create_sub_model_node(node_id: int, sub_model: QueryPlan[U, V]) -> SubModelNode[U, V]:
    return SubModelNode(node_id, sub_model)

def execute_query_plan(query_plan: QueryPlan[U, V], params: U) -> LogDensity:
    return jnp.sum(jnp.array([node.evaluate(params) for node in query_plan.nodes]))

def bind_data(target_id: int, data: V, query_plan: QueryPlan[U, V]) -> QueryPlan[U, V]:
    def update_node(node: NodeType) -> NodeType:
        if isinstance(node, LikelihoodNode) and node.node_id == target_id:
            return LikelihoodNode.bind_data(node, data)
        elif isinstance(node, SubModelNode):
            updated_sub_model = bind_data(target_id, data, node.sub_model)
            return SubModelNode(node.node_id, updated_sub_model)
        return node
    
    updated_nodes = [update_node(node) for node in query_plan.nodes]
    return QueryPlan(updated_nodes)

class BayesianExecutionModel(eqx.Module, Generic[U, V]):
    query_plan: QueryPlan[U, V]

    def __init__(self, query_plan: QueryPlan[U, V]):
        self.query_plan = query_plan

    def __call__(self, params: U) -> LogDensity:
        return execute_query_plan(self.query_plan, params)

    def potential_energy(self, params: U) -> LogDensity:
        return -self(params)

    def gradient(self, params: U) -> Array:
        return jax.grad(self.potential_energy)(params)


class Model(eqx.Module, Generic[U, V]):
    query_plan: QueryPlan[U, V]
    data: V
    params: U
    
    def __init__(self, query_plan: QueryPlan[U, V], params: U, data: BayesVar[V]):
        self.query_plan = query_plan
        self.params = params
        self.data = data
    
    def log_prob(self) -> LogDensity:
        """Evaluates the total log probability of the model."""
        return execute_query_plan(self.query_plan, self.params)
    
    def potential_energy(self) -> LogDensity:
        """Returns the potential energy (negative log probability)."""
        return -self.log_prob()
    
    def gradient(self) -> Array:
        """Returns the gradient of the potential energy with respect to parameters."""
        return jax.grad(lambda p: -execute_query_plan(self.query_plan, p))(self.params)
    
    @classmethod
    def builder(cls):
        """Returns a ModelBuilder to construct models in a more Stan-like way."""
        return ModelBuilder()

class ModelBuilder(Generic[U, V]):
    def __init__(self) -> None:
        self.nodes: list[NodeType] = []
        self._current_node_id: int = 0
    
    def add_prior(self, name: str, distribution: str, *args: Any, **kwargs: Any) -> 'ModelBuilder':
        """Adds a prior to the model in a Stan-like way.
        Example: builder.add_prior('mu', 'normal', 0, 10)
        """
        param_index = kwargs.pop('param_index', IdentityParam())
        
        if distribution == 'normal':
            mean, std = args
            log_density = normal_prior(
                lambda _: jnp.array(mean),
                lambda _: jnp.array(std),
                param_index
            )
        elif distribution == 'exponential':
            rate = args[0]
            log_density = exponential_prior(lambda _: jnp.array(rate))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        node: PriorNode[U] = create_prior_node(self._current_node_id, param_index, log_density)
        self.nodes.append(node)
        self._current_node_id += 1
        return self
    
    def add_likelihood(self, name: str, distribution: str, 
                      mean_fn: ParamFunction, std_fn: ParamFunction) -> 'ModelBuilder':
        """Adds a likelihood to the model.
        Example: builder.add_likelihood('y', 'normal', mean_fn, std_fn)
        """
        if distribution == 'normal':
            log_likelihood = normal_likelihood(mean_fn, std_fn)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        node: LikelihoodNode[U, V] = create_likelihood_node(self._current_node_id, 
                                    mean_fn.param_index, 
                                    log_likelihood)
        self.nodes.append(node)
        self._current_node_id += 1
        return self
    
    def build(self, params: U, data: BayesVar[V]) -> Model[U, V]:
        """Builds the final model with the specified parameters and data."""
        query_plan: QueryPlan[U, V] = QueryPlan(self.nodes)
        return Model(query_plan, params, data)
    

    
    
