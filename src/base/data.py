from typing import Literal, Generic, TypeVar, Protocol
from jaxtyping import Array
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class JaxHMCData:
    epsilon: float
    L: int
    current_q: Array
    key: Array

class Index:
    """Unified indexing for parameters"""
    index: slice

    def __init__(self, index: slice):
        self.index = index

    @staticmethod
    def single(i: int) -> 'Index':
        """Create an index for a single value"""
        return Index(slice(i, i + 1))

    @staticmethod
    def vector(start: int, end: int | None = None) -> 'Index':
        """Create an index for a vector of values"""
        if start < 0:
            raise ValueError("Start index cannot be negative")
        if end is not None and end <= start:
            raise ValueError("End index must be greater than start index")
        return Index(slice(start, end))

    def select(self, random_var: Array) -> Array:
        """Select values from an array using this index"""
        return random_var[self.index]

    def get_shape(self) -> tuple[int, ...]:
        """Infer the shape from the slice index"""
        start = 0 if self.index.start is None else self.index.start
        stop = self.index.stop if self.index.stop is not None else start + 1
        return (stop - start,)


# Define type variants
VarKind = Literal["parameter", "observed"]
P = TypeVar("P", bound=VarKind)

class ValueProvider(Protocol):
    def __call__(self, state: Array | None = None) -> Array:
        """
        Unified interface for both parameters and data
        - For data: state is ignored (None)
        - For parameters: state is the current parameter vector
        """
        ...

# Implementation for data
def make_data_provider(data: Array) -> ValueProvider:
    def provider(state: Array | None = None) -> Array:
        return data
    return provider

# Implementation for parameters
def make_parameter_selector(index: Index) -> ValueProvider:
    def selector(state: Array | None = None) -> Array:
        if state is None:
            raise ValueError("Parameter selector requires state")
        return index.select(state)
    return selector

class RandomVar(Generic[P]):
    def __init__(self, 
                 name: str, 
                 shape: tuple[int, ...], 
                 provider: ValueProvider,
                 var_kind: P) -> None:
        if not name:
            raise ValueError("Name cannot be empty")
        if not shape:
            raise ValueError("Shape tuple cannot be empty")
        if not all(dim > 0 for dim in shape):
            raise ValueError("All dimensions must be positive")
        
        self._name = name
        self._shape = shape
        self._provider = provider
        self._var_kind = var_kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def var_kind(self) -> P:
        return self._var_kind

    def get_value(self, state: Array | None = None) -> Array:
        return self._provider(state)

class RandomVarFactory:
    @staticmethod
    def from_data(name: str, data: Array) -> RandomVar[Literal["observed"]]:
        return RandomVar(name, data.shape, make_data_provider(data), "observed")

    @staticmethod
    def from_parameter(name: str, index: Index) -> RandomVar[Literal["parameter"]]:
        return RandomVar(name, index.get_shape(), make_parameter_selector(index), "parameter")

# Type aliases using the new syntax
Parameter = RandomVar[Literal["parameter"]]
ObservedVariable = RandomVar[Literal["observed"]]

