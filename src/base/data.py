from typing import Protocol
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

class ValueProvider(Protocol):
    def __call__(self, state: Array) -> Array:

        ...

def make_var_provider(index: Index) -> ValueProvider:
    def provider(state: Array) -> Array:
        if state is None:
            raise ValueError("state required")
        return index.select(state)
    return provider

class RandomVar:
    def __init__(self, 
                 name: str, 
                 shape: tuple[int, ...], 
                 provider: ValueProvider
                 ):
        if not name:
            raise ValueError("Name cannot be empty")
        if not shape:
            raise ValueError("Shape tuple cannot be empty")
        if not all(dim > 0 for dim in shape):
            raise ValueError("All dimensions must be positive")
        
        self._name = name
        self._shape = shape
        self._provider = provider

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def get_value(self, state: Array) -> Array:
        return self._provider(state)


