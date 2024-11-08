from jaxtyping import Array
import jax_dataclasses as jdc
from typing import Callable

@jdc.pytree_dataclass
class JaxHMCData:
    epsilon: float
    L: int
    current_q: Array
    key: Array


class Index:
    """Unified indexing for both parameters and data"""
    indices: tuple[slice, ...]

    def __init__(self, indices: tuple[slice, ...]):
        self.indices = indices

    @staticmethod
    def single(i: int) -> 'Index':
        """Create an index for a single value"""
        return Index((slice(i, i + 1),))

    @staticmethod
    def vector(start: int, end: int | None = None) -> 'Index':
        """Create an index for a vector of values"""
        return Index((slice(start, end),))

    def select(self, random_var: Array) -> Array:
        """Select values from an array using this index"""
        if len(self.indices) == 1:
            return random_var[self.indices[0]]
        return random_var[self.indices]

    
DataProvider = Callable[[], Array]
    
def from_idx(vec: Array, idx: Index) -> DataProvider:
    def idx_provider() -> Array:
        return idx.select(vec)
    return idx_provider

class RandomVar:
    def __init__(self, name: str, shape: tuple[int, ...], provider: DataProvider) -> None:
        self._name = name
        self._shape = shape
        self._provider: DataProvider = provider

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def get_value(self) -> Array:
        return self._provider()

    @classmethod
    def from_index(cls, name: str, index: Index, vec: Array) -> 'RandomVar':
        """Convenience constructor for index-based random variables"""
        selected_data = index.select(vec)
        return cls(name, selected_data.shape, from_idx(vec, index))


Parameter = RandomVar
"""Type alias for unobserved random variables (parameters to be inferred)"""

ObservedVariable = RandomVar
"""Type alias for observed random variables (data)"""