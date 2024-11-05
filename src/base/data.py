from jaxtyping import Array
import jax_dataclasses as jdc
from typing import TypeVar, Protocol, Union, cast, runtime_checkable

T = TypeVar('T')

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


@runtime_checkable
class DataProvider(Protocol):
    def __call__(self, random_vars: Array) -> Array: ...

class RandomVar:
    def __init__(self, name: str, shape: tuple[int, ...], provider: Union[DataProvider, Index]) -> None:
        self._name = name
        self._shape = shape
        # Provider can be either a callable or an Index
        self._provider: DataProvider = (
            provider if isinstance(provider, DataProvider) 
                                      else cast(DataProvider, lambda x: provider.select(x)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def get_value(self, random_vars: Array) -> Array:
        return self._provider(random_vars)

    @classmethod
    def from_index(cls, name: str, shape: tuple[int, ...], index: Index) -> 'RandomVar':
        """Convenience constructor for index-based random variables"""
        return cls(name, shape, index)
