from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Callable, Tuple, TypeVar, Generic

class ArrayLike(Protocol):
    def __getitem__(self, key) -> 'ArrayLike': ...
    def __setitem__(self, key, value) -> None: ...
    def __add__(self, other: 'ArrayLike') -> 'ArrayLike': ...
    def __sub__(self, other: 'ArrayLike') -> 'ArrayLike': ...
    def __mul__(self, other: 'ArrayLike') -> 'ArrayLike': ...
    def __truediv__(self, other: 'ArrayLike') -> 'ArrayLike': ...
    def __neg__(self) -> 'ArrayLike': ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    def dot(self, other: 'ArrayLike') -> 'ArrayLike': ...


T = TypeVar('T', bound=ArrayLike)

# Define type annotations for clarity
PotentialFn = Callable[[T], float]
GradientFn = Callable[[T], T]

@dataclass
class BaseHMCInput(Generic[T]):
    U: PotentialFn
    grad_U: GradientFn
    epsilon: float
    L: int
    current_q: T

@dataclass
class BaseHMCOuput(Generic[T]):
    q: T


class HMCProtocol(Protocol[T]):

    def hmc(self, input: BaseHMCInput[T]) -> BaseHMCOuput[T]:
        pass

    def run_hmc(self, input: BaseHMCInput[T], num_samples: int) -> T:
        pass

class AbstractHMC(ABC, HMCProtocol[T]):

    @abstractmethod
    def hmc(self, input: BaseHMCInput[T]) -> BaseHMCOuput[T]:
        pass


    @abstractmethod
    def run_hmc(self, input: BaseHMCInput[T], num_samples: int) -> T:
        pass
