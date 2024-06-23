from typing import Protocol, Callable, Union
import numpy as np
import jax.numpy as jnp

ArrayLike = Union[np.ndarray, jnp.ndarray]


# Define type annotations for clarity
PotentialFn = Callable[[ArrayLike], float]
GradientFn = Callable[[ArrayLike], ArrayLike]

class HMC(Protocol):

    def hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, current_q: ArrayLike) -> ArrayLike:
        pass

    def run_hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, initial_q: ArrayLike, num_samples: int) -> ArrayLike:
        pass


