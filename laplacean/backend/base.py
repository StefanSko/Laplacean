from typing import Protocol, Callable, Tuple, Union, Float
import numpy as np
import jax.numpy as jnp

ArrayLike = Union[np.ndarray, jnp.ndarray]


# Define type annotations for clarity
PotentialFn = Callable[[ArrayLike], Float]
GradientFn = Callable[[ArrayLike], ArrayLike]

class HMC(Protocol):

    def hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, current_q: ArrayLike) -> Tuple[ArrayLike, Float]:
        pass

    def run_hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, initial_q: ArrayLike, num_samples: int) -> ArrayLike:
        pass


