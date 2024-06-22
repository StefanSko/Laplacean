from typing import Protocol, Callable, Tuple, Union
import numpy as np
import jax.numpy as jnp

ArrayLike = Union[np.ndarray, jnp.ndarray]

class HMC(Protocol):

    def hmc(self, U: Callable[[ArrayLike], float], grad_U: Callable[[ArrayLike], ArrayLike], epsilon: float, L: int, current_q: ArrayLike, key: jnp.ndarray) -> Tuple[ArrayLike, float]:
        pass

    def run_hmc(self, U: Callable[[ArrayLike], float], grad_U: Callable[[ArrayLike], ArrayLike], epsilon: float, L: int, initial_q: ArrayLike, num_samples: int, key: jnp.ndarray) -> ArrayLike:
        pass


