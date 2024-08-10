from jaxtyping import Array
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class JaxHMCData:
    epsilon: float
    L: int
    current_q: Array
    key: Array
