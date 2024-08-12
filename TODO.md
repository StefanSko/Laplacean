- check how to be able to jax.jit the gradient and requirements for that:
  - check following approach:

```
from methods.potential_energy import PotentialEnergy
from jaxtyping import Array


def U(q: Array) -> float:
    return 0.5

def create_test_f(energy_func: Callable[[jnp.ndarray], float]):
    @jax.jit
    def test_f(x: float) -> float:
        return energy_func(jnp.array(x))
    return test_f

energy = PotentialEnergy(U)

jit_test_f = create_test_f(energy.__call__)

test_energy = jit_test_f(1)

print(test_energy)
```

with jit takes:
```
Min Time: 0.239738 seconds
Max Time: 0.313666 seconds
Avg Time: 0.252053 seconds
```

without jit takes:
```
Min Time: 0.239547 seconds
Max Time: 0.258471 seconds
Avg Time: 0.245333 seconds
```
-does not seem to make much difference! try to understand why

- still not fully happy with current structure:
  - sampler should just run a function basically, no need for it to know about the input data which should be encapsulated in the step