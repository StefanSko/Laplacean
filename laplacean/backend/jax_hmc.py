from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array
import jax_dataclasses as jdc


import jax.random as random

import seaborn as sns
import matplotlib.pyplot as plt

# Define type annotations for clarity
PotentialFn = Callable[[Array], float]
GradientFn = Callable[[Array], Array]

@jdc.pytree_dataclass
class JaxHMCData:
    epsilon: float
    L: int
    current_q: Array
    key: Array


class HMCProtocol:

    def hmc(self, input: JaxHMCData) -> JaxHMCData:  # type: ignore
        ...

    def run_hmc(self, input: JaxHMCData, num_samples: int) -> Array:  # type: ignore
        ...

class JaxHMC(HMCProtocol):

    def __init__(self, U: PotentialFn, grad_U: GradientFn):
        self.U = U
        self.grad_U = grad_U

    def hmc_step(self, input: JaxHMCData) -> JaxHMCData:
        q = input.current_q
        key, subkey = random.split(input.key)
        p = random.normal(subkey, q.shape)
        epsilon = input.epsilon
        L = input.L
        current_p = p
        # Make a half step for momentum at the beginning
        p = p - epsilon * self.grad_U(q) / 2

        def loop_body(_, carry):
            q, p = carry
            # Make a full step for the position
            q = q + epsilon * p
            # Make a full step for the momentum
            p = p - epsilon * self.grad_U(q)
            return q, p

        q, p = jax.lax.fori_loop(0, L-1, loop_body, (q, p))
        
        # Make a half step for momentum at the end
        q = q + epsilon * p
        p = p - epsilon * self.grad_U(q) / 2
        # Negate momentum at end of trajectory to make the proposal symmetric
        p = -p
        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = self.U(input.current_q)
        current_K = jnp.sum(current_p ** 2) / 2
        proposed_U = self.U(q)
        proposed_K = jnp.sum(p ** 2) / 2
        # Accept or reject the state at the end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)
        key, subkey = random.split(key)
        accept = random.uniform(subkey) < accept_prob
        q_new = jax.lax.cond(accept, lambda _: q, lambda _: input.current_q, operand=None)
        return JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=q_new, key=key)
        

    def run_hmc(self, input: JaxHMCData, num_samples: int) -> Array:
        def body_fun(carry, _):
            input, key = carry
            output = self.hmc_step(JaxHMCData(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.current_q
        _, samples = jax.lax.scan(body_fun, (input, input.key), jnp.zeros(num_samples))
        return samples


#
# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
@jax.jit
def U(q: Array) -> Array:
    return 0.5 * jnp.sum(q ** 2)

@jax.jit
def grad_U(q: Array) -> Array:
    return jax.grad(U)(q)

initial_q = jnp.array([1.])

hmc: HMCProtocol = JaxHMC(U=U, grad_U=grad_U)
input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
samples = hmc.run_hmc(input, 1000)

print(jnp.mean(samples)) 
print(jnp.var(samples))


sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()