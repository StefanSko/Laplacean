from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax_dataclasses as jdc


import jax.random as random

import seaborn as sns
import matplotlib.pyplot as plt

# Define type annotations for clarity
PotentialFn = Callable[[Array], float]
GradientFn = Callable[[Array], Array]

@jdc.pytree_dataclass
class JaxHMCInput:
    epsilon: float
    L: int
    current_q: Array
    key: Callable[[int], Array]

@jdc.pytree_dataclass
class JaxHMCOuput:
    q: Array
    key: Callable[[int], Array]

class HMCProtocol:

    def hmc(self, input: JaxHMCInput) -> JaxHMCOuput:
        pass

    def run_hmc(self, input: JaxHMCInput, num_samples: int) -> Array:
        pass

class JaxHMC:

    def __init__(self, U: PotentialFn, grad_U: GradientFn):
        self.U = U
        self.grad_U = grad_U

    def hmc_step(self, input: JaxHMCInput) -> JaxHMCOuput:
        q = input.current_q
        key, subkey = random.split(input.key)
        p = random.normal(subkey, q.shape)
        epsilon = input.epsilon
        L = input.L
        current_p = p
        # Make a half step for momentum at the beginning
        p = p - epsilon * self.grad_U(q) / 2

        for i in range(L):
            # Make a full step for the position
            q = q + epsilon * p
            # Make a full step for the momentum, except at the end of trajectory
            if i != L - 1:
                p = p - epsilon * self.grad_U(q)
        
        # Make a half step for momentum at the end
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
        return JaxHMCOuput(q=q_new, key=key)
        

    def run_hmc(self, input: JaxHMCInput, num_samples: int) -> Array:
        def body_fun(carry, _):
            input, key = carry
            output = self.hmc_step(JaxHMCInput(epsilon=input.epsilon, L=input.L, current_q=input.current_q, key=key))
            return (output, output.key), output.q
        key, subkey = random.split(input.key)
        _, samples = jax.lax.scan(body_fun, (input, subkey), jnp.zeros(num_samples))
        return samples


#
# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
@jax.jit
def U(q: jnp.array) -> Float:
    return 0.5 * jnp.sum(q ** 2)

@jax.jit
def grad_U(q: jnp.array) -> jnp.array:
    return jax.grad(U)(q)

initial_q = jnp.array([1.])

#TODO: Fix the issue with JaxHMCInput not being a valid type for jax	
hmc: HMCProtocol = JaxHMC(U=U, grad_U=grad_U)
input: JaxHMCInput = JaxHMCInput(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
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