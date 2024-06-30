import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import jax.random as random
from laplacean.backend.base import HMC, PotentialFn, GradientFn


import seaborn as sns
import matplotlib.pyplot as plt

class JaxHMC(HMC):

    def __init__(self, key: jax.random.key):
        self.key = key

    def _hmc_step(self, U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, current_q: Array, key: jax.random.key) -> Array:
        q = jnp.array(current_q)
        key, subkey = random.split(key)
        p = random.normal(subkey, q.shape)

        jax.debug.print("key: {key}", key = key)


        current_p = p
        # Make a half step for momentum at the beginning
        p = p - epsilon * grad_U(q) / 2

        for i in range(L):
            # Make a full step for the position
            q = q + epsilon * p
            # Make a full step for the momentum, except at the end of trajectory
            if i != L - 1:
                p = p - epsilon * grad_U(q)
        
        # Make a half step for momentum at the end
        p = p - epsilon * grad_U(q) / 2
        # Negate momentum at end of trajectory to make the proposal symmetric
        p = -p
        jax.debug.print("p: {p}", p = p)
        jax.debug.print("q: {q}", q = q)
        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = U(current_q)
        jax.debug.print("U: {U}", U = current_U)
        current_K = jnp.sum(current_p ** 2) / 2
        jax.debug.print("K: {K}", K = current_K)
        proposed_U = U(q)
        jax.debug.print("prop_U: {prop_U}", prop_U = proposed_U)
        proposed_K = jnp.sum(p ** 2) / 2
        jax.debug.print("prop_K: {prop_K}", prop_K = proposed_K)
        # Accept or reject the state at the end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)
        print("====================")
        return q, accept_prob, key
    
    def hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, current_q: Array) -> Array:
        q, accept_prob, self.key = self._hmc_step(U, grad_U, epsilon, L, current_q, self.key)
        self.key, subkey = random.split(self.key)
        accept = random.uniform(subkey) < accept_prob
        q_new = jax.lax.cond(accept, lambda _: q, lambda _: current_q, operand=None)
        return q_new

    def run_hmc(self, U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, initial_q: Array, num_samples: int) -> Array:
        def body_fun(carry, _):
            q, key = carry
            q_new = self.hmc(U, grad_U, epsilon, L, q)
            return (q_new, key), q_new
        _, samples = jax.lax.scan(body_fun, (initial_q, self.key), jnp.arange(num_samples))
        return samples


#
# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
def U(q: Array) -> Float:
    return 0.5 * jnp.sum(q ** 2)

def grad_U(q: Array) -> Array:
    return jax.grad(U)(q)

initial_q = jnp.array([1.])

hmc = JaxHMC(random.PRNGKey(0))
samples = hmc.run_hmc(U, grad_U, 0.1, 10, initial_q, 2)

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