import jax
import jax.numpy as jnp
from jax import grad
from jax.random import PRNGKey
from jaxtyping import Array, Float
from typing import Callable, Tuple

import jax.random as random

import seaborn as sns
import matplotlib.pyplot as plt

# Define type annotations for clarity
PotentialFn = Callable[[Array], Float]
GradientFn = Callable[[Array], Array]


def hmc(U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, current_q: Array, key: jax.random.PRNGKey) -> Tuple[Array, Float]:
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
    proposed_K = jnp.sum(p ** 2) / 2
    jax.debug.print("prop_U: {prop_U}", prop_U = proposed_U)
    jax.debug.print("prop_K: {prop_K}", prop_K = proposed_K)
    # Accept or reject the state at the end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)
    print("====================")
    return q, accept_prob, key

def hmc_with_key(U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, current_q: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[Array, jax.random.PRNGKey]:
    q, accept_prob, key = hmc(U, grad_U, epsilon, L, current_q, key)
    key, subkey = random.split(key)
    accept = random.uniform(subkey) < accept_prob
    q_new = jax.lax.cond(accept, lambda _: q, lambda _: current_q, operand=None)
    return q_new, key

def run_hmc(U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, initial_q: jnp.ndarray, num_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    def body_fun(carry, _):
        q, key = carry
        q, key = hmc_with_key(U, grad_U, epsilon, L, q, key)
        return (q, key), q
    
    _, samples = jax.lax.scan(body_fun, (initial_q, key), jnp.arange(num_samples))
    return samples

#
# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
def U(q: Array) -> Float:
    return 0.5 * jnp.sum(q ** 2)

def grad_U(q: Array) -> Array:
    return jax.grad(U)(q)

initial_q = jnp.array([1.])
key = jax.random.PRNGKey(0)
samples = run_hmc(U, grad_U, epsilon=0.1, L=10, initial_q=initial_q, num_samples=2, key=key)
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


