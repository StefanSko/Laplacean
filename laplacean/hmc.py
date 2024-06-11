import jax.numpy as jnp
from jax import grad
from jax.random import PRNGKey, normal
from jaxtyping import Array, Float
from typing import Callable, Tuple

import jax.random as random

import seaborn as sns
import matplotlib.pyplot as plt

# Define type annotations for clarity
PotentialFn = Callable[[Array], Float]
GradientFn = Callable[[Array], Array]


def hmc(U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, initial_q: Array, key: PRNGKey) -> Tuple[Array, PRNGKey]:
    key, subkey = random.split(key)
    q = initial_q
    p = normal(subkey, shape=initial_q.shape)  # Independent standard normal variates

    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2

    # Alternate full steps for position and momentum
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

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(initial_q)
    current_K = jnp.sum(initial_q ** 2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p ** 2) / 2

    # Accept or reject the new state based on Hamiltonian dynamics
    accept = random.uniform(key) < jnp.exp(current_U - proposed_U + current_K - proposed_K)
    new_q = jnp.where(accept, q, initial_q)
    return new_q, key

def run_hmc(U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, initial_q: Array, num_samples: int, key: PRNGKey) -> Array:
    samples = []
    q = initial_q

    # Burn-in phase (optional)
    burn_in_steps = 100
    for _ in range(burn_in_steps):
        q, key = hmc(U, grad_U, epsilon, L, q, key)

    # Sampling phase
    for _ in range(num_samples):
        q, key = hmc(U, grad_U, epsilon, L, q, key)
        samples.append(q)

    return jnp.array(samples)

# Example usage:
def U(q: Array) -> Float:
    return 0.5 * jnp.sum(q ** 2)

key = PRNGKey(0)
initial_q = jnp.array([1.0])
epsilon = 0.3
L = 10
num_samples = 2000

# Calculate gradient of U
grad_U = grad(U)

# Run HMC to get samples
samples = run_hmc(U, grad_U, epsilon, L, initial_q, num_samples, key)

# Compute the mean and covariance matrix of the samples
mean = jnp.mean(samples, axis=0)
cov = jnp.cov(samples, rowvar=False)

print("Mean:", mean)
print("Covariance matrix:", cov)

sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()
