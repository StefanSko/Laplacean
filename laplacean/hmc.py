import jax.numpy as jnp
from jax.random import PRNGKey, normal
from jaxtyping import Array, Float
from typing import Callable

import jax.random as random


# Define type annotations for clarity
PotentialFn = Callable[[Array], Float]
GradientFn = Callable[[Array], Array]

def hmc(U: PotentialFn, grad_U: GradientFn, epsilon: Float, L: int, initial_q: Array) -> Array:
    key = PRNGKey(0)
    q = initial_q
    p = normal(key, shape=initial_q.shape)  # Independent standard normal variates

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
    current_K = jnp.sum(current_U ** 2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p ** 2) / 2

    accept = random.uniform(key) < jnp.exp(current_U - proposed_U + current_K - proposed_K)
    return jnp.where(accept, q, initial_q)

# Example potential energy function and its gradient
def U(q: Array) -> Float:
    return 0.5 * jnp.sum(q ** 2)

def grad_U(q: Array) -> Array:
    return q

# Test inputs
epsilon = 0.9
L = 1000
initial_q = jnp.array([0.1, 0.1])

# Run the HMC sampler for a large number of steps
num_steps = 10000
from jax import lax

def body_func(carry, input):
    initial_q = carry  # Get the current position from the carry
    new_q = hmc(U, grad_U, epsilon, L, initial_q)  # Generate a new position
    return new_q, new_q  # Return the new position as both the new carry and the output

_, samples = lax.scan(body_func, initial_q, jnp.arange(num_steps))

# Compute the mean and covariance matrix of the samples
mean = jnp.mean(samples, axis=0)
cov = jnp.cov(samples, rowvar=False)

print("Mean:", mean)
print("Covariance matrix:", cov)
