import timeit

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import seaborn as sns
from matplotlib import pyplot as plt

from methods.hmc import step
from methods.potential_energy import PotentialEnergy
from sampler.sampling import Sampler
from base.data import JaxHMCData


# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
@jax.jit
def U(q: Array) -> Array:
    return 0.5 * jnp.sum(q ** 2)



initial_q = jnp.array([1.])

input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
sampler = Sampler()



potential_energy: PotentialEnergy = PotentialEnergy(prior_func=U)

samples = sampler(step, input, potential_energy)

def function():
    sampler(step, input, potential_energy)

# Use timeit to time the function 100 times
times = timeit.repeat("function()", setup="from __main__ import function", repeat=100, number=1)

# Calculate statistics
min_time = min(times)
max_time = max(times)
avg_time = sum(times) / len(times)

# Print out the statistics
print(f"Min Time: {min_time:.6f} seconds")
print(f"Max Time: {max_time:.6f} seconds")
print(f"Avg Time: {avg_time:.6f} seconds")

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