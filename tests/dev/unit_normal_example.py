
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float
import equinox as eqx
import seaborn as sns
from matplotlib import pyplot as plt

from methods.hmc import step_filter
from methods.potential_energy import LaplaceanPotentialEnergy, GaussianLogDensity, LogDensity
from sampler.sampling import Sampler
from base.data import JaxHMCData


U = GaussianLogDensity(mean=jnp.array([0.]), var=jnp.array([1.]))

initial_q = jnp.array([1.])

input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
sampler = Sampler()

# Define the prior (Gaussian with mean 0 and variance 1)
prior = GaussianLogDensity(mean=jnp.array([0.]), var=jnp.array([1.]))

class ConstantLogDensity(LogDensity):
    @eqx.filter_jit
    def __call__(self, _: Array) -> Float[Array, ""]:  # noqa: F722
        return jnp.array(0.0)

likelihood = ConstantLogDensity()

# Create the potential energy
potential_energy = LaplaceanPotentialEnergy(log_prior=prior, log_likelihood=likelihood)

samples = sampler(step_filter, input, potential_energy)

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