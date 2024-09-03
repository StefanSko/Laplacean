
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from methods.hmc import step
from methods.potential_energy import LaplaceanPotentialEnergy, GaussianLogDensity, ConstantLogDensity
from sampler.sampling import Sampler
from base.data import JaxHMCData


U = GaussianLogDensity(mean=jnp.array([0.]), var=jnp.array([1.]))

initial_q = jnp.array([1.])

input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
sampler = Sampler()

# Define the prior (Gaussian with mean 0 and variance 1)
prior: GaussianLogDensity = GaussianLogDensity(mean=jnp.array([0.]), var=jnp.array([1.]))

likelihood: ConstantLogDensity = ConstantLogDensity()

# Create the potential energy
potential_energy: LaplaceanPotentialEnergy = LaplaceanPotentialEnergy(log_prior=prior, log_likelihood=likelihood)

samples = sampler(step, input, potential_energy)

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