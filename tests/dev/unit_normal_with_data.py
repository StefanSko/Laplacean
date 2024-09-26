import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from logging_utils import logger
from methods.hmc import step
from methods.potential_energy import BayesianModel, bind_data, parameterized_normal_log_density
from sampler.sampling import Sampler
from base.data import JaxHMCData

# Generate data from N(1, 2)
key = random.PRNGKey(0)
data = random.normal(key, (100,)) * 2 + 1

# Define the prior (which will act as our model)
def mean(x):
    return jnp.array(0.0)
def std(x):
    return jnp.array(1.0)

normal = parameterized_normal_log_density(mean, std)

# Create BayesianModel with just the prior
model = BayesianModel((normal,))

# Bind data
model_with_data = bind_data(model, {'y': data})

# Set up HMC
initial_q = jnp.array([0.0])  # initial guess for the mean
input = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(1))
sampler = Sampler()

# Run HMC
samples = sampler(step, input, model_with_data)

# Analyze results
mean = jnp.mean(samples)
std = jnp.std(samples)

logger.info(f"Estimated mean: {mean:.3f} ± {std:.3f}")

# Plot results
plt.figure(figsize=(8, 4))

plt.subplot(121)
sns.histplot(data, bins=20, kde=True)
plt.xlabel('Data')
plt.title('Histogram of Data')

plt.subplot(122)
sns.kdeplot(samples)
plt.xlabel('Mean')
plt.title('Posterior Distribution of Mean')

plt.tight_layout()
plt.show()