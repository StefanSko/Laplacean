
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from logging_utils import logger
from methods.hmc import step
from methods.potential_energy import BayesianModel, normal_log_density
from sampler.sampling import Sampler
from base.data import JaxHMCData


initial_q = jnp.array([1.])

input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
sampler = Sampler()

normal = normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0))

# Define the prior (Gaussian with mean 0 and variance 1)
model = BayesianModel((normal,))

samples = sampler(step, input, model)

mean = jnp.mean(samples)
var = jnp.var(samples)

logger.info(f"mean = {mean}")
logger.info(f"var = {var}")

sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()