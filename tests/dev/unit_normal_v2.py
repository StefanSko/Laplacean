import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from methods.bayesian_network_v2 import (
    ModelBuilder, get_initial_params
)
from methods.hmc import step
from sampler.sampling import Sampler
from base.data import JaxHMCData

# Create model using Stan-like interface
model = (ModelBuilder()
         .parameters()
         .real("x")  # single parameter to sample
         .done()
         .model()
         # Single standard normal prior
         .normal("x", 0, 1)
         .done())

# Initialize HMC sampler
initial_params = get_initial_params(model, random_key=random.PRNGKey(0))
input_data = JaxHMCData(
    epsilon=0.1,
    L=10,
    current_q=initial_params,
    key=random.PRNGKey(0)
)

# Create and run sampler
sampler = Sampler()
samples = sampler(step, input_data, model)

# Calculate statistics
mean = jnp.mean(samples)
var = jnp.var(samples)

print("Sample Statistics:")
print(f"mean = {mean:.3f}")
print(f"var = {var:.3f}")

# Visualizations
plt.figure(figsize=(12, 4))

# Distribution plot
plt.subplot(121)
sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')

# Trace plot
plt.subplot(122)
sns.lineplot(data=samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')

plt.tight_layout()
plt.show()