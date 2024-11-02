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

# Simulate data
key = random.PRNGKey(0)
true_mu, true_sigma = 1.0, 2.0
n_samples = 1000
data = random.normal(key, shape=(n_samples,)) * true_sigma + true_mu

# Create model using Stan-like interface
# Build the model
model = (ModelBuilder()
    .data()
        .int_scalar("N", jnp.array(n_samples))                # Number of observations
        .vector("y", "N")                                # Data vector
    .done()
    .parameters()
        .real("mu")                           # Mean parameter
        .real("sigma")                        # Standard deviation parameter
    .done()
    .model()
        # Priors
        .normal("mu", loc=0.0, scale=1.0)     # Prior for mean: N(0,1)
        .exponential("sigma", rate=1.0)       # Prior for std: Exp(1) [must be positive]
        # Likelihood
        .normal("y", loc="mu", scale="sigma") # y ~ Normal(mu, sigma)
    .done()
)

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

mu_samples = samples[:, 0]
sigma_samples = samples[:, 1]

print(f"True mu: {true_mu}, Estimated mu: {jnp.mean(mu_samples):.4f}")
print(f"True sigma: {true_sigma}, Estimated sigma: {jnp.mean(sigma_samples):.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(mu_samples, kde=True, ax=ax1)
ax1.axvline(true_mu, color='r', linestyle='--')
ax1.set_title('Posterior distribution of μ')
ax1.set_xlabel('μ')

sns.histplot(sigma_samples, kde=True, ax=ax2)
ax2.axvline(true_sigma, color='r', linestyle='--')
ax2.set_title('Posterior distribution of σ')
ax2.set_xlabel('σ')

plt.tight_layout()
plt.show()

# Trace plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(mu_samples)
ax1.set_title('Trace of μ')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('μ')

ax2.plot(sigma_samples)
ax2.set_title('Trace of σ')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('σ')

plt.tight_layout()
plt.show()
