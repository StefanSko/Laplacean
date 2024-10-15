import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from methods.bayesian_execution_network import (
    QueryPlan, create_prior_node, normal_prior, exponential_prior, BayesianExecutionModel, normal_likelihood,
    create_likelihood_node, bind_data
)
from methods.hmc import step
from sampler.sampling import Sampler
from base.data import JaxHMCData

# Set up the model
def mu_mean(params):
    return jnp.array(0.0)

def mu_std(params):
    return jnp.array(1.0)

def sigma_rate(params):
    return jnp.array(1.0)  # rate parameter for exponential distribution

def likelihood_mean(params, data):
    return params[0]  # mu

def likelihood_std(params, data):
    return params[1]  # sigma (no need for exp now)


# Create nodes
mu_prior = create_prior_node(0, normal_prior(mu_mean, mu_std))
sigma_prior = create_prior_node(1, exponential_prior(sigma_rate))
likelihood = create_likelihood_node(2, normal_likelihood(likelihood_mean, likelihood_std))

# Create query plan
query_plan = QueryPlan([mu_prior, sigma_prior, likelihood])


# Simulate data
key = random.PRNGKey(0)
true_mu, true_sigma = 1.0, 2.0
n_samples = 1000
data = random.normal(key, shape=(n_samples,)) * true_sigma + true_mu


# Bind data to the likelihood node
query_plan = bind_data(2, data, query_plan)

# Create model
model = BayesianExecutionModel(query_plan)

# Set up HMC
initial_params = jnp.array([0.0, 1.0])  # [mu, sigma]
input_data = JaxHMCData(epsilon=0.01, L=20, current_q=initial_params, key=random.PRNGKey(1))
sampler = Sampler()

# Run HMC
n_warmup = 1000
n_iterations = 10000
samples = sampler(step, input_data, model, num_warmup=n_warmup, num_samples=n_iterations)


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