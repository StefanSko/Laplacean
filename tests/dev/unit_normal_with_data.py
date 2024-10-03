import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from logging_utils import logger
from methods.hmc import step
from sampler.sampling import Sampler
from base.data import JaxHMCData
from methods.bayesian_execution_network import (
    BayesianExecutionModel, create_prior_node, create_likelihood_node, QueryPlan,
    normal_prior, normal_likelihood, bind_data
)

# Set up the random key
key = random.PRNGKey(0)

# Generate data from N(1, 2)
true_mean = 1.0
true_std = 2.0
num_samples = 1000
key, subkey = random.split(key)
data = random.normal(subkey, shape=(num_samples,)) * true_std + true_mean

# Set up the model
initial_q = jnp.array([0.0, 1.0])  # Initial guess for [mean, log(std)]

input: JaxHMCData = JaxHMCData(epsilon=0.01, L=10, current_q=initial_q, key=key)
sampler = Sampler()

# Prior: N(0, 1) for mean, and N(0, 1) for log(std)
def prior_mean(x):
    return jnp.array([0.0])

def prior_std(x):
    return jnp.array([1.0])

prior = normal_prior(prior_mean, prior_std)

# Likelihood: N(mean, exp(log_std))
def likelihood_mean(_, data):
    return jnp.mean(data)

def likelihood_std(_, data):
    return jnp.std(data)

likelihood = normal_likelihood(likelihood_mean, likelihood_std)

# Create nodes
prior_node = create_prior_node(0, prior)
likelihood_node = create_likelihood_node(1, likelihood)

# Create a query plan
query_plan = QueryPlan([prior_node, likelihood_node])

# Bind data to the likelihood node
query_plan = bind_data(1, data, query_plan)

# Create a BayesianExecutionModel
model = BayesianExecutionModel(query_plan)

# Run HMC
num_samples = 5000
samples = sampler(step, input, model, num_samples)

# Extract mean and std from samples
mean_samples = samples[:, 0]
std_samples = jnp.exp(samples[:, 1])

# Calculate statistics
mean_estimate = jnp.mean(mean_samples)
std_estimate = jnp.mean(std_samples)

logger.info(f"True mean: {true_mean}, Estimated mean: {mean_estimate}")
logger.info(f"True std: {true_std}, Estimated std: {std_estimate}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(mean_samples, kde=True)
plt.axvline(true_mean, color='r', linestyle='--')
plt.title('Posterior Distribution of Mean')

plt.subplot(2, 2, 2)
sns.histplot(std_samples, kde=True)
plt.axvline(true_std, color='r', linestyle='--')
plt.title('Posterior Distribution of Std Dev')

plt.subplot(2, 2, 3)
sns.scatterplot(x=mean_samples, y=std_samples, alpha=0.1)
plt.axvline(true_mean, color='r', linestyle='--')
plt.axhline(true_std, color='r', linestyle='--')
plt.title('Joint Posterior Distribution')
plt.xlabel('Mean')
plt.ylabel('Std Dev')

plt.subplot(2, 2, 4)
sns.lineplot(x=range(len(mean_samples)), y=mean_samples)
sns.lineplot(x=range(len(std_samples)), y=std_samples)
plt.title('Trace Plot')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend(['Mean', 'Std Dev'])

plt.tight_layout()
plt.show()