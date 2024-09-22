import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import seaborn as sns
from matplotlib import pyplot as plt

from methods.hmc import step
from methods.potential_energy import BayesianModel, exponential_log_density, normal_log_density, \
    parameterized_normal_log_density, bind_data
from sampler.sampling import Sampler
from base.data import JaxHMCData

#TODO: FIX Sampling for regression

# Generate some observed data
n = 300
x = jnp.linspace(0, 1, n)
alpha_true = 1.0
beta_true = 2.0
sigma_true = 0.5
key = random.PRNGKey(0)
epsilon = sigma_true * random.normal(key, (n,))
y = alpha_true + beta_true * x + epsilon


# Define the model components
prior_alpha = normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0))
prior_beta = normal_log_density(mean=jnp.array(0.0), std=jnp.array(1.0))
prior_sigma = exponential_log_density()

def mean_function(params: Array, x: Array) -> Array:
    alpha, beta, _ = params
    return alpha + beta * x

likelihood = parameterized_normal_log_density(
    mean=lambda params, data: mean_function(params, data['x']),
    std=lambda params, data: jnp.exp(params[2])
)

# Create the initial Bayesian model (without data)
model = BayesianModel((prior_alpha, prior_beta, prior_sigma, likelihood))

# Bind the data to the model
data = {'x': x, 'y': y}
bound_model = bind_data(model, data)

# Initialize the HMC sampler
initial_params = jnp.array([0.0, 0.0, jnp.log(0.5)])
input_data = JaxHMCData(epsilon=0.005, L=12, current_q=initial_params, key=random.PRNGKey(1))

# Create and run the sampler
sampler = Sampler()
samples = sampler(step, input_data, bound_model, num_warmup=1000, num_samples=4000)

# Compute the mean and standard deviation of the posterior distribution
alpha_mean, beta_mean, log_sigma_mean = jnp.mean(samples, axis=0)
alpha_std, beta_std, log_sigma_std = jnp.std(samples, axis=0)
sigma_mean = jnp.exp(log_sigma_mean)
sigma_std = jnp.exp(log_sigma_std)

# Print the results
print(f"alpha: {alpha_mean:.2f} +/- {alpha_std:.2f}")
print(f"beta: {beta_mean:.2f} +/- {beta_std:.2f}")
print(f"sigma: {sigma_mean:.2f} +/- {sigma_std:.2f}")

# Plot the posterior distribution of alpha, beta, and sigma
plt.figure(figsize=(12, 4))
plt.subplot(131)
sns.kdeplot(samples[:, 0], label="alpha")
plt.axvline(alpha_true, color='r', linestyle='--', label='True value')
plt.xlabel("alpha")
plt.legend()
plt.subplot(132)
sns.kdeplot(samples[:, 1], label="beta")
plt.axvline(beta_true, color='r', linestyle='--', label='True value')
plt.xlabel("beta")
plt.legend()
plt.subplot(133)
sns.kdeplot(jnp.exp(samples[:, 2]), label="sigma")
plt.axvline(sigma_true, color='r', linestyle='--', label='True value')
plt.xlabel("sigma")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Mean regression line with 95% confidence interval
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, color='blue', label='Observed data')

# Plot the mean regression line
y_mean = alpha_mean + beta_mean * x
y_mean_upper = y_mean + 1.96 * sigma_mean
y_mean_lower = y_mean - 1.96 * sigma_mean
plt.plot(x, y_mean, color='black', label='Mean regression line')
plt.fill_between(x, y_mean_lower, y_mean_upper, color='black', alpha=0.2, label='95% CI')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Mean Regression Line with 95% Confidence Interval')
plt.show()

# Posterior predictive distribution
key = random.PRNGKey(2)

# Function to generate predictions for a single sample
# Define a function to generate predictions for a single sample
def predict_single_sample(params, x, key):
    alpha, beta, log_sigma = params
    mu = alpha + beta * x
    sigma = jnp.exp(log_sigma)
    key, subkey = random.split(key)
    return random.normal(subkey, mu.shape) * sigma + mu

# Vectorize the prediction function
predict_vectorized = jax.vmap(predict_single_sample, in_axes=(0, None, 0))
# Generate predictions for all samples

# Generate a unique key for each prediction
keys = random.split(key, len(samples))

pred_samples = predict_vectorized(samples, x, keys)

# Compute the 89% credible interval
lower_bound = jnp.percentile(pred_samples, 5.5, axis=0)
upper_bound = jnp.percentile(pred_samples, 94.5, axis=0)

# Plot predictive intervals
plt.figure(figsize=(10, 6))
plt.plot(x, y_mean, color='black', label='Mean regression line')
plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label='89% predictive interval')
plt.scatter(x, y, color='blue', label='Observed data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Posterior Predictive Distribution')
plt.show()

# Add diagnostic plots
plt.figure(figsize=(15, 5))
for i, (param, true_value) in enumerate(zip(['alpha', 'beta', 'log_sigma'], [alpha_true, beta_true, jnp.log(sigma_true)])):
    plt.subplot(1, 3, i+1)
    plt.plot(samples[:, i])
    plt.axhline(true_value, color='r', linestyle='--')
    plt.title(f'{param} trace plot')
plt.tight_layout()
plt.show()

# Plot autocorrelation
plt.figure(figsize=(15, 5))
for i, param in enumerate(['alpha', 'beta', 'log_sigma']):
    plt.subplot(1, 3, i+1)
    plt.acorr(samples[:, i] - jnp.mean(samples[:, i]), maxlags=100)
    plt.title(f'{param} autocorrelation')
plt.tight_layout()
plt.show()