import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import seaborn as sns
from matplotlib import pyplot as plt

from laplacean.backend.jax_hmc import HMCProtocol, JaxHMC, JaxHMCData

# Generate some observed data
n = 300
x = jnp.linspace(0, 1, n)
alpha_true = 1.0
beta_true = 2.0
sigma_true = 0.5
epsilon = sigma_true * random.normal(random.PRNGKey(0), (n,))
y = alpha_true + beta_true * x + epsilon

# Define the potential energy function
def U(params: Array) -> float:
    alpha, beta, log_sigma = params
    sigma = jnp.exp(log_sigma)
    y_pred = alpha + beta * x
    log_likelihood = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * jnp.sum((y - y_pred)**2) / sigma**2
    log_prior_alpha = -0.5 * alpha**2
    log_prior_beta = -0.5 * beta**2
    log_prior_sigma = -sigma
    return -(log_likelihood + log_prior_alpha + log_prior_beta + log_prior_sigma)

# Define the gradient of the potential energy function
grad_U = jax.grad(U)

# Initialize the HMC sampler
initial_params = jnp.array([0.0, 0.0, jnp.log(0.5)])
hmc: HMCProtocol = JaxHMC(U=U, grad_U=grad_U)
input: JaxHMCData = JaxHMCData(epsilon=0.01, L=12, current_q=initial_params, key=random.PRNGKey(1))

# Draw samples from the posterior distribution
samples = hmc.run_hmc(input, 4000, 500)

# Compute the mean and standard deviation of the posterior distribution
alpha_mean, beta_mean, log_sigma_mean = jnp.mean(samples, axis=0)
alpha_std, beta_std, log_sigma_std = jnp.std(samples, axis=0)
sigma_mean = jnp.exp(log_sigma_mean)
sigma_std = jnp.exp(log_sigma_std)

# Print the results
print(f"alpha: {alpha_mean:.2f} +/- {alpha_std:.2f}")
print(f"beta: {beta_mean:.2f} +/- {beta_std:.2f}")
print(f"sigma: {sigma_mean:.2f} +/- {sigma_std:.2f}")

# Plot the posterior distribution of alpha and beta
sns.kdeplot(samples[:, 0], label="alpha")
sns.kdeplot(samples[:, 1], label="beta")
sns.kdeplot(jnp.exp(samples[:, 2]), label="sigma")
plt.xlabel("Parameter value")
plt.ylabel("Density")
plt.title("Posterior distribution of alpha and beta")
plt.legend()
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
def predict_single_sample(sample, x, key):
    alpha_sample, beta_sample, log_sigma_sample = sample
    sigma_sample = jnp.exp(log_sigma_sample)
    key, subkey = random.split(key)
    noise = sigma_sample * random.normal(subkey, shape=(len(x),))
    return alpha_sample + beta_sample * x + noise

# Vectorize the prediction function
predict_vectorized = jax.vmap(predict_single_sample, in_axes=(0, None, None))
# Generate predictions for all samples
pred_samples = predict_vectorized(samples, x, key)

# Compute the 89% credible interval
lower_bound = jnp.percentile(pred_samples, 5.5, axis=0)
upper_bound = jnp.percentile(pred_samples, 94.5, axis=0)


# Plot predictive intervals
plt.figure(figsize=(10, 6))
plt.plot(x, y_mean, color='black', label='Mean regression line')
plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label='95% predictive interval')
plt.scatter(x, y, color='blue', label='Observed data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Posterior Predictive Distribution')
plt.show()