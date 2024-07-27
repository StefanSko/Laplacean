import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import seaborn as sns
from matplotlib import pyplot as plt

from laplacean.backend.jax_hmc import HMCProtocol, JaxHMC, JaxHMCData

# Generate some observed data
n = 100
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
    log_prior_alpha = -0.5 * alpha**2 / 10.0
    log_prior_beta = -0.5 * beta**2 / 10.0
    log_prior_sigma = 2 * jnp.log(sigma) - 0.5 * sigma**2 / 0.5
    return -(log_likelihood + log_prior_alpha + log_prior_beta + log_prior_sigma)

# Define the gradient of the potential energy function
grad_U = jax.grad(U)

# Initialize the HMC sampler
initial_params = jnp.array([0.0, 0.0, 0.0])
hmc: HMCProtocol = JaxHMC(U=U, grad_U=grad_U)
input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_params, key=random.PRNGKey(1))

# Draw samples from the posterior distribution
samples = hmc.run_hmc(input, 1000)

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
sns.kdeplot(samples[:, 2], label="sigma")
plt.xlabel("Parameter value")
plt.ylabel("Density")
plt.title("Posterior distribution of alpha and beta")
plt.legend()
plt.show()