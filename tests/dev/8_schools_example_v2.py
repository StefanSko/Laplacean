import jax.numpy as jnp
import jax.random as random
from base.data import JaxHMCData
from methods.bayesian_network_v2 import (
    ModelBuilder, bind_data, get_initial_params
)
from methods.hmc import step
from sampler.sampling import Sampler

# Data for the 8 schools problem
J = jnp.array(8)  # number of schools
y = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])  # observed effects
sigma = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])  # standard deviations

# Create model using Stan-like interface
model = (ModelBuilder()
         .data()
         .int_scalar("J", J)
         .vector("y", "J")
         .vector("sigma", "J")
         .done()
         .parameters()
         .real("mu")  # population mean
         .real("tau")  # population standard deviation
         .vector("theta", "J")  # school effects
         .done()
         .model()
         # Priors
         .normal("mu", 0, 10)  # weakly informative prior for mean
         .exponential("tau", 1)  # weakly informative prior for scale
         # Likelihood
         .normal("theta", "mu", "tau")  # population distribution
         .normal("y", "theta", "sigma")  # observed data
         .done()
         .build())

# Bind data to model
model = bind_data(model, {
    "y": y,
    "sigma": sigma
})

# Initialize HMC sampler
initial_params = get_initial_params(model, random_key=random.PRNGKey(0))
input_data = JaxHMCData(
    epsilon=0.005,
    L=12,
    current_q=initial_params,
    key=random.PRNGKey(1)
)

# Create and run the sampler
sampler = Sampler()
samples = sampler(step, input_data, model, num_warmup=500, num_samples=1000)

# Extract and analyze results
mu_samples = samples[:, 0]
tau_samples = samples[:, 1]
theta_samples = samples[:, 2:10]

# Print summary statistics
print("\nPosterior Summary:")
print(f"mu: mean = {jnp.mean(mu_samples):.2f}, std = {jnp.std(mu_samples):.2f}")
print(f"tau: mean = {jnp.mean(tau_samples):.2f}, std = {jnp.std(tau_samples):.2f}")
print("\nSchool Effects (theta):")
for i in range(J):
    mean = jnp.mean(theta_samples[:, i])
    std = jnp.std(theta_samples[:, i])
    print(f"School {i + 1}: mean = {mean:.2f}, std = {std:.2f}")

# Optional: Add visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot traces
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(mu_samples)
    ax1.set_title('Trace of μ')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('μ')

    ax2.plot(tau_samples)
    ax2.set_title('Trace of τ')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('τ')

    plt.tight_layout()
    plt.show()

    # Plot posterior distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(mu_samples, ax=ax1, kde=True)
    ax1.set_title('Posterior distribution of μ')
    ax1.set_xlabel('μ')

    sns.histplot(tau_samples, ax=ax2, kde=True)
    ax2.set_title('Posterior distribution of τ')
    ax2.set_xlabel('τ')

    plt.tight_layout()
    plt.show()

except ImportError:
    print("Matplotlib and/or seaborn not available for plotting")