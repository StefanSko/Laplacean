import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import seaborn as sns
from matplotlib import pyplot as plt

from laplacean.backend.jax_hmc import HMCProtocol, JaxHMC, JaxHMCData
from potential_energy import PotentialEnergy


# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
@jax.jit
def U(q: Array) -> Array:
    return 0.5 * jnp.sum(q ** 2)


initial_q = jnp.array([1.])

potential_energy: PotentialEnergy = PotentialEnergy(prior_func=U)

hmc: HMCProtocol = JaxHMC(U=potential_energy)
input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
samples = hmc.run_hmc(input, 1000, 500)

print(jnp.mean(samples))
print(jnp.var(samples))


sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()