import numpy as np
from typing import Callable
from numpy import typing as npt


import seaborn as sns
import matplotlib.pyplot as plt


PotentialFn = Callable[[npt.ArrayLike], float]
GradientFn = Callable[[npt.ArrayLike], npt.ArrayLike]

def hmc(U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, current_q: npt.ArrayLike) -> npt.ArrayLike:
    q = np.array(current_q)
    p = np.random.normal(size=q.shape)
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2
    
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p
        # Make a full step for the momentum, except at the end of trajectory
        if i != L - 1:
            p = p - epsilon * grad_U(q)
    
    # Make a half step for momentum at the end
    p = p - epsilon * grad_U(q) / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    #Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = np.sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = np.sum(p ** 2) / 2
    # Accept or reject the state at the end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    if np.random.uniform() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q
    else:
        return current_q
    

def run_hmc(U: PotentialFn, grad_U: GradientFn, epsilon: float, L: int, initial_q: npt.ArrayLike, num_samples: int) -> npt.ArrayLike:
    
    #burn-in phase (optional)
    burn_in_steps = 500
    for _ in range(burn_in_steps):
        initial_q = hmc(U, grad_U, epsilon, L, initial_q)
    
    samples = np.zeros((num_samples, *initial_q.shape))
    for i in range(num_samples):
        q = hmc(U, grad_U, epsilon, L, initial_q)
        samples[i] = q
    return samples

# Example usage for potential energy function corresponding to a standard normal distribution with mean 0 and variance 1
# U needs to return the negative log of the density
def U(q: npt.ArrayLike) -> float:
    return 0.5 * np.sum(q ** 2)

def grad_U(q: npt.ArrayLike) -> npt.ArrayLike:
    return q


initial_q = np.array([1])

epsilon = 0.2
L = 11
num_samples = 50000

samples = run_hmc(U, grad_U, epsilon, L, initial_q, num_samples)
print(samples.shape)
print(samples.mean())
print(samples.std())

sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()






    

