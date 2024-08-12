from typing import Optional

from jaxtyping import Array
import jax

from base.functions import PriorFunc, LikelihoodFunc, PotentialEnergyFunc


def create_potential_energy(prior_func: PriorFunc, likelihood_func: LikelihoodFunc, data: Optional[Array]) -> PotentialEnergyFunc:
    def potential_energy(q):
        prior = prior_func(q)
        likelihood = likelihood_func(q, data)
        return prior + likelihood
    return potential_energy

def noop_likelihood(q: Array, data: Array) -> float:
    return 0.0


class PotentialEnergy:
    def __init__(self, prior_func: PriorFunc, likelihood_func: Optional[LikelihoodFunc] = None, data: Optional[Array] = None):
        if likelihood_func is None or data is None:
            likelihood_func = noop_likelihood
            data = None
        self.potential_energy = create_potential_energy(prior_func, likelihood_func, data)
    
    def __call__(self, q: Array) -> float:
        return self.potential_energy(q)
    def gradient(self, q: Array) -> Array:
        return jax.grad(self.__call__)(q)
    
