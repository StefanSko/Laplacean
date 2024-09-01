from typing import Callable
from jax import Array

# Define the base
PriorFunc = Callable[[Array], float]
PotentialEnergyFunc = Callable[[Array], float]
LikelihoodFunc = Callable[[Array, Array], float]