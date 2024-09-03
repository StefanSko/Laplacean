
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import equinox as eqx

import abc
import functools

from util import conditional_jit


class LogDensity(eqx.Module):
    @abc.abstractmethod
    def __call__(self, q: Array) -> Float[Array, ""]:  # noqa: F722
        pass


class LaplaceanPotentialEnergy(eqx.Module):
    log_prior: LogDensity
    log_likelihood: LogDensity
    
    def __init__(self, log_prior: LogDensity, log_likelihood: LogDensity):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood

    @conditional_jit(use_jit=False)
    def __call__(self, q: Array) -> Float[Array, ""]:  # noqa: F722
        return self.log_prior(q) + self.log_likelihood(q)

    @conditional_jit(use_jit=False)
    def gradient(self, q: Array) -> Array:
        return jax.grad(self.__call__)(q)

class ConstantLogDensity(LogDensity):
    @conditional_jit(use_jit=False)
    def __call__(self, _: Array) -> Float[Array, ""]:  # noqa: F722
        return jnp.array(0.0)

class GaussianLogDensity(LogDensity):

    mean: Array
    var: Array

    def __init__(self, mean: Array, var: Array):
        self.mean = mean
        self.var = var

    @conditional_jit(use_jit=False)
    def __call__(self, q: Array) -> Float[Array, ""]:  # noqa: F722
        return jnp.sum(-0.5 * ((q - self.mean) / self.var) ** 2)