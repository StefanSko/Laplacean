import jax.numpy as jnp
import jax.random as random
import seaborn as sns
from matplotlib import pyplot as plt

from logging_utils import logger
from methods.hmc import step
from sampler.sampling import Sampler
from base.data import JaxHMCData
from methods.bayesian_execution_network import (
    BayesianExecutionModel, create_prior_node, QueryPlan, normal_prior, SingleParam
)

initial_q = jnp.array([1.])

input: JaxHMCData = JaxHMCData(epsilon=0.1, L=10, current_q=initial_q, key=random.PRNGKey(0))
sampler = Sampler()

def f_mean(x):
    return jnp.array(0.0)

def f_std(x):
    return jnp.array(1.0)

normal = normal_prior(f_mean, f_std)

# Create a prior node
# Note: We now need to provide both a node_id and a ParamIndex
prior_node = create_prior_node(0, SingleParam(0), normal)

# Create a query plan
query_plan = QueryPlan([prior_node])

# Create a BayesianExecutionModel
model = BayesianExecutionModel(query_plan)

samples = sampler(step, input, model)

mean = jnp.mean(samples)
var = jnp.var(samples)

logger.info(f"mean = {mean}")
logger.info(f"var = {var}")

sns.kdeplot(samples)
plt.xlabel('Sample Value')
plt.title('Distribution of Samples')
plt.show()

sns.lineplot(samples)
plt.xlabel('Step')
plt.ylabel('Sample Value')
plt.title('Trace of Samples')
plt.show()
