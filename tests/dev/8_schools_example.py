import jax.numpy as jnp
import jax.random as random


from base.data import JaxHMCData
from methods.bayesian_execution_network import create_prior_node, SingleParam, normal_prior, exponential_prior, \
    ParamVector, create_likelihood_node, normal_likelihood, ParamFunction, QueryPlan, BayesianExecutionModel, \
    IdentityParam, bind_data
from methods.hmc import step
from sampler.sampling import Sampler

num_schools = 8
observed_effects = [28, 8, -3, 7, -1, 1, 18, 12]
stddevs = [15, 10, 16, 11, 9, 11, 10, 18]

mu = create_prior_node(0, SingleParam(0), normal_prior(
    lambda _: jnp.array(0.0), 
    lambda _: jnp.array(10.0))
    )
tau =  create_prior_node(1, SingleParam(1), exponential_prior(
    lambda _: jnp.array(1.0)))
theta_i = create_prior_node(
        node_id=3,
        param_index=ParamVector(start=2, end=2+num_schools),
        log_density=normal_prior(
            mean=lambda params: params[0],  # μ
            std=lambda params: params[1]    # τ
        )
    )
y = create_likelihood_node(
        node_id=4,
        param_index=ParamVector(start=2, end=2+num_schools),
        log_likelihood=normal_likelihood(
            mean=ParamFunction(
                func=lambda params, _: params,
                param_index=ParamVector(start=0, end=num_schools)
            ),
            std=ParamFunction(
                func = lambda _, data: jnp.array(stddevs),
                param_index= IdentityParam()
            )
        )
    )

# Initialize the HMC sampler
initial_params = jnp.zeros(num_schools+2)
input_data = JaxHMCData(epsilon=0.005, L=12, current_q=initial_params, key=random.PRNGKey(1))


# Create query plan
query_plan = QueryPlan([mu, tau, theta_i, y])

query_plan = bind_data(4, jnp.array(observed_effects), query_plan)
model = BayesianExecutionModel(query_plan)

# Create and run the sampler
sampler = Sampler()
samples = sampler(step, input_data, model, num_warmup=3, num_samples=5)

#TODO FIX
#jax.debug.print(mean) -> [0. 0. 0. 0. 0. 0. 0. 0.]
#jax.debug.print(std) -> [15 10 16 11  9 11 10 18]
#jax.debug.print(mean) -> [nan nan nan nan nan nan nan nan]
#jax.debug.print(std) -> [15 10 16 11  9 11 10 18]