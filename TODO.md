- refactor jax_hmc.py 
  - the sampling is independent of the actual algorithm for the generating the proposed step
  - factor that out into a base class that runs the sampling and takes the algorith (e.g. hmc) as an argument for that
    - see turing.jl api as an example for that
- check how to be able to jax.jit the gradient and requirements for that