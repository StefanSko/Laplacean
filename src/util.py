
import equinox as eqx

def conditional_jit(use_jit=True):
    def decorator(func):
        if use_jit:
            return eqx.filter_jit(func)
        return func
    return decorator