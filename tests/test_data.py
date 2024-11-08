import jax.numpy as jnp


from base.data import Index, RandomVar


def test_single_index():
    test_index: Index = Index.single(0)
    test_nums: jnp.array = jnp.array([num for num in range(0,20)])
    assert jnp.allclose(test_index.select(test_nums), jnp.array(0.0), atol=1e-6)
    test_index_2: Index = Index.single(5)
    assert jnp.allclose(test_index_2.select(test_nums), jnp.array(5.0), atol=1e-6)


def test_vector_index():
    test_index: Index = Index.vector(start=0, end=10)
    test_nums: jnp.array = jnp.array([num for num in range(0, 20)])
    assert jnp.allclose(test_index.select(test_nums), test_nums[:10], atol=1e-6)


def test_random_var():
    len = 10
    test_index: Index = Index.vector(start=0, end=len)
    test_nums: jnp.array = jnp.array([num for num in range(0, len+10)])
    random_var: RandomVar = RandomVar.from_index(name='mu', index=test_index, vec=test_nums)
    assert random_var.name == 'mu'
    assert jnp.allclose(random_var.get_value(), test_nums[:10], atol=1e-6)