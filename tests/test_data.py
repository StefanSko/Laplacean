import pytest
import jax.numpy as jnp


from base.data import Index, RandomVar, make_var_provider


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
    var_provider = make_var_provider(index=test_index)
    random_var: RandomVar = RandomVar(name='mu', shape = test_index.get_shape(), provider = var_provider)
    assert random_var.name == 'mu'
    assert jnp.allclose(random_var.get_value(test_nums), test_nums[:10], atol=1e-6)


def test_view_on_random_var():
    len = 10
    test_index_1: Index = Index.vector(start=0, end=len)
    var_provider_1 = make_var_provider(index=test_index_1)
    test_index_2: Index = Index.vector(start=len, end=2*len)
    var_provider_2 = make_var_provider(index=test_index_2)
    test_nums: jnp.array = jnp.array([num for num in range(0, len+10)])
    random_var_1 = RandomVar(name='mu_1', shape = test_index_1.get_shape(), provider = var_provider_1)
    random_var_2 = RandomVar(name='mu_2', shape = test_index_2.get_shape(), provider = var_provider_2)
    assert jnp.allclose(random_var_1.get_value(test_nums), test_nums[:10], atol=1e-6)
    assert jnp.allclose(random_var_2.get_value(test_nums), test_nums[10:], atol=1e-6)


def test_index_validations():
    
    # Test negative start index
    with pytest.raises(ValueError, match="Start index cannot be negative"):
        Index.vector(start=-1)
    
    # Test invalid end index
    with pytest.raises(ValueError, match="End index must be greater than start index"):
        Index.vector(start=5, end=3)

    # Test basic array indexing
    test_index = Index(slice(0, 1))
    test_array = jnp.array([1, 2, 3])
    selected = test_index.select(test_array)
    assert jnp.allclose(selected, jnp.array([1]))


