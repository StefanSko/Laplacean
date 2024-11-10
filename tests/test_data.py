import pytest
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


def test_index_validations():
    # Test empty indices
    with pytest.raises(ValueError, match="Indices tuple cannot be empty"):
        Index(())
    
    # Test invalid index type
    with pytest.raises(TypeError, match="All indices must be slice objects"):
        Index((1,))  # trying to pass an int instead of slice
    
    # Test negative start index
    with pytest.raises(ValueError, match="Start index cannot be negative"):
        Index.vector(start=-1)
    
    # Test invalid end index
    with pytest.raises(ValueError, match="End index must be greater than start index"):
        Index.vector(start=5, end=3)
    
    # Test array dimension mismatch
    test_index = Index((slice(0, 1), slice(0, 1)))  # 2D index
    test_array = jnp.array([1, 2, 3])  # 1D array
    with pytest.raises(ValueError, match="Too many indices .* for array"):
        test_index.select(test_array)


def test_random_var_validations():
    test_array = jnp.array([1, 2, 3])
    valid_index = Index.vector(0, 2)
    
    # Test empty name
    with pytest.raises(ValueError, match="Name cannot be empty"):
        RandomVar.from_index("", valid_index, test_array)
