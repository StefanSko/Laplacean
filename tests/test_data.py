import pytest
import jax.numpy as jnp


from base.data import Index, RandomVar, RandomVarFactory, Parameter, make_parameter_selector


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
    random_var: RandomVar = RandomVarFactory.from_parameter(name='mu', index=test_index)
    assert random_var.name == 'mu'
    assert jnp.allclose(random_var.get_value(test_nums), test_nums[:10], atol=1e-6)


def test_view_on_random_var():
    len = 10
    test_index_1: Index = Index.vector(start=0, end=len)
    test_index_2: Index = Index.vector(start=len, end=2*len)
    test_nums: jnp.array = jnp.array([num for num in range(0, len+10)])
    random_var_1: Parameter = RandomVarFactory.from_parameter(name='mu_1', index=test_index_1)
    random_var_2: Parameter = RandomVarFactory.from_parameter(name='mu_2', index=test_index_2)
    #TODO How to make mypy trigger a warning when no array is passed?
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


def test_random_var_validations():
    valid_index = Index.vector(0, 2)
    
    # Test empty name
    with pytest.raises(ValueError, match="Name cannot be empty"):
        RandomVar("", (2,), make_parameter_selector(valid_index), "parameter")
    
    # Test empty shape
    with pytest.raises(ValueError, match="Shape tuple cannot be empty"):
        RandomVar("test", (), make_parameter_selector(valid_index), "parameter")
    
    # Test non-positive dimensions
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        RandomVar("test", (0,), make_parameter_selector(valid_index), "parameter")


def test_random_var_from_data():
    # Test with 1D array
    test_data_1d = jnp.array([1.0, 2.0, 3.0])
    random_var_1d = RandomVarFactory.from_data(name='data_1d', data=test_data_1d)
    assert random_var_1d.name == 'data_1d'
    assert random_var_1d.shape == (3,)
    assert random_var_1d.var_kind == "observed"
    assert jnp.allclose(random_var_1d.get_value(), test_data_1d)
    # Verify state is ignored for observed variables
    assert jnp.allclose(random_var_1d.get_value(jnp.array([9.9, 9.9])), test_data_1d)

    # Test with 2D array
    test_data_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    random_var_2d = RandomVarFactory.from_data(name='data_2d', data=test_data_2d)
    assert random_var_2d.name == 'data_2d'
    assert random_var_2d.shape == (2, 2)
    assert random_var_2d.var_kind == "observed"
    assert jnp.allclose(random_var_2d.get_value(), test_data_2d)

    # Test with empty array (should raise error due to shape validation)
    empty_data = jnp.array([])
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        RandomVarFactory.from_data(name='empty_data', data=empty_data)
