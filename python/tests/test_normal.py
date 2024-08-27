import pytest
from bvhar.random import normal
import numpy as np

def test_generate_mnormal():
    num_sim = 10
    mean = np.array([1.0, 2.0, 3.0])
    covariance = np.array([[1.0, .5, .3],
                           [.5, 2.0, .7],
                           [.3, .7, 3.0]])
    seed = 2
    method = 2
    result = normal.generate_mnormal(num_sim, mean, covariance, seed, method)

    assert result.shape == (num_sim, 3)

def test_generate_mnormal_invalid():
    num_sim = 10
    mean = np.array([1.0, 2.0, 3.0])
    covariance = np.array([[1.0, .5],
                           [.5, 2.0]])
    seed = 2
    method = 2
    with pytest.raises(ValueError, match="Invalid 'mean' size"):
        normal.generate_mnormal(num_sim, mean, covariance, seed, method)
