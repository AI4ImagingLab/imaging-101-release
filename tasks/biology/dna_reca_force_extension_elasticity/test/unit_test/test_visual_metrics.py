import numpy as np
from src.visualization import rmse,local_covariance
def test_known_rmse():
    assert rmse(np.array([1.,3.]),np.array([0.,0.]))==np.sqrt(5)
def test_known_covariance():
    np.testing.assert_array_equal(local_covariance(np.vstack([np.eye(4),np.eye(4)]),4.,8),np.eye(4))
