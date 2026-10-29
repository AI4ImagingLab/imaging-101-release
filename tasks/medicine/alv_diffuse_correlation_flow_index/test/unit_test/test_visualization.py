import numpy as np
from src.visualization import curve_rmse
def test_known_masked_rmse():
    np.testing.assert_array_equal(curve_rmse(np.array([[3.,99.],[4.,5.]]),np.array([[1.,0.],[2.,3.]]),np.array([[True,False],[True,True]])),[2.,2.])
