import numpy as np
from src.visualization import summary


def test_known_rank_energy_and_signed_sparse_fraction():
    first=np.array([[1.,2.],[3.,4.]]);low=np.array([first,2*first])[None]
    sparse=np.zeros_like(low);sparse[0,1,0,0]=-.002
    result=summary({'reconstruction':low,'sparse_component':sparse},{'iterations':7,'converged':True,'relative_primal':1e-8,'lambda':.1})
    assert result['rank_1e_6']==1
    assert result['sparse_fraction_1e_3']==.125
    assert result['low_energy']==150.
    np.testing.assert_allclose(result['sparse_energy'],4e-6)
