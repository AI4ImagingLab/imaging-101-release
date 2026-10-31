import numpy as np
from src.physics_model import internal_states
def test_threshold_zero_has_four_exponential_components():
 q=np.array([50.,20.,.01,1000.,500.,100.,10.,60.,70.,80.,90.,0.]);eps=np.linspace(0,.01,20)
 a,k,s=internal_states(q,eps);expected=np.sqrt(2/3)*q[7:11][None,:]*(1-np.exp(-eps[:,None]*q[3:7]));assert np.allclose(a,expected,atol=1e-12)
 assert np.allclose(s,np.sqrt(1.5)*expected.sum(1)+k,atol=1e-12)
 assert np.all(a[0]==0) and k[0]==0 and s[0]==0
