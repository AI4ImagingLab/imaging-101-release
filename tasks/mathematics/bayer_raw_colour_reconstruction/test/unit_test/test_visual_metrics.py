import numpy as np
from src.visualization import compute_metrics

def test_known_visual_metric_definition():
    assert compute_metrics(np.array([1.,0]),np.array([0.,1]))=={'ncc':0.,'nrmse':1.}
    a=np.array([0.,2.,1.]);b=np.array([0.,1.,2.])
    m=compute_metrics(a,b)
    np.testing.assert_allclose([m['ncc'],m['nrmse']],[.8,np.sqrt(2/3)/2],rtol=1e-14)
