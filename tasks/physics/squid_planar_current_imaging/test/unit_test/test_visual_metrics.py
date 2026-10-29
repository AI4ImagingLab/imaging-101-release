import numpy as np
import pytest
from src.visualization import metrics

def test_independent_known_component_metric_cases():
    reference=np.array([[0.,1.],[0.,2.]])
    scores=metrics(reference*2,reference)
    assert scores=={'ncc_jx':1.,'relative_l2_jx':1.,'ncc_jy':1.,'relative_l2_jy':1.}
    zero=metrics(np.zeros_like(reference),reference)
    assert zero['ncc_jx']==0 and zero['relative_l2_jy']==1
    with pytest.raises(ValueError):metrics(reference,np.zeros_like(reference))
