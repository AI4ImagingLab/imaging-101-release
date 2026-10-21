import numpy as np
import pytest
from src.visualization import ncc,nrmse
def test_known_review_metrics_match_definitions():
    assert ncc(np.array([0.,2.]),np.array([0.,1.]))==1
    assert ncc(np.zeros(2),np.array([0.,1.]))==0
    assert nrmse(np.array([1.,3.]),np.array([0.,2.]))==.5
    with pytest.raises(ValueError):ncc(np.ones(2),np.zeros(2))
    with pytest.raises(ValueError):nrmse(np.ones(2),np.ones(2))
