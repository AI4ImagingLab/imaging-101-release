import numpy as np
import pytest
from src.visualization import comparison_metrics

def test_presentation_metrics_known_case():
    actual=comparison_metrics(np.array([1.,2.,3.]),np.array([2.,2.,4.]))
    assert actual['ncc']==pytest.approx(18/np.sqrt(336))
    assert actual['nrmse']==pytest.approx(np.sqrt(2/3)/2)
    assert actual['max_abs_mpa']==1.

def test_zero_estimate_and_exact_identity_metrics():
    x=np.array([-2.,1.,5.])
    assert comparison_metrics(x,x)==dict(ncc=1.,nrmse=0.,max_abs_mpa=0.)
    assert comparison_metrics(x*0,x)['ncc']==0.

def test_presentation_metrics_reject_degenerate_reference():
    with pytest.raises(ValueError):comparison_metrics(np.arange(3),np.ones(3))
