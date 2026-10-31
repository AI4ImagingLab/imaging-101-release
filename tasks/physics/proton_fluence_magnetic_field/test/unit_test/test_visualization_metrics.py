import numpy as np
import pytest
from src.visualization import image_metrics,field_metrics

def test_plot_metrics_independently_known_case():
    ref=np.array([[1.,2.],[3.,4.]])
    result=image_metrics(ref+1,ref)
    assert result['nrmse']==1/3
    assert result['ncc']==pytest.approx(40/(np.sqrt(54)*np.sqrt(30)))
    assert result['relative_l2']==pytest.approx(2/np.sqrt(30))
    assert image_metrics(np.zeros_like(ref),ref)['ncc']==0
    fields=field_metrics(np.stack([ref,ref*2]),np.stack([ref,ref*2]))
    assert all(v['nrmse']==0 for v in fields.values())

def test_plot_metrics_reject_degenerate_reference():
    with pytest.raises(ValueError):image_metrics(np.zeros((2,2)),np.zeros((2,2)))
