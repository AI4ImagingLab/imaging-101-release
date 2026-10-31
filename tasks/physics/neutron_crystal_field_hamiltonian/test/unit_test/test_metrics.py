import numpy as np
from src.visualization import metrics

def test_metric_known_nonconstant_case():
    y=np.array([[1.,3.]]*3);data=dict(intensity=y,reported_error=np.ones((3,2)),fit_mask=np.ones((3,2),bool),temperature=np.array([1.8,15,100]))
    estimated=y.copy();estimated[0]=[2.,4.]
    prediction=dict(neutron_spectrum=np.stack([estimated,y])[None])
    rows=metrics(prediction,data);first=rows[0]
    assert first['nrmse']==.5
    assert abs(first['ncc']-14/(np.sqrt(20)*np.sqrt(10)))<1e-15
    assert first['chi2']==2.
    assert rows[3]['total_chi2']==2.
    assert rows[3]['conditional_chi2_per_dof']==2/274
    assert rows[3]['source_display_chi2']==2/524
