import importlib.util
from pathlib import Path
import numpy as np
from src.visualization import field_metrics
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_for_metrics',T/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
def test_independent_metric_implementations_known_nonzero_error():
 y=np.tile(np.array([[1.,2.],[3.,4.]]),(10,10));x=np.tile(np.array([[4.,3.],[2.,1.]]),(10,10));m=np.ones(y.shape,bool)
 a=field_metrics(x,y,m);b=J.aligned_metrics(x,y,m)
 for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-14,atol=1e-14)
 np.testing.assert_allclose(a['source_lmse'],1-(20/30)**2,atol=1e-14)
