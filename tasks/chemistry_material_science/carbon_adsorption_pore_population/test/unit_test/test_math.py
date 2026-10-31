from pathlib import Path
import sys,numpy as np
from scipy.interpolate import BSpline,PchipInterpolator,interp1d
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load_observations,calibration_with_origin,width_steps
from src.physics_model import build_kernel,forward,objective,objective_gradient
from src.solvers import spline_matrix,reconstruct
from src.visualization import metrics
def test_full_calibration_positive_shape_preserving_and_exact_nodes():
    raw=load_observations(T/'data/raw_data.npz');p,c=calibration_with_origin(raw['kernel_relative_pressure'],raw['kernel_loading'])
    assert np.min(c)==0 and np.all(np.diff(c,axis=0)>=0)
    f=PchipInterpolator(p,c,axis=0,extrapolate=False)
    np.testing.assert_allclose(f(p),c,atol=1e-12,rtol=1e-14)
    dense=np.r_[0.,np.geomspace(1e-12,p[-1],10000)]
    assert np.min(f(dense))>=0 and np.min(np.diff(f(dense),axis=0))>=-1e-12
    # Knownoriginalcubicovershoot is independently reproduced from publiccalibration.
    cubic=interp1d(p,c,axis=0,kind='cubic')(raw['relative_pressure'])
    assert np.sum(cubic<0)==3
    assert np.min(build_kernel(raw['relative_pressure'],raw['kernel_relative_pressure'],raw['kernel_loading']))>0
def test_forward_linearity_and_analytic_gradient():
    rng=np.random.default_rng(43);K=rng.uniform(.01,5,(7,13));a=rng.uniform(.01,.1,7);y=rng.uniform(.5,2,13)
    np.testing.assert_allclose(forward(2*a,K),2*forward(a,K))
    expected=objective_gradient(a,K,y);numeric=[]
    for j in range(7):
        d=np.zeros(7);d[j]=1e-6;numeric.append((objective(a+d,K,y)-objective(a-d,K,y))/2e-6)
    np.testing.assert_allclose(expected,numeric,rtol=1e-8,atol=1e-8)
def test_spline_basis_independent_and_mass_convention():
    for degree in [2,5]:
        S=spline_matrix(77,degree);knots=np.r_[np.zeros(degree),np.arange(77-degree+1),np.full(degree,77-degree)];u=np.linspace(0,77-degree,100)
        reference=BSpline(knots,np.eye(77),degree)(u)
        np.testing.assert_allclose(S,reference,atol=1e-14)
        assert np.min(S)>=0;np.testing.assert_allclose(S.sum(axis=1),1,atol=1e-14)
    np.testing.assert_array_equal(spline_matrix(4,0),np.eye(4))
    np.testing.assert_allclose(width_steps(np.array([.4,.5,.8])),[.4,.1,.3])
def test_known_metrics():
    a=np.array([1.,2.,4.]);m=metrics(a,a);assert m['nrmse']==0 and abs(m['ncc']-1)<1e-15
    assert abs(metrics(2*a,a)['nrmse']-np.sqrt(7)/3)<1e-15
