from pathlib import Path
import numpy as np
from src.main import reconstruct
from src.preprocessing import fit_circle,bond_number,prepare
from src.physics_model import YoungLaplaceSolution
from src.solvers import jacobian_residual,solve
ROOT=Path(__file__).resolve().parents[2]
def test_native_stages_and_pipeline():
    _,contour,r=reconstruct(ROOT/'data/raw_data.npz');points=prepare(contour)
    with np.load(ROOT/'test/fixtures/native.npz') as q:
        ini=fit_circle(points);actual=np.r_[ini,bond_number(points,*ini),0.]
        np.testing.assert_allclose(actual,q['initial'],atol=1e-8,rtol=0)
        # Equal-height quicksort ties differ between NumPy1.26 and2.4.
        np.testing.assert_array_equal(points[np.lexsort(points.T)],q['sorted_profile'][np.lexsort(q['sorted_profile'].T)])
        np.testing.assert_allclose(YoungLaplaceSolution(.2,190)(q['arclength']),q['ode'],atol=1e-7,rtol=0)
        assert np.max(abs((r['parameters']-q['parameters'])/[1,1,1,.01,.01]))<=1e-3
        np.testing.assert_allclose(r['reconstruction'],q['profile'],atol=.002,rtol=0)
        p=q['parameters'].copy();p[[1,4]]*=-1
        J,res,_=jacobian_residual(q['sorted_profile'],p,float(q['profile_size']))
        np.testing.assert_allclose(J,q['jacobian'],atol=1e-4,rtol=0)
        assert r['history'].shape==(5,5) and r['warnings']==0 and r['reason']==1
def test_spherical_limit_and_bond_sensitivity():
    # A sphere closes at pi; do not integrate its r=0 singularity to the
    # production cache's s=4, which is valid only for the bound pendant shape.
    s=np.linspace(.01,2.7,101);v=YoungLaplaceSolution._solve(0.,2.7)(s)
    np.testing.assert_allclose(v[:,0],np.sin(s),atol=1e-4,rtol=0)
    np.testing.assert_allclose(v[:,1],1-np.cos(s),atol=1e-4,rtol=0)
    h=1e-4;v=YoungLaplaceSolution(.2,190)(s)
    fd=(YoungLaplaceSolution(.2+h,190)(s)[:,:3]-YoungLaplaceSolution(.2-h,190)(s)[:,:3])/(2*h)
    np.testing.assert_allclose(fd,v[:,3:],atol=1e-3,rtol=1e-3)
def test_geometric_jacobian():
    _,contour,r=reconstruct(ROOT/'data/raw_data.npz');points=prepare(contour);p=r['parameters'].copy();p[[1,4]]*=-1
    J,res,_=jacobian_residual(points,p,r['profile_size'])
    for j,h in enumerate([.001,.001,.001,1e-5,1e-5]):
        a=p.copy();b=p.copy();a[j]+=h;b[j]-=h
        ea=jacobian_residual(points,a,r['profile_size'])[1][:,1];eb=jacobian_residual(points,b,r['profile_size'])[1][:,1]
        # Native closest-arclength tolerance is1e-6 (~.00019px). Relative
        # normal direction is unstable at nearzero residual; test meaningful
        # finite differences on actual measurements >.05px from the curve.
        mask=abs(res[:,1])>.05
        assert mask.sum()>250
        np.testing.assert_allclose(((ea-eb)/(2*h))[mask],J[mask,j],atol=.015,rtol=.002)
def test_translation_covariance():
    _,contour,r=reconstruct(ROOT/'data/raw_data.npz');shift=np.array([10.,-15.]);s=solve(contour+shift)
    np.testing.assert_allclose(s['parameters'][:2],r['parameters'][:2]+shift,atol=.001)
    np.testing.assert_allclose(s['parameters'][2:],r['parameters'][2:],atol=.0001)
