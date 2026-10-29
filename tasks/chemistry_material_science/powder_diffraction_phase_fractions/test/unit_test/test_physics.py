import numpy as np
from src.preprocessing import natural_spline
from src.physics_model import forward,rwp,rwp_gradient,finite_gradient,mass_fractions
from src.solvers import nnls_active_set
from src.visualization import composition_rmse,observation_rwp
def test_natural_spline_affine_including_extrapolation():
    x=np.arange(5.); t=np.linspace(-2,7,30)
    np.testing.assert_allclose(natural_spline(x,3*x+2,t),3*t+2,atol=1e-12)
def test_forward_adjoint_and_objective_gradient():
    rng=np.random.default_rng(4); a=rng.uniform(.1,2,(50,5)); x=rng.uniform(.1,1,5); y=rng.uniform(1,3,50); v=rng.normal(size=50)
    np.testing.assert_allclose(forward(a,x)@v,x@(a.T@v),atol=1e-12)
    np.testing.assert_allclose(rwp_gradient(x,a,y),finite_gradient(lambda z:rwp(z,a,y),x,1e-6),rtol=1e-7,atol=1e-9)
def test_nnls_kkt():
    rng=np.random.default_rng(3); a=rng.uniform(0,2,(40,5)); y=a@np.array([1.,0.,.3,0.,.4])+rng.normal(0,.01,40)
    x=nnls_active_set(a,y); gradient=a.T@(a@x-y)
    assert np.all(x>=0) and np.min(gradient)>-1e-8 and np.max(abs(x*gradient))<1e-8
def test_known_internal_standard_ratios():
    rir=np.array([1.,2.,.5]); x=np.array([.4,.8,.2]); np.testing.assert_array_equal(mass_fractions(x,rir),[20,20,20])
def test_metrics_known_cases():
    a=np.ones((2,7)); b=a.copy(); b[:,1:]+=2
    np.testing.assert_array_equal(composition_rmse(b,a),[2,2])
    np.testing.assert_array_equal(observation_rwp(np.full((2,5),4.),np.full((2,5),2.)),[1,1])

def test_public_acquisition_constants():
    from pathlib import Path
    from src.preprocessing import load_data
    root=Path(__file__).resolve().parents[2]
    data=load_data(root/'data/raw_data.npz')
    readme=(root/'README.md').read_text(encoding='utf8')
    np.testing.assert_allclose(data['angles'],np.linspace(5,65,3001),rtol=0,atol=1e-12)
    assert data['counts'].shape==(8,3001) and data['standards'].shape==(3001,7)
    assert data['rir'][0]==1
    assert '5 to65 inclusive;0.02' in readme and 'Corundum20' in readme
    for phase in ['CORUNDUM','QUARTZ','ORDERED_MICROCLINE','LABRADORITE','KAOLINITE_DRY_BRANCH','ILLITE_1M_RM30','MONTMORILLONITE_WYO']:
        assert phase in readme
