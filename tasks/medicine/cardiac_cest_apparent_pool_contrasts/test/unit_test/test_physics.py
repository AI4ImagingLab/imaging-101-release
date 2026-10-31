import numpy as np
from src.physics_model import lorentzian,lorentzian_jacobian,predict

def test_lorentzian_limits_symmetry_width():
    a,w,c=.2,1.6,3.5
    np.testing.assert_allclose(lorentzian(np.array([c,c+w/2,c-w/2]),a,w,c),[a,a/2,a/2])
    assert lorentzian(1e10,a,w,c)<1e-20

def test_independent_jacobian():
    x=np.linspace(-6,6,31);p=np.array([.2,1.6,3.5]);h=1e-6
    expected=np.column_stack([(lorentzian(x,*(p+np.eye(3)[k]*h))-lorentzian(x,*(p-np.eye(3)[k]*h)))/(2*h) for k in range(3)])
    np.testing.assert_allclose(lorentzian_jacobian(x,*p),expected,atol=1e-10,rtol=1e-6)

def test_shift_gauge_and_zero_saturation():
    p=np.tile([.1,1.,0.],(6,1));x=np.array([-1.,0.,2.])
    np.testing.assert_allclose(predict(x,p,.3),predict(x-.3,p,0))
    p[:,0]=0;np.testing.assert_array_equal(predict(x,p,0),np.ones(3))

def test_backend_import():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before

def test_bound_activity_respects_narrow_frequency_gauge():
    from src.visualization import bound_activity
    lower=np.array([0.,-1e-6]);upper=np.array([1.,1e-6])
    p=np.array([[0.,-1e-6],[.5,0.],[1.,1e-6]])
    np.testing.assert_allclose(bound_activity(p,np.ones(3,dtype=bool),lower,upper),[2/3,2/3])
