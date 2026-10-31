import numpy as np
from src.solvers import major_axis_fit
from src.physics_model import principal_direction,forward_arai,scatter_box
from src.preprocessing import polar_vectors

def test_exact_physical_line_and_scaling():
    x=np.array([0.,.1,.4,.7,1.]);y=forward_arai(x,45.,30.,1.8)
    f=major_axis_fit(x,y,30.)
    np.testing.assert_allclose(f['field'],45.,atol=1e-12)
    g=major_axis_fit(7*x,7*y,30.);np.testing.assert_allclose(f['field'],g['field'],atol=1e-12)

def test_polar_axes_and_vector_difference():
    np.testing.assert_allclose(polar_vectors([1,1,1],[0,90,0],[0,0,90]),np.eye(3),atol=1e-15)
    nrm=np.array([.3,.4,.5]);trm=np.array([0.,0.,.2]);np.testing.assert_allclose(np.linalg.norm((nrm+trm)-nrm),.2)

def test_pca_sign_and_unit_vector():
    direction=np.array([1.,2.,-3.]);direction/=np.linalg.norm(direction)
    points=np.arange(1.,6.)[:,None]*direction
    found,mad,values=principal_direction(points)
    assert np.isfinite(mad) and mad<1e-5
    np.testing.assert_allclose(found,direction,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(found),1.,atol=1e-15)

def test_scatter_detects_inconsistent_repeat_measurement():
    x=np.linspace(.1,1,8);y=1.2-.8*x
    empty=dict(points=np.empty((0,2)),temperature=np.empty(0),starting=np.empty(0))
    assert scatter_box(x,y,-.8,dict(ptrm=empty),0,600)[0]
    bad=dict(points=np.array([[.2,3.]]),temperature=np.array([300]),starting=np.array([350]))
    assert not scatter_box(x,y,-.8,dict(ptrm=bad),0,600)[0]
