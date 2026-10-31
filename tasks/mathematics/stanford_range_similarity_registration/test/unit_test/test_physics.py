import numpy as np
from src.physics_model import transform,expectation
from src.solvers import one_step

def test_similarity_distance_units_and_proper_rotation():
    r=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);p=np.array([[0.,0,0],[1,2,3],[3,2,1]])
    q=transform(p,2.,r,np.array([5.,6,7]))
    np.testing.assert_allclose(np.linalg.norm(q[1:]-q[0],axis=1),2*np.linalg.norm(p[1:]-p[0],axis=1))
    np.testing.assert_array_equal(q[0],[5,6,7]);assert np.linalg.det(r)==1

def test_independent_two_component_probability_and_outlier_mass():
    x=np.array([[0.,0.],[2.,0.]]);y=x.copy();v=.5
    p,p1,pt1,px,mass=expectation(x,y,v,0.)
    a=np.exp(-4.)
    np.testing.assert_allclose(p,[[1/(1+a),a/(1+a)],[a/(1+a),1/(1+a)]],rtol=1e-14)
    np.testing.assert_allclose(pt1,1,atol=1e-15);assert mass==2
    assert np.all(expectation(x,y,v,.5)[0].sum(0)<1)

def test_weighted_procrustes_recovers_known_similarity():
    # Tight separated Gaussians give independently known one-to-one posteriors.
    y=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
    r=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);t=np.array([.3,.2,.1]);x=1.2*y@r+t
    step=one_step(x,y,x.copy(),1e-6,np.inf,0.)
    np.testing.assert_allclose(step['R'],r,atol=1e-14);np.testing.assert_allclose(step['t'],t,atol=1e-14);assert abs(step['s']-1.2)<1e-14
    assert np.linalg.det(step['R'])>0
