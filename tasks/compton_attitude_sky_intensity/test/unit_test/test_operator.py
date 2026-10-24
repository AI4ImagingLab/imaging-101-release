import numpy as np
from src.physics_model import ComptonOperator,ring_pixel_angles

def toy_operator():
    coordinates=np.indices((2,5,3)).reshape(3,-1)
    weights=(np.arange(coordinates.shape[1])%7+1).astype(np.float32)/13
    response=(np.arange(12).reshape(3,4)+1).astype(np.float32)/17
    return ComptonOperator(response,coordinates,weights,pixel_area=.4),coordinates,weights,response

def test_explicit_compton_sum():
    operator,coordinates,weights,response=toy_operator();field=np.arange(5,dtype=np.float32)/3+.2
    expected=np.zeros((2,4),dtype=np.float64)
    for (attitude,sky,local),weight in zip(coordinates.T,weights):
        expected[attitude]+=.4*float(weight)*float(field[sky])*response[local].astype(np.float64)
    np.testing.assert_allclose(operator.forward(field,add_floor=False),expected,rtol=1e-6)

def test_adjoint_and_exposure():
    operator,*_=toy_operator();field=np.arange(5,dtype=np.float32)+1
    values=np.arange(8,dtype=np.float32).reshape(2,4)/11
    left=np.sum(operator.forward(field,add_floor=False).astype(np.float64)*values)
    right=np.dot(field.astype(np.float64),operator.adjoint(values))
    assert abs(left-right)/abs(right)<1e-6
    np.testing.assert_allclose(operator.compute_exposure(),operator.adjoint(np.ones((2,4),dtype=np.float32)),rtol=1e-6)

def test_equal_area_ring_geometry():
    theta,phi=ring_pixel_angles()
    assert theta.shape==phi.shape==(3072,)
    assert np.all((theta>0)&(theta<np.pi)) and np.all((phi>=0)&(phi<2*np.pi))
    assert abs(np.mean(np.cos(theta)))<1e-14
    vectors=np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    assert np.max(abs(np.linalg.norm(vectors,axis=1)-1))<1e-14
