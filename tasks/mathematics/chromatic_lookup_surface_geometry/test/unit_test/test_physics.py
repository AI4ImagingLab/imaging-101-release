import numpy as np
import pytest
from src.preprocessing import normalize_observations
from src.solvers import decode_rays
from src.physics_model import project_depth_to_points
from src.generate_data import perturb_normalized

def test_known_euclidean_norm_tie_and_mask():
    colors=np.array([[[0,0,0],[1,2,2],[1,2,2]],[[0,0,0],[2,0,0],[4,0,0]],[[0,0,0],[1,0,0],[2,0,0]]],dtype=np.float32)
    depths=np.array([[10,20,30],[11,21,31],[12,22,32]],dtype=np.float32)
    query=np.array([[1,2,2],[3,0,0],[2,0,0]],dtype=np.float32)
    d,i,r=decode_rays(colors,depths,query,np.array([True,True,False]))
    assert np.array_equal(d,np.array([20,21,-1],dtype=np.float32))
    assert np.array_equal(i,[1,1,0])
    assert np.array_equal(r[:2],[0,1]) and np.isinf(r[2])

def test_normalization_uses_minimum_white_channel_and_float64_divide():
    p=np.array([[[[100,200,300],[100,200,300],[0,100,200]]]],dtype=np.uint16)
    w=np.array([[[[1000,2000,3000],[0,2000,3000],[49,1000,1000]]]],dtype=np.uint16)
    q,m=normalize_observations(p,w)
    assert np.array_equal(m,[[[True,False,False]]])
    assert np.array_equal(q[0,0,0],np.array([.1,.1,.1],dtype=np.float32))
    assert not q[~m].any() and q.dtype==np.float32

def test_every_dictionary_sample_is_a_photometric_roundtrip():
    colors=np.array([[[0,0,1],[1,0,1],[0,1,1],[1,1,1]]],dtype=np.float32)
    depths=np.array([[700,710,720,730]],dtype=np.float32)
    for expected in range(4):
        d,i,r=decode_rays(colors,depths,colors[:,expected],np.array([True]))
        assert d[0]==depths[0,expected] and i[0]==expected and r[0]==0

def test_pinhole_axial_depth_units_and_axes():
    depth=np.array([[2,3],[4,5]],dtype=float)
    points=project_depth_to_points(depth,np.eye(3),np.zeros(5),roi=(0,0,2,2))
    assert np.array_equal(points,np.array([[[0,0,2],[3,0,3]],[[0,4,4],[5,5,5]]]))

def test_brown_conrady_forward_inverse_physical_roundtrip():
    K=np.array([[3500,0,967],[0,3500,509],[0,0,1.]])
    k1,k2,p1,p2,k3=dist=np.array([-.052,-.059,.0014,-.00031,.358])
    points=project_depth_to_points(np.full((4,5),700.),K,dist,roi=(650,200,655,204))
    x=points[:,:,0]/700;y=points[:,:,1]/700;r=x*x+y*y;rad=1+r*(k1+r*(k2+r*k3))
    xd=x*rad+2*p1*x*y+p2*(r+2*x*x);yd=y*rad+p1*(r+2*y*y)+2*p2*x*y
    yy,xx=np.indices((4,5))
    np.testing.assert_allclose(xd*K[0,0]+K[0,2],xx+650,rtol=0,atol=1e-9)
    np.testing.assert_allclose(yd*K[1,1]+K[1,2],yy+200,rtol=0,atol=1e-9)

def test_noise_zero_endpoints_and_input_immutability():
    q=np.ones((1,1,2,3),dtype=np.float32);mask=np.array([[[True,False]]]);before=q.copy()
    out=perturb_normalized(q,mask,0)
    assert np.array_equal(out[mask],q[mask]) and not out[~mask].any() and np.array_equal(q,before)
    changed=perturb_normalized(q,mask,.01)
    assert np.array_equal(changed,perturb_normalized(q,mask,.01)) and np.array_equal(q,before)

@pytest.mark.parametrize('sigma',[-1,np.nan,np.inf])
def test_noise_rejects_unphysical_settings(sigma):
    with pytest.raises(ValueError):perturb_normalized(np.ones((1,1,2,3)),np.ones((1,1,2),dtype=bool),sigma)
