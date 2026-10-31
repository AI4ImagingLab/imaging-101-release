import numpy as np
import pytest
from src.physics_model import shape_functions, green_strain, prepare_reference, element_pixels


def test_shape_partition_and_corner_identity():
    x = np.array([0., 1., 1., 0.]); y = np.array([0., 0., 1., 1.])
    np.testing.assert_array_equal(shape_functions(x,y), np.eye(4))
    np.testing.assert_allclose(shape_functions(x*.4+.2,y*.4+.2).sum(axis=-1), 1)


def test_affine_kinematics_and_strain():
    nodes = np.array([[10.,10.],[20.,10.],[20.,20.],[10.,20.]])
    elements = np.array([[0,1,2,3]])
    angle=.3
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    np.testing.assert_allclose(green_strain(nodes,nodes@rotation.T+3,elements),0,atol=1e-14)
    F=np.array([[1.02,.04],[.01,.98]])
    np.testing.assert_allclose(green_strain(nodes,nodes@F.T,elements)[0],(F.T@F-np.eye(2))/2,atol=1e-14)


def test_reference_jacobian_by_finite_difference():
    y,x=np.mgrid[:40,:40]
    image=x*x*.2+x*y*.1+y*y*.15
    nodes=np.array([[10.,10.],[20.,10.],[20.,20.],[10.,20.]])
    (N,B,I0),=prepare_reference(image,nodes,np.array([[0,1,2,3]]))[0]
    xp,yp,_=element_pixels(nodes)
    direction=np.arange(8)*.03
    mapped=N@direction.reshape(2,4).T
    eps=1e-5
    def f(xx,yy):return xx*xx*.2+xx*yy*.1+yy*yy*.15
    finite=(f(xp+eps*mapped[:,0],yp+eps*mapped[:,1])-f(xp-eps*mapped[:,0],yp-eps*mapped[:,1]))/(2*eps)
    np.testing.assert_allclose(B.T@direction,finite,atol=2e-8)


def test_no_convergence_is_not_silently_accepted():
    from pathlib import Path
    from src.preprocessing import load_data
    from src.solvers import reconstruct
    root=Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError,match='did not meet'):
        reconstruct(*load_data(root/'data/raw_data.npz'),max_iterations=1)
