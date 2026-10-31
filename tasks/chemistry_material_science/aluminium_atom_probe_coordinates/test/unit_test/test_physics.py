import numpy as np
from src.physics_model import project_to_detector,radius_evolution,depth_increments
from src.solvers import reconstruct
def test_inverse_projection_and_volume_conservation():
    xy=np.array([[1.,.5],[-.8,.4],[.2,-1.5],[0.,0.]],dtype=float);v=np.array([3600.,3600.,3650.,3650.])
    xyz=reconstruct(xy,v)
    np.testing.assert_allclose(project_to_detector(xyz,v),xy,rtol=1e-13,atol=1e-14)
    radius=radius_evolution(v);area=np.pi*np.max(np.linalg.norm(xy*10,axis=1))**2/(110/(1.4*radius))**2
    dz=depth_increments(xy,v)
    np.testing.assert_allclose(np.sum(area*dz),len(xy)/(60.2*.7),rtol=1e-13)
def test_axial_detector_ray():
    v=np.array([3800.,3800.]);xy=np.array([[1.,0.],[0.,0.]])
    xyz=reconstruct(xy,v);assert xyz[1,0]==0 and xyz[1,1]==0
    assert xyz[1,2]>0
