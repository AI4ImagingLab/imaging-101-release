import numpy as np,pytest
from src.preprocessing import camera_parameters
from src.physics_model import project,rays,refine,predict,detect
from src.generate_data import known_particle_images
@pytest.mark.parametrize('point',[[0,0,0],[-.73,.42,.61],[6,-3,12],[-10,6,-8]])
def test_known_singleton_coordinate_refinement(data,point):
    cal=camera_parameters(data);p=np.asarray(point,float);images=known_particle_images(cal,p);otf=np.tile([100.,1.2,1.4],(4,1));initial=p+np.array([.017,-.012,.009]);result,score,flag=refine(initial[None],images,cal,otf)
    assert np.linalg.norm(result[0]-p)<1e-4
    def full_image_error(x):
        return np.linalg.norm(known_particle_images(cal,x)-images)
    assert full_image_error(result[0])<=full_image_error(initial)
    assert flag[0]==0
def test_camera_inverse_physical_lines(data):
    cal=camera_parameters(data);p=np.array([[0.,0.,0.],[3.,-2.,1.]])
    for camera in cal:
        ray=rays(project(p,camera),camera);d=p-ray[:,:3];perp=d-(d*ray[:,3:]).sum(axis=1)[:,None]*ray[:,3:];assert np.max(np.linalg.norm(perp,axis=1))<1e-8
def test_constant_predictor_and_insufficient_history():
    np.testing.assert_allclose(predict(np.tile([.2,-2.,4.],(6,1))),[.2,-2.,4.],rtol=0,atol=1e-12)
    with pytest.raises(ValueError):predict(np.zeros((2,3)))
def test_peak_tie_convention():
    im=np.zeros((7,7));im[3,3:5]=100;points=detect(im);assert len(points)==1 and points[0,0]>3
