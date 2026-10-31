from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.physics_model import rectify,determinant_polynomial,transform_points,inverse_radius_lookup
from src.preprocessing import load_data,normalize_points,CENTER
from src.visualization import line_residuals
def test_uniform_scene_preservation():
    image=np.full((506,688,3),213,dtype=np.uint8)
    a=np.array([1.,0.,1e-6,0.,9e-12])
    r=rectify(image,a)
    assert np.all(r[r!=0]==213)

def test_quartic_against_independent_scatter():
    d=load_data();q,_=normalize_points(d['line_points']);p=determinant_polynomial(q,d['line_ids'])
    k2,k4=.1,.02;transformed=q*(1+k2*np.sum(q*q,axis=1)+k4*np.sum(q*q,axis=1)**2)[:,None]
    expected=0.
    for k in np.unique(d['line_ids']):
        z=transformed[d['line_ids']==k];z-=z.mean(axis=0)
        expected+=np.linalg.det(z.T@z)
    np.testing.assert_allclose(np.polynomial.polynomial.polyval2d(k2,k4,p),expected,rtol=1e-10)

def test_inverse_radius_physical_roundtrip():
    a=np.array([1.,0.,1e-6,0.,9e-12]);lut=inverse_radius_lookup(a,500.)
    radii=np.arange(1,len(lut));original=radii*lut[1:]
    np.testing.assert_allclose(original*np.polynomial.polynomial.polyval(original,a),radii,rtol=1e-12,atol=1e-10)

def test_identity_and_line_normal_distance():
    p=np.array([[1.,2.],[2.,4.],[4.,8.]])
    np.testing.assert_array_equal(transform_points(p,np.array([1.,0,0,0,0])),p)
    np.testing.assert_allclose(line_residuals(p,np.zeros(3)),0,atol=1e-14)
