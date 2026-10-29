import numpy as np,pytest
from src.physics_model import kinematics,hermite_virtual_fields,bending_stiffness,pressure_virtual_work,PIXEL_SIZE,DENSITY,THICKNESS
from src.solvers import pressure_from_fields
from src.preprocessing import filter_deflection

def test_material_stiffness_units_and_thickness_cube():
    a,b=bending_stiffness();assert a==pytest.approx(2110.3114697802203);assert b==pytest.approx(.3*a)
    c,_=bending_stiffness(thickness=2*THICKNESS);assert c==pytest.approx(8*a)

def test_spatial_constant_acceleration_inertial_pressure():
    t=np.arange(9)/75000.;acc=12.;w=.5*acc*t[:,None,None]**2*np.ones((9,25,27));f=kinematics(w);v=hermite_virtual_fields();q={k:x[4] for k,x in f.items() if k!='time'}
    p,terms=pressure_from_fields(q,v);np.testing.assert_allclose(p,DENSITY*THICKNESS*acc,rtol=2e-14);np.testing.assert_allclose(terms['bending'],0,atol=1e-7)

def test_static_quartic_independent_plate_equilibrium_and_quadrature_bias():
    # w=(x^4+y^4)/24 has exact continuum load2D11. Midpoint Hermite
    # virtual quadrature at N=20 has known factor1-4/N^2=0.99; preserve it.
    x,y=np.meshgrid(np.arange(30)*PIXEL_SIZE,np.arange(32)*PIXEL_SIZE,indexing='ij');z=np.zeros_like(x)
    f={'curv_xx':-x*x/2,'curv_yy':-y*y/2,'curv_xy':z,'acceleration':z};p,_=pressure_from_fields(f,hermite_virtual_fields());D,_=bending_stiffness()
    np.testing.assert_allclose(p/(2*D),.99,rtol=0,atol=1e-13)
    assert np.max(np.abs(p-2*D))/(2*D)<.011

def test_forward_constant_pressure_virtual_work():
    v=hermite_virtual_fields();p=np.full((26,29),77.);out=pressure_virtual_work(p,v['deflection'])/v['deflection'].sum();np.testing.assert_allclose(out,77.,rtol=3e-15)

def test_signed_gradient_and_rigid_offset_invariance():
    x,y=np.meshgrid(np.arange(26)*PIXEL_SIZE,np.arange(27)*PIXEL_SIZE,indexing='ij');w=np.tile((2*x+3*y)[None],(5,1,1));f=kinematics(w)
    np.testing.assert_allclose(f['slope_x'],-2,atol=2e-14);np.testing.assert_allclose(f['slope_y'],-3,atol=2e-14)
    for k in ['curv_xx','curv_yy','curv_xy','acceleration']:np.testing.assert_allclose(f[k],0,atol=1e-10)

def test_hermite_symmetry_and_midpoint_positive_normalization():
    v=hermite_virtual_fields();assert np.all(v['deflection']>0)
    for k,a in v.items():np.testing.assert_allclose(a,a[::-1,::-1],rtol=1e-13,atol=1e-9)
    for n in [3,5,0]:
        with pytest.raises(ValueError):hermite_virtual_fields(n)

def test_filter_constant_and_nonnegative_sigma():
    a=np.ones((1,9,3,4));np.testing.assert_array_equal(filter_deflection(a),a)
    with pytest.raises(ValueError):filter_deflection(a,-1)


def test_visualization_metric_known_case():
    from src.visualization import ncc_nrmse
    a,b=ncc_nrmse(np.array([0.,1.,0.]),np.array([0.,1.,2.]))
    assert a==pytest.approx(1/np.sqrt(5));assert b==pytest.approx(np.sqrt(4/3)/2)

def test_public_constants_and_output_contract():
    from pathlib import Path
    r=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    for value in ['0.0030065919701586503','75000.0','7934.0','0.00495','190000000000','1.385','(1, 250, 60, 78)']:assert value in r
