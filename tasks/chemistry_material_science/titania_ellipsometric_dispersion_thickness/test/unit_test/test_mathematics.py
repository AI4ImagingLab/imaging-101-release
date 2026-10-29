import numpy as np
import pytest
from src.physics_model import fresnel,reflection,cauchy_index
from src.solvers import initial_parameters,to_internal,from_internal
from src.preprocessing import load_observations

def test_fresnel_known_normal_incidence_and_lossless_flux():
    rs,rp,ts,tp=fresnel(1.,1.5,0.,0.)
    np.testing.assert_allclose([rs,rp,ts,tp],[-.2,.2,.8,.8],rtol=0,atol=1e-15)
    for r,t in [(rs,ts),(rp,tp)]:assert abs(abs(r)**2+1.5*abs(t)**2-1)<1e-14

def test_brewster_angle_known_air_glass():
    angle=np.arctan(1.5);transmitted=np.arcsin(np.sin(angle)/1.5)
    assert abs(fresnel(1.,1.5,angle,transmitted)[1])<1e-14

def test_zero_thickness_stack_reduces_to_single_interface():
    data=load_observations();parameters=initial_parameters();parameters[3]=0.
    rs,rp=reflection(data['wavelength_nm'],parameters,data['silicon_index'],oxide_thickness=0.)
    index=data['silicon_index'];angle=np.deg2rad(70);a=np.full(len(index),angle);b=np.lib.scimath.arcsin(np.sin(angle)/index)
    expected=fresnel(1.,index,a,b)
    np.testing.assert_allclose(rs,expected[0],atol=2e-14,rtol=2e-14)
    np.testing.assert_allclose(rp,expected[1],atol=2e-14,rtol=2e-14)

@pytest.mark.parametrize('p',[[-100,-40000,-40000,0],[100,40000,40000,40000],[2.236,451,251,20],[2.,0.,-30.,140.]])
def test_explicit_two_sided_bound_transform_round_trip(p):
    np.testing.assert_allclose(from_internal(to_internal(p)),p,atol=1e-11,rtol=1e-12)

def test_cauchy_coefficient_scaling_and_wavelength_units():
    value=cauchy_index(np.array([500.]),np.array([2.,500.,250.,25.]))
    np.testing.assert_allclose(value,[2.+50000/500**2+2.5e9/500**4],atol=1e-15)

def test_whole_grid_against_independent_recursive_reflection():
    data=load_observations();l=data['wavelength_nm'];parameters=initial_parameters()
    index=np.stack([np.ones_like(l),cauchy_index(l,parameters),1.452+3600/l**2,data['silicon_index']])
    epsilon=index**2;q=np.sqrt(epsilon-np.sin(np.deg2rad(70))**2+0j)
    q=np.where(q.imag<0,-q,q);expected=[]
    for pol in ['s','p']:
        interfaces=(q[:-1]-q[1:])/(q[:-1]+q[1:]) if pol=='s' else (epsilon[1:]*q[:-1]-epsilon[:-1]*q[1:])/(epsilon[1:]*q[:-1]+epsilon[:-1]*q[1:])
        total=interfaces[-1]
        for j in [1,0]:
            phase=np.exp(4j*np.pi*q[j+1]*[parameters[3],276.36][j]/l)
            total=(interfaces[j]+total*phase)/(1+interfaces[j]*total*phase)
        expected.append(total)
    measured=reflection(l,parameters,data['silicon_index'])
    for a,b in zip(measured,expected):np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-13)

def test_importing_entry_point_preserves_matplotlib_backend():
    import matplotlib
    previous=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==previous

def test_public_known_constants_agree_with_local_acquisition_model():
    from pathlib import Path
    from src.physics_model import MODEL_ANGLE_DEG,ACTUAL_ANGLE_DEG,OXIDE_THICKNESS_NM
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert MODEL_ANGLE_DEG==70. and ACTUAL_ANGLE_DEG==70.06 and OXIDE_THICKNESS_NM==276.36
    for value in ['70.06','276.36','1.452+3600/L²','400–800','926','100a/L^2','10^7b/L^4']:
        assert value in text,value



def test_visual_metrics_independent_circular_known_case():
    from src.visualization import comparison_metrics, rho_angles
    observed=np.resize(np.array([179.,-179.,359.,1.]),926)
    predicted=np.resize(np.array([-179.,179.,1.,359.]),926)
    rho=np.exp(-1j*np.deg2rad(predicted))
    data={'psi_deg':np.full(926,45.),'delta_deg':observed}
    result=comparison_metrics(data,rho)
    np.testing.assert_allclose(result['relative_complex_rho_rms'],2*np.sin(np.deg2rad(1.)),rtol=1e-13)
    assert result['psi_rms_deg']<1e-13
    np.testing.assert_allclose(result['circular_delta_rms_deg'],2.,atol=1e-12)
    psi,delta=rho_angles(rho)
    np.testing.assert_allclose(psi,45.,atol=1e-13)
    np.testing.assert_allclose((delta-predicted+180)%360-180,0.,atol=1e-12)
