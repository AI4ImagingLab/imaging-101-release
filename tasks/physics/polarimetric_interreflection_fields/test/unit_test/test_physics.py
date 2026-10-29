import numpy as np
import pytest
from src.physics_model import design_matrix,coefficients_to_fields,fields_to_coefficients,forward,components
from src.solvers import reconstruct,unrestricted_mueller

def angles():
    g,c=np.meshgrid(np.deg2rad([0,45,90,135]),np.deg2rad([0,45,90,135]),indexing='ij')
    return g.ravel(),c.ravel()

def known_case():
    g,c=angles();u=1.;Cs=.3+.2j;Cd=-.4+.1j
    # Independent direct trigonometric measurement, without runtime operators.
    y=.5*(u+abs(Cs)+abs(Cd))+.5*np.real(Cs*np.exp(-2j*(c+g))+Cd*np.exp(-2j*(c-g)))
    fields=np.array([u,Cs.real,Cs.imag,Cd.real,Cd.imag])[:,None,None]
    return g,c,y[:,None,None],fields

def test_independent_sum_difference_malus_acquisition_recovers_every_field():
    g,c,y,truth=known_case()
    np.testing.assert_allclose(reconstruct(y,g,c)['reconstruction'],truth,atol=1e-14)
    np.testing.assert_allclose(forward(truth,g,c),y,atol=1e-14)

def test_coefficient_field_mapping_and_angular_power_conservation():
    g,c,y,truth=known_case();x=fields_to_coefficients(truth)
    np.testing.assert_allclose(coefficients_to_fields(x),truth,atol=1e-14)
    comp=components(truth)
    np.testing.assert_allclose(2*y.mean(),comp['unpolarized']+comp['sum_amplitude']+comp['difference_amplitude'],atol=1e-14)

def test_positive_radiance_units_scale_all_five_fields():
    g,c,y,truth=known_case()
    np.testing.assert_allclose(reconstruct(7*y,g,c)['reconstruction'],7*truth,atol=2e-14)

def test_frame_permutation_preserves_solution_and_rank_failure_is_explicit():
    g,c,y,truth=known_case();p=np.random.default_rng(41).permutation(16)
    np.testing.assert_allclose(reconstruct(y[p],g[p],c[p])['reconstruction'],truth,atol=1e-14)
    with pytest.raises(ValueError,match='Rank-five'):reconstruct(y[:3],g[:3],c[:3])

def test_independent_generator_angle_derivative_signs():
    g,c,y,truth=known_case();step=1e-6
    numeric=(forward(truth,g+step,c)-forward(truth,g-step,c))/(2*step)
    _,sr,si,dr,di=truth[:,0,0]
    expected=-sr*np.sin(2*(c+g))+si*np.cos(2*(c+g))+dr*np.sin(2*(c-g))-di*np.cos(2*(c-g))
    np.testing.assert_allclose(numeric[:,0,0],expected,atol=2e-10,rtol=2e-9)

def test_independent_jones_mirror_parity_not_an_assumed_metadata_handedness():
    g,c=angles();mirror=np.diag([1.,-1.]);single=[];double=[]
    for a,b in zip(g,c):
        incident=np.array([np.cos(a),np.sin(a)]);analyzer=np.array([np.cos(b),np.sin(b)])
        single.append((analyzer@mirror@incident)**2);double.append((analyzer@mirror@mirror@incident)**2)
    np.testing.assert_allclose(single,np.cos(c+g)**2,atol=1e-14)
    np.testing.assert_allclose(double,np.cos(c-g)**2,atol=1e-14)

def test_unrestricted_linear_mueller_control_keeps_four_additional_terms():
    g,c=angles();W=design_matrix(g,c,complete=True)
    coefficients=np.arange(1,10)[:,None,None]*.01
    y=np.tensordot(W,coefficients,axes=(1,0))+.5
    recovered=unrestricted_mueller(y,g,c)['coefficients'];coefficients[0]+=.5
    np.testing.assert_allclose(recovered,coefficients,atol=1e-14)

def test_unpolarized_signed_source_component_is_not_silently_clamped():
    coeff=np.array([.1,.3,0.,0.,0.])[:,None,None]
    fields=coefficients_to_fields(coeff)
    assert fields[0,0,0]<0
    np.testing.assert_allclose(fields_to_coefficients(fields),coeff,atol=1e-14)
