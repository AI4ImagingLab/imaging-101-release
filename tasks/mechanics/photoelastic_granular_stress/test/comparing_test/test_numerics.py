from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare,disk_mask,FSIGMA,CENTER,RADIUS,NOMINAL_PIXELS_PER_METRE,EFFECTIVE_PIXELS_PER_METRE
from src.physics_model import stress_basis,forward,principal_difference
from src.solvers import reconstruct,constraint_matrix,objective_gradient,normjac,normbound,decode
ROOT=Path(__file__).resolve().parents[2]

def test_original_unmodified_forward_and_corrected_reference():
    f=np.load(ROOT/'test/fixtures/balanced_upstream.npz');s=np.load(ROOT/'test/fixtures/balanced_stress.npz');d=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(d['rgb'],f['input_rgb']);np.testing.assert_array_equal(d['contact_angles'],f['input_beta']);np.testing.assert_array_equal(prepare(d['rgb']),f['input_intensity'])
    b=stress_basis(d['contact_angles'],alpha=f['output_alpha'])
    np.testing.assert_allclose(forward(f['output_forces'],b),f['output_forward'],rtol=1e-8,atol=2e-10)
    np.testing.assert_allclose(principal_difference(f['output_forces'],b),s['output_stress'],rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(objective_gradient(f['output_cartesian'],prepare(d['rgb']),d['contact_angles']),f['output_gradient'],rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(constraint_matrix(d['contact_angles']),f['input_constraint_matrix'])
    r,info=reconstruct(d);assert info['objective']<.75*517.0640695772449
    assert abs(info['objective']-166.19705634301323)<1e-4
    np.testing.assert_allclose(r['forces'][0],f['output_forces'],rtol=1e-3,atol=1e-9)
    np.testing.assert_allclose(r['contact_directions'][0],f['output_alpha'],rtol=1e-3,atol=1e-8)
    assert abs(info['constraint_residual']).max()<1e-9

def test_static_constraints_and_analytic_jacobians():
    beta=np.array([0.,np.pi/2,np.pi,-np.pi/2]);A=constraint_matrix(beta);q=np.tile([.04,0.],4)
    np.testing.assert_allclose(A@q,0,atol=1e-16)
    h=1e-6;numeric=np.stack([(A@(q+v)-A@(q-v))/(2*h) for v in np.eye(8)*h],axis=1)
    np.testing.assert_allclose(numeric,A,atol=1e-10)
    numeric=np.stack([(normbound(q+v)-normbound(q-v))/(2*h) for v in np.eye(8)*h],axis=1)
    np.testing.assert_allclose(numeric,normjac(q),atol=1e-8)
    q[1]=.01;assert abs((A@q)[2]-.01)<1e-15

def test_elastic_superposition_and_optical_units():
    beta=np.array([0.,np.pi]);b=stress_basis(beta,center=(90.5,90.5),radius=73);f=np.array([.1,.1]);stress=principal_difference(f,b)
    np.testing.assert_allclose(principal_difference(2*f,b),2*stress,rtol=1e-14,atol=1e-13)
    np.testing.assert_allclose(stress[1:,1:],stress[1:,1:][::-1,::-1],rtol=1e-8,atol=1e-7)
    np.testing.assert_allclose(forward(2*f,b,2*FSIGMA),forward(f,b,FSIGMA),rtol=1e-13,atol=1e-13)
    assert np.all(forward(np.zeros(2),b)==0)
    assert np.min(forward(f,b))>=0 and np.max(forward(f,b))<=1

def test_observation_geometry_and_constants():
    f=np.load(ROOT/'test/fixtures/geometry.npz');np.testing.assert_array_equal(CENTER,f['output_center']);assert RADIUS==f['output_radius']
    text=(ROOT/'README.md').read_text(encoding='utf8');assert str(RADIUS) in text and str(CENTER[0]) in text
    assert FSIGMA==140 and NOMINAL_PIXELS_PER_METRE==EFFECTIVE_PIXELS_PER_METRE==160/.007
