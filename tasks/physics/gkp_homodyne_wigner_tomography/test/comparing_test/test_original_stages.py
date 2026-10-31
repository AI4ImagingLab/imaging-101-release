from pathlib import Path
import numpy as np
from src import physics_model as pm
from src.solvers import maximum_likelihood
ROOT=Path(__file__).resolve().parents[2]

def test_actual_original_Hermite_and_Born_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        a=pm.hermite_amplitudes(z['q'],z['theta'],cutoff=30)
        np.testing.assert_allclose(a,z['R_fac'],rtol=1e-10,atol=1e-12)
        p=pm.born_probabilities(np.eye(31)/31,a)
        np.testing.assert_allclose(p,z['denom_step_1'],rtol=1e-10,atol=1e-12)

def test_actual_original_seven_density_R_operator_updates():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        a=pm.hermite_amplitudes(z['q'],z['theta']);rho=np.eye(31)/31
        for i in range(1,8):
            rho,R,p=pm.density_update(rho,a)
            np.testing.assert_allclose(R,z[f'R_step_{i}'],rtol=1e-10,atol=1e-10*np.max(abs(R)))
            np.testing.assert_allclose(rho,z[f'rho_step_{i}'],rtol=1e-10,atol=1e-10)

def test_actual_original_solver_histories():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        r=maximum_likelihood(z['q'],z['theta'],iterations=7)
        np.testing.assert_allclose(r['density_matrix'],z['rho'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(r['likelihood'],z['L'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(r['step_fidelity'],z['F'],rtol=1e-10,atol=1e-10)

def test_original_Wigner_and_discrete_probabilities_on_independent_states():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        g=z['physics_grid']
        for name in ('vacuum','one_photon','coherent'):
            W=pm.wigner_function(z[name+'_rho'],g,g)
            np.testing.assert_allclose(W,z[name+'_W'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(W,z[name+'_W_analytic'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(pm.quadrature_probability(z[name+'_rho'],g,.37),z[name+'_q_probability'],rtol=1e-10,atol=1e-12)

def test_original_nontrivial_displacements():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for key,shape,alpha in [('displacement_p',(31,31),1.86/np.sqrt(2)),('displacement_q',(31,31),1j*np.sqrt(2)*np.pi/1.86),('displacement_complex',(9,7),.4+.6j)]:
            np.testing.assert_allclose(pm.displacement(shape,alpha),z[key],rtol=1e-10,atol=1e-12)

def test_independent_likelihood_directional_derivative():
    with np.load(ROOT/'test/fixtures/likelihood_physics.npz',allow_pickle=False) as z:
        a=pm.hermite_amplitudes(z['q'],z['theta']);rho=z['density'];h=z['direction'];eps=1e-6
        fd=-(pm.log_likelihood(rho+eps*h,a)-pm.log_likelihood(rho-eps*h,a))/(2*eps*len(z['q']))
        np.testing.assert_allclose(fd,z['finite_difference'],rtol=0,atol=1e-8)
        _,R,_=pm.density_update(rho,a)
        np.testing.assert_allclose(-np.trace(R@h).real/len(z['q']),z['source_gradient'],rtol=0,atol=1e-10)
