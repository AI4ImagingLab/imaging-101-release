"""Audit separately recorded complete CLI and active-Notebook executions.

These checks avoid repeating another six-minute primary inverse for each pytest
call. Missing actively generated assets fail; their existence never substitutes
for the mandatory actual native subprocess records in validation.
"""
from pathlib import Path
import json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_active_complete_primary_matches_full_original_and_diagnostics():
    with np.load(ROOT/'notebooks/reference_outputs/fresh.npz',allow_pickle=False) as fresh,np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as original:
        assert int(fresh['iterations'])==100
        assert fresh['q_accepted'].shape==(1881028,)
        np.testing.assert_allclose(fresh['density_matrix'],original['density_matrix'][0],rtol=2e-8,atol=2e-8*np.max(abs(original['density_matrix'])))
        np.testing.assert_allclose(fresh['wigner'],original['wigner'][0],rtol=2e-8,atol=2e-8*np.max(abs(original['wigner'])))
        np.testing.assert_allclose(fresh['likelihood'],original['likelihood'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(fresh['step_fidelity'],original['step_fidelity'],rtol=1e-10,atol=1e-10)
        for n in (1,2,7,25,50,100):np.testing.assert_allclose(fresh[f'rho_step_{n}'],original[f'rho_step_{n}'],rtol=1e-10,atol=1e-10)

def test_all_sensitivity_inverses_have_computed_finite_full_histories():
    with np.load(ROOT/'notebooks/reference_outputs/fresh_comparisons.npz',allow_pickle=False) as z:
        for label,n in [('Cutoff_10',11),('Cutoff_20',21),('Uncorrected_phase',31)]:
            rho=z[label+'_density_matrix'];W=z[label+'_wigner']
            assert rho.shape==(n,n) and W.shape==(201,201)
            assert np.isfinite(rho).all() and np.isfinite(W).all() and np.linalg.norm(W)>0
            assert abs(np.trace(rho)-1)<1e-12
            assert z[label+'_likelihood'].shape==(100,) and z[label+'_step_fidelity'].shape==(100,)

def test_complete_submission_contract_matches_active_primary():
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as submission,np.load(ROOT/'notebooks/reference_outputs/fresh.npz',allow_pickle=False) as fresh:
        assert set(submission.files)=={'density_matrix','wigner'}
        assert submission['density_matrix'].shape==(1,31,31) and submission['density_matrix'].dtype==np.complex128
        assert submission['wigner'].shape==(1,201,201) and submission['wigner'].dtype==np.float64
        np.testing.assert_array_equal(submission['density_matrix'][0],fresh['density_matrix'])
        np.testing.assert_array_equal(submission['wigner'][0],fresh['wigner'])

def test_executed_primary_passes_independent_standalone_Judge(tmp_path):
    report=tmp_path/'judge.json'
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    assert json.loads(report.read_text())['passed']
