from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,log_measurements,TAU
from src.physics_model import design_matrix
from src.solvers import fit_signals,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(TAU,f['config_tau'])
        X,indices=design_matrix()
        np.testing.assert_array_equal(X,f['output_design'])
        for c in ['csfnull','nonull','gmnull','wmnull']:
            logs=log_measurements(f[c+'_input_signals'])
            np.testing.assert_allclose(logs,f[c+'_output_log'],atol=1e-14,rtol=1e-14)
            np.testing.assert_allclose(logs[:,indices].T,f[c+'_output_Y'],atol=1e-14,rtol=1e-14)
            p,_=fit_signals(f[c+'_input_signals'])
            np.testing.assert_allclose(p.T,f[c+'_output_p'],atol=1e-10,rtol=1e-10)

def test_complete_pipeline_original():
    output,diag=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in output: np.testing.assert_allclose(output[k],f[k],atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(diag['mask'].sum(axis=(1,2,3)),[4425,4855,3206,1672])
