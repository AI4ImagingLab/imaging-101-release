from pathlib import Path
import numpy as np
from src.physics_model import transmission
from src.solvers import paired_lm,jacobian,reconstruct,outputs
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        fixed=np.setdiff1d(np.arange(44),f['free']);pred=transmission(f['velocity'],f['initial_params'])
        np.testing.assert_allclose(pred,f['initial_model'],atol=1e-10,rtol=0)
        J,_=jacobian(transmission,f['initial_params'],f['velocity'],fixed,pred)
        np.testing.assert_allclose(J,f['jacobian'],rtol=1e-4,atol=1e-7)
        for outer in [1,2,5]:
            p,_=paired_lm(f['velocity'],f['observation'],f['initial_params'],fixed,f['bounds'],outer_limit=outer)
            np.testing.assert_allclose(p,f['outer_'+str(outer)],atol=.001,rtol=0)

def test_complete_original_pipeline():
    with np.load(ROOT/'test/fixtures/upstream_full.npz') as f:
        result,_=reconstruct(f['velocity'],f['observation']);ref=outputs(f['params'],f['velocity'])
        np.testing.assert_allclose(result['reconstruction'],ref['reconstruction'],atol=.001,rtol=0)
        np.testing.assert_allclose(result['optical_thickness'],ref['optical_thickness'],atol=.002,rtol=0)
        np.testing.assert_allclose(result['baseline'],ref['baseline'],atol=1e-5,rtol=0)
        np.testing.assert_allclose(result['transmission'][0],f['prediction'],atol=1e-5,rtol=0)
