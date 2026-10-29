from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import analyze_movie
ROOT=Path(__file__).resolve().parents[2]

def test_full_native_original_parity_and_pipeline(tmp_path):
    from src.main import run
    from evaluation.judge import evaluate
    frames=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as frozen:
        for i,movie in enumerate(frames[0]):
            a=analyze_movie(movie,i)
            keys={'output_dqt':a['packed'][[0,1,9,99,998,1998]],'output_power':a['packed'][-2],'output_variance':a['packed'][-1],'output_azimuthal':a['averages'][:,:-2],'output_azimuthal_err':a['errors'][:,:-2],'output_q':a['q'],'output_resampled':a['data'],'output_resampled_err':a['error'],'output_fit_parameters':a['parameters']}
            for key,value in keys.items():np.testing.assert_allclose(value,frozen[f'{key}_{i}'],rtol=1e-10,atol=1e-7,equal_nan=True)
    path=tmp_path/'reconstruction.npz';run(ROOT/'data/raw_data.npz',path)
    with np.load(path) as actual,np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
        for key in ('reconstruction','speed'):np.testing.assert_allclose(actual[key],ref[key],rtol=2e-4,atol=1e-5)
    assert evaluate(path,tmp_path/'report.json')==0
