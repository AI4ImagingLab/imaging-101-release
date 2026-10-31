from pathlib import Path
import importlib.util
import hashlib
import numpy as np
from src.preprocessing import load_observations,GRID_SHAPE,GRID_EXTENT

ROOT=Path(__file__).resolve().parents[2]

def test_bound_full_data_and_declared_native_coordinates():
    path=ROOT/'data/raw_data.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='36ce5c1aab6e6b601dbfc99ee1dbddc609291f32c0054ae645108e402e1a1a76'
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'positions','noise_std','flux_contrast'}
        assert z['positions'].shape==(1,64332,3)
        assert z['noise_std'].shape==z['flux_contrast'].shape==(1,64332)
        assert all(z[k].dtype==np.float64 for k in z.files)
    obs=load_observations(path)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        idx=z['indices'].astype(int)
        np.testing.assert_array_equal(obs.positions[idx],z['pixels'][:,:3])
        np.testing.assert_array_equal(obs.noise[idx],z['pixels'][:,3])
        np.testing.assert_array_equal(obs.flux[idx],z['pixels'][:,4])
        np.testing.assert_allclose(np.linalg.norm(z['initial_residual']),z['solve_info'][2],rtol=2e-14)
    assert GRID_SHAPE==(60,48,876) and GRID_EXTENT==(30.,24.,438.)
    assert obs.noise.min()==float(np.float32(.2)) and obs.positions[:,0].min()<0

def test_full_fresh_standard_output_against_frozen_independent_reference():
    spec=importlib.util.spec_from_file_location('full_pipeline_judge',ROOT/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz')
    assert code==0,report
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as z:
        assert z['reconstruction'].shape==(1,60,48,876)

def test_importing_main_does_not_change_notebook_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before

def test_all_observation_diagnostics_against_independent_original_capture():
    refs=ROOT/'notebooks/reference_outputs'
    with np.load(refs/'original_diagnostics.npz',allow_pickle=False) as original, np.load(refs/'fresh_measurements.npz',allow_pickle=False) as fresh:
        for old,new in [('coefficients','alpha'),('predicted_pixels','predicted_pixels'),('residual_norms','residual_norms')]:
            np.testing.assert_allclose(fresh[new],original[old],rtol=2e-11,atol=5e-12)
