from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load_observations,estimate_background,radius_grid,bin_widths
from src.physics_model import build_kernel
from src.solvers import reconstruct
from src.main import run
def fixture():return np.load(T/'test/fixtures/upstream_numeric.npz',allow_pickle=False)
def test_original_complete_acquisition_and_stages():
    with fixture() as d:
        q,I,e=load_observations(T/'data/raw_data.npz')
        for a,key in zip([q,I,e],['q','intensity','error']):np.testing.assert_array_equal(a,d[key])
        assert q.size==500
        # Every public sigma is redundant with the public observation, not a
        # privileged hidden-noiseless-intensity proxy.
        np.testing.assert_allclose(e,.03*I,rtol=8e-16,atol=0)
        assert estimate_background(q,I)==d['background']
        np.testing.assert_array_equal(radius_grid(),d['r_grid'])
        np.testing.assert_array_equal(bin_widths(radius_grid()),d['bin_widths'])
        np.testing.assert_array_equal(build_kernel(q,radius_grid()),d['kernel'])
def test_original_full_inverse_pipeline():
    with fixture() as d:
        fit=reconstruct(d['q'],d['intensity'],d['error'])
        assert fit['iterations']==101
        np.testing.assert_allclose(fit['reconstruction'],d['distribution'],rtol=1e-11,atol=1e-19)
        np.testing.assert_allclose(fit['model']-fit['background'],d['model_intensity'],rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(fit['residual'],d['residuals'],rtol=1e-10,atol=1e-10)
        assert float(fit['residual']@fit['residual'])<500
        assert np.all(np.diff(fit['history'])<=1e-6)
def test_main_and_readme_constants(tmp_path):
    import matplotlib
    previous=matplotlib.get_backend();out=tmp_path/'prediction.npz';run(output_path=out)
    assert matplotlib.get_backend()==previous
    with np.load(out,allow_pickle=False) as d:assert d['reconstruction'].shape==(1,500) and d['background'].shape==(1,1)
    s=(T/'README.md').read_text(encoding='utf8')
    for item in ['30000','500','10^-24','10^20','10^-4']:assert item in s
    assert '0.5' in (T/'plan/approach.md').read_text(encoding='utf8')
    assert 'github' not in s.lower()
