from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load_observations
from src.physics_model import build_kernel
from src.solvers import reconstruct
from src.visualization import metrics
from src.main import run
def test_independent_source_full96_and_kernel():
    raw=load_observations(T/'data/raw_data.npz')
    with np.load(T/'test/fixtures/upstream_numeric.npz') as f:
        np.testing.assert_array_equal(raw['relative_pressure'],f['pressure']);np.testing.assert_array_equal(raw['loading'],f['loading']);np.testing.assert_array_equal(raw['pore_width'],f['widths'])
        assert raw['loading'].shape==(96,)
        np.testing.assert_allclose(build_kernel(raw['relative_pressure'],raw['kernel_relative_pressure'],raw['kernel_loading']),f['kernel'],rtol=1e-14,atol=1e-12)
def test_corrected_original_complete_inverse_and_spline():
    raw=load_observations(T/'data/raw_data.npz');fit=reconstruct(raw)
    with np.load(T/'test/fixtures/upstream_numeric.npz') as f:
        for estimate,reference in [(fit['bin_weights'],f['coefficients']),(fit['reconstruction'],f['pore_distribution'])]:
            m=metrics(estimate,reference);assert m['nrmse']<1e-4 and m['ncc']>.999999
        np.testing.assert_allclose(fit['width_axis'],f['pore_widths'],rtol=1e-14,atol=1e-14)
        assert np.max(abs(fit['cumulative_weights']-f['pore_volume_cumulative']))<1e-4
        assert np.linalg.norm(fit['fitted_loading']-raw['loading'])/np.linalg.norm(raw['loading'])<.04
        assert abs(fit['objective']-20.97009013489817)<1e-6
def test_entrypoint_backend_contract_and_allquantities(tmp_path):
    import matplotlib
    before=matplotlib.get_backend();out=tmp_path/'result.npz';run(output_path=out);assert matplotlib.get_backend()==before
    with np.load(out) as z:
        assert set(z.files)=={'reconstruction','bin_weights','cumulative_weights'}
        assert z['reconstruction'].shape==(1,100) and z['bin_weights'].shape==(1,77)
    s=(T/'README.md').read_text(encoding='utf8');assert 'github' not in s.lower()
    for value in ['96','77','100','77.355','1.0132507274270477']:assert value in s
