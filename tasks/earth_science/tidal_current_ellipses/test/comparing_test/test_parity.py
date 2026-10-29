from pathlib import Path
import numpy as np
from src.preprocessing import load_data,polar_to_velocity
from src.solvers import solve
from src.physics_model import ellipse_parameters
ROOT=Path(__file__).resolve().parents[2]
def test_independent_original_all_stages_and_full_pipeline():
    a=load_data(ROOT/'data/raw_data.npz');v=polar_to_velocity(a['speed'],a['direction_deg']);r,d=solve(a['time_seconds'],v,a['frequency_cph'],return_diagnostics=True)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)as f:
        np.testing.assert_allclose(d['B'][::97],f['stage_B_probe'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(d['m'],f['stage_m'],rtol=1e-10,atol=1e-10)
        assert d['reference_days']==float(f['reference_time_days'])
        np.testing.assert_allclose((r['reconstruction'][0,0]+1j*r['reconstruction'][0,1])[::97],f['stage_xmod_probe'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(ellipse_parameters(r['cosine'][0],r['sine'][0]),f['output_ellipse'][0],rtol=1e-10,atol=1e-10)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:
        for k in r:np.testing.assert_allclose(r[k],f[k],rtol=1e-10,atol=1e-10)
