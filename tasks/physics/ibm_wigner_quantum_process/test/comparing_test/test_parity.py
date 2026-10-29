from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct, optimization_inputs, objective, optimize_comparison
ROOT=Path(__file__).resolve().parents[2]

def test_all_original_stages():
    d=load_data(ROOT/'data/raw_data.npz');a=reconstruct(d)
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        for k in ['droplets','basis','correlation']:
            np.testing.assert_allclose(a[k],f[k],atol=1e-12,rtol=0)
        for k in ['quaternion_history','reconstruction']:
            np.testing.assert_allclose(a[k],f[k],atol=1e-10,rtol=0)
        np.testing.assert_allclose(a['correlation']/0.00995,f['scaled_correlation'],atol=1e-12,rtol=0)

def test_optimizer_original_objective_and_final():
    d=load_data(ROOT/'data/raw_data.npz');channels,x=optimization_inputs(d)
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(channels,f['normalized_channels'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(x,f['optimizer_initial'],atol=1e-12,rtol=0)
        measured=np.array([objective(z,channels) for z in f['optimized_angles']])
        np.testing.assert_allclose(measured,f['optimized_cost'],atol=1e-12,rtol=0)
        fresh=optimize_comparison(d)
        assert abs(fresh['objective'][-1]-f['optimized_cost'][-1])<1e-7
        # An independent minimization need not follow the same iterates.
        assert np.linalg.norm(fresh['reconstruction']-f['optimized_reconstruction'])<1e-5

def test_pipeline(tmp_path):
    import subprocess,sys
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')],cwd=tmp_path)
    assert p.returncode==0
