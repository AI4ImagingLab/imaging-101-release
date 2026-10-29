from pathlib import Path
import numpy as np
from src.preprocessing import prepare,sample_line
from src.physics_model import compression_matrix,heat_gram,profile_residual
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_independent_upstream_stages():
    with np.load(ROOT/'data/raw_data.npz') as data,np.load(ROOT/'test/fixtures/upstream.npz') as f:
        p=prepare(data)
        np.testing.assert_array_equal(p['profile'],f['output_profile'][:,0])
        np.testing.assert_array_equal(p['control'],f['output_control'][:,0])
        np.testing.assert_array_equal(p['normalized'],f['output_normalized'])
        c=compression_matrix(89)
        np.testing.assert_allclose(c,f['output_compression_matrix'],rtol=1e-14,atol=1e-14)
        q=p['normalized']@c.T
        np.testing.assert_allclose(q,f['output_compressed'],rtol=0,atol=1e-5)
        np.testing.assert_allclose(heat_gram(.9,p['epsilon']).reshape(41,9,41,9),f['output_kernel'],rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(profile_residual(.9,q,p['epsilon'],p['background']),f['output_fixed_residual'],rtol=0,atol=.0003)

def test_pipeline_against_original_ensemble():
    with np.load(ROOT/'data/raw_data.npz') as data,np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
        result=reconstruct(data)
        assert result['success']
        assert abs(result['diffusivity'][0,0]/ref['diffusivity'][0,0]-1)<.05
        a=result['reconstruction'];b=ref['reconstruction']
        assert np.sqrt(np.mean((a-b)**2))/np.ptp(b)<.003
        assert result['cost']<.204

def test_full_cli(tmp_path):
    import subprocess,sys,shutil
    shutil.copytree(ROOT/'src',tmp_path/'src')
    shutil.copytree(ROOT/'data',tmp_path/'data')
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    with np.load(tmp_path/'output/reconstruction.npz',allow_pickle=False) as a:
        assert a['reconstruction'].shape==(1,41,89)
        assert a['diffusivity'].shape==(1,1)
