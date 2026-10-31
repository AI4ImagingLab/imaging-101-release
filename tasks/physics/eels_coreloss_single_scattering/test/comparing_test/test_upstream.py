from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_all_independently_frozen_source_stages():
    low,core=load_observations(ROOT/'data/raw_data.npz')
    out=reconstruct(low,core)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for key in f.files:np.testing.assert_allclose(np.asarray(out[key]).reshape(-1),f[key].reshape(-1),rtol=1e-9,atol=1e-8,err_msg=key)
        nrmse=np.sqrt(np.mean((out['reconstruction']-f['reconstruction'])**2))/np.ptp(f['reconstruction'])
        assert nrmse<1e-8
def test_fresh_entrypoint_and_judge():
    import subprocess,sys
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert run.returncode==0,run.stderr
    judge=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(ROOT/'output/judge_report.json')],cwd=ROOT.parent,capture_output=True,text=True,encoding='utf-8')
    assert judge.returncode==0,judge.stderr
