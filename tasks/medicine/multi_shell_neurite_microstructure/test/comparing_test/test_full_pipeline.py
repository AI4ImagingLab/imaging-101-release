from pathlib import Path
import os,subprocess,sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def test_fresh_native_full_cli_and_standalone_judge(tmp_path):
    environment=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1'
    prediction=tmp_path/'fresh.npz';report=tmp_path/'judge.json'
    run=subprocess.run([sys.executable,'-m','src.main','--output',str(prediction)],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=1200)
    assert run.returncode==0,run.stderr
    with np.load(prediction) as f, np.load(ROOT/'evaluation/baseline_reference.npz') as g, np.load(ROOT/'data/raw_data.npz') as d:
        mask=d['observation_mask']
        for key in g.files:
            p=f[key];r=g[key];assert p.shape==(1,96,96,19) and np.isfinite(p).all() and np.all(p[~mask]==0)
            assert np.sqrt(np.mean((p[mask]-r[mask])**2))/np.ptp(r[mask])<=1e-5
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,env=environment,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    result=json.loads(report.read_text());assert result['passed'] and len(result['metrics'])==8
