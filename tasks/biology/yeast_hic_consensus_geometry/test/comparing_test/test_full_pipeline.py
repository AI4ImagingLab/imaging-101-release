"""Fresh CLI reproduction on every bound native locus and observed pair."""
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def test_full_fresh_cli_and_original_geometry(tmp_path):
    output=tmp_path/'fresh.npz';report=tmp_path/'judge.json'
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    run=subprocess.run([sys.executable,'-m','src.main','--output',str(output)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=900)
    assert run.returncode==0,run.stderr
    with np.load(output,allow_pickle=False) as f:x=f['reconstruction']
    with np.load(ROOT/'test/fixtures/original_pm1.npz',allow_pickle=False) as f:original=f['coordinates'].copy()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:mask=f['bead_support'][0]
    original[mask]-=original[mask].mean(0);original[~mask]=0
    np.testing.assert_allclose(x[0],original,rtol=1e-10,atol=1e-10)
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(output),'--report',str(report)],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['passed']
