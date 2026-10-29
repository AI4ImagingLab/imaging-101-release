from pathlib import Path
import importlib.util
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def test_fresh_full_pipeline_source_parity_and_judge():
    result=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,shell=False)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as fresh,np.load(ROOT/'notebooks/reference_outputs/upstream_result.npz',allow_pickle=False) as source:
        for key in ['ktrans','extracellular_fraction']:
            x=fresh[key];y=source[key]
            nrmse=np.sqrt(np.mean((x-y)**2))/np.ptp(y)
            assert nrmse<=1e-6,(key,nrmse)
            assert x.shape==(1,60,50) and np.isfinite(x).all()
    judged=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'],cwd=ROOT,capture_output=True,shell=False)
    assert judged.returncode==0,judged.stderr.decode(errors='replace')

def test_main_import_preserves_plot_backend():
    code='import matplotlib; b=matplotlib.get_backend(); import src.main; assert matplotlib.get_backend()==b'
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,shell=False)
    assert result.returncode==0,result.stderr.decode(errors='replace')
