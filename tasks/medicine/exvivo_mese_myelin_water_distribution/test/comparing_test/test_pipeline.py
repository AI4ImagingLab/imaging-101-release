from pathlib import Path
import subprocess,sys
import numpy as np
from src.preprocessing import LAMBDA
ROOT=Path(__file__).resolve().parents[2]

def test_public_interface_and_native_precision():
    text=(ROOT/'README.md').read_text()
    assert str(LAMBDA) in text
    assert all(x in text for x in ['15<T2<=40','2629','1752','-128','128','1e-6','1e-8'])
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert z['flip_scale'].dtype==np.float32 and z['echo_magnitudes'].dtype==np.float64 and z['foreground_complex_echoes'].dtype==np.complex128
        for key in z.files:assert '| '+key+' |' in text and z[key].shape[0]==1
        assert np.sum(z['estimation_mask'])==2629 and np.sum(z['display_mask'])==1752

def test_standard_cli_and_standalone_foreign_judge(tmp_path):
    a=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True);assert a.returncode==0,a.stderr
    b=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'foreign/report.json')],cwd=tmp_path,capture_output=True,text=True);assert b.returncode==0,b.stderr
