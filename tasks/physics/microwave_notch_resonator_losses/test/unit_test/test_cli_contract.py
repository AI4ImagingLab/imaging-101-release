import subprocess,sys
from pathlib import Path
import numpy as np

TASK=Path(__file__).parents[2]

def test_cwd_and_write_failure(tmp_path):
    command=[sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(TASK/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==0,p.stderr
    command[-1]=str(tmp_path)
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr

def test_missing_assets(tmp_path):
    import shutil
    standalone=tmp_path/'evaluation';standalone.mkdir()
    shutil.copyfile(TASK/'evaluation/judge.py',standalone/'judge.py')
    p=subprocess.run([sys.executable,str(standalone/'judge.py'),'--prediction',str(TASK/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')],capture_output=True)
    assert p.returncode==2

def test_data_binding_and_no_backend_side_effect():
    from src.preprocessing import load_data
    f,z=load_data(TASK/'data/raw_data.npz')
    assert f.shape==(1001,) and f[0]==5922000000. and f[-1]==5923000000.
    assert np.all(np.diff(f)==1000.)
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
