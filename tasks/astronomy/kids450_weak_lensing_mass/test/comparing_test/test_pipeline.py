from pathlib import Path
import subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_standard_entry_point_and_fresh_judge(tmp_path):
    r=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    r=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report',str(tmp_path/'report.json')],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
def test_public_input_is_batch_first_and_documented():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        for key in f.files:
            assert f[key].shape[0]==1
            assert '| '+key+' |' in text
    assert not any(x in text.lower() for x in ['github.com','github.io','githubusercontent','git.io','codeload.github','ground_truth.npz','baseline_reference.npz'])
