import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_cli_and_independent_judge():
    r=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    r=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr

def test_import_main_does_not_change_notebook_backend():
    import matplotlib
    backend=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==backend
