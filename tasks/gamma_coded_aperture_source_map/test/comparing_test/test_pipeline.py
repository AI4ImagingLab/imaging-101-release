from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_standard_cli():
    result=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'],cwd=ROOT,capture_output=True,text=True);assert result.returncode==0,result.stderr
def test_acquisition_constants_and_no_backend_side_effect():
    from src.physics_model import DETECTOR_MM,MASK_MM,MASK_TO_DETECTOR_MM,SOURCE_PLANE_MM,PITCH_MM
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in [DETECTOR_MM,MASK_MM,MASK_TO_DETECTOR_MM,SOURCE_PLANE_MM,PITCH_MM]:assert str(value) in text
    import matplotlib
    prior=matplotlib.get_backend();import src.main;assert matplotlib.get_backend()==prior
