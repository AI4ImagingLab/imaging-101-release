from pathlib import Path
import subprocess
import sys
import json
import importlib.util
import numpy as np
from src.visualization import similarity

ROOT=Path(__file__).resolve().parents[2]

def test_visual_metrics_known_case():
    a=np.array([1.,2.,3.]);b=np.array([2.,2.,4.])
    expected=(18/np.sqrt(14*24),np.sqrt(2/3)/2)
    np.testing.assert_allclose(similarity(a,b),expected,rtol=1e-15)

def test_standard_cli_and_foreign_cwd_judge(tmp_path):
    pred=tmp_path/'submission.npz';report=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,'-m','src.main','--output',str(pred)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    parsed=json.loads(report.read_text());assert parsed['passed'] is True
    assert parsed['status']=='pass' and set(parsed)=={'status','passed','metrics','thresholds','errors'}

def test_report_write_failure(tmp_path):
    target=tmp_path/'directory';target.mkdir()
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(target)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Unable to write report' in result.stderr

def test_invalid_evaluator_configuration_and_reference(tmp_path):
    spec=importlib.util.spec_from_file_location('judge_contracts',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    assets=tmp_path/'evaluation';assets.mkdir()
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    config['ncc_boundary']='wrong'
    (assets/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2
    config['ncc_boundary']=.9;(assets/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2
    np.savez(assets/'baseline_reference.npz',reconstruction=np.zeros((1,25)))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2

def test_import_main_preserves_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
