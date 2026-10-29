from pathlib import Path
import subprocess,sys,json

TASK=Path(__file__).resolve().parents[2]

def test_actual_cli_and_independent_judge(tmp_path):
    prediction=tmp_path/'new/reconstruction.npz';report=tmp_path/'new/report.json'
    p=subprocess.run([sys.executable,'-m','src.main','--output',str(prediction)],cwd=TASK,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['iterations']==50
    p=subprocess.run([sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(report.read_text(encoding='utf8'))['passed']
