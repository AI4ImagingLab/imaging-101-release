"""Full active pipeline validation; intentionally runs the complete recording."""
from pathlib import Path
import json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]

def test_complete_native_pipeline_and_foreign_cwd_judge(tmp_path):
    prediction=tmp_path/'submission.npz';report=tmp_path/'report.json'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    run=subprocess.run([sys.executable,'-m','src.main','--output',str(prediction)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=1200)
    assert run.returncode==0,run.stderr
    evaluated=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=120)
    assert evaluated.returncode==0,evaluated.stderr
    result=json.loads(report.read_text(encoding='utf-8'));assert result['passed'] and result['metrics']['detection_f1']>=.95
