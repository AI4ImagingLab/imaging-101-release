"""Fresh complete bound reconstruction, actual native standalone evaluator."""
from pathlib import Path
import subprocess,sys,json
from src.main import reconstruct
T=Path(__file__).resolve().parents[2]

def test_complete_real_pipeline(tmp_path):
    prediction=tmp_path/'prediction.npz';report=tmp_path/'nested/result.json'
    submission,details=reconstruct(T/'data/raw_data.npz',prediction)
    assert details['support'].sum()==89588 and submission['reconstruction'].shape==(1,22,128,128,160)
    # Release large live solver arrays before the independent child loads its input.
    del submission,details
    run=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert run.returncode==0,run.stdout.decode(errors='replace')+run.stderr.decode(errors='replace')
    scored=json.loads(report.read_text(encoding='utf8'));assert scored['passed'] and len(scored['metrics'])==9
