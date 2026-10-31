from pathlib import Path
import importlib.util
import numpy as np
from src.preprocessing import load_data
from src.main import reconstruct


def test_fresh_complete_selected_path(tmp_path):
    task=Path(__file__).resolve().parents[2]
    p,q,history=reconstruct(load_data(task/'data/raw_data.npz'))
    assert len(history)==6 and [h['stage'] for h in history]==list(range(6))
    assert all(len(h['lp_trace'])==2 for h in history)
    prediction=tmp_path/'fresh.npz';np.savez(prediction,positions=p[None],quarter_turns=q[None])
    # The evaluator remains independent; this test only exercises its public CLI.
    import subprocess,sys,json
    report=tmp_path/'report.json'
    run=subprocess.run([sys.executable,str(task/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True,encoding='utf-8')
    assert run.returncode==0,run.stderr
    result=json.loads(report.read_text(encoding='utf-8'))
    assert result['passed'] and result['metrics']['joint_piece_accuracy']>=.95
