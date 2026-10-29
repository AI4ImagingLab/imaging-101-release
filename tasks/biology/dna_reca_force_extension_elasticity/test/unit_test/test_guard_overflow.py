"""Regression for finite submissions whose arithmetic overflows."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_finite_huge_submission_writes_error_json(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        values={k:np.full(z[k].shape,1e308,dtype=np.float64) for k in z.files}
    prediction=tmp_path/'finite_huge.npz';report=tmp_path/'nested/report.json'
    np.savez_compressed(prediction,**values)
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2,result.stderr
    assert report.is_file(),result.stderr
    value=json.loads(report.read_text(encoding='utf8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    assert value['status']=='error' and value['passed'] is False
    assert value['metrics']=={} and value['errors']
