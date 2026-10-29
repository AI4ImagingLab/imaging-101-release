from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from numpy.testing import assert_allclose
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def setup_evaluator(tmp_path,monkeypatch):
    hidden=tmp_path/'evaluation';hidden.mkdir();(tmp_path/'data').mkdir()
    for name in ['metrics.json','baseline_reference.npz']:(hidden/name).write_bytes((ROOT/'evaluation'/name).read_bytes())
    (tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes());monkeypatch.setattr(judge,'HERE',hidden)
    with np.load(hidden/'baseline_reference.npz') as z:out={k:z[k].copy() for k in z.files}
    path=tmp_path/'prediction.npz';np.savez_compressed(path,**out)
    return path,out,hidden

def test_independent_original_case_and_each_actual_boundary(tmp_path,monkeypatch):
    path,out,hidden=setup_evaluator(tmp_path,monkeypatch)
    report=judge.evaluate(path);assert report['passed']
    assert_allclose(report['metrics']['weighted_dispersion_rms'],.11813111766049578,atol=1e-10)
    original=json.loads((hidden/'metrics.json').read_text())
    for key in judge.METRICS:
        candidate={k:v.copy() for k,v in out.items()}
        if key.startswith('velocity'):candidate['reconstruction'][0,int(key.split('_')[1])-1]+=.01
        elif key.startswith('thickness'):candidate['thickness'][0,int(key.split('_')[1])-1]+=.001
        np.savez_compressed(path,**candidate)
        measured=judge.evaluate(path)['metrics'][key];assert measured>0
        config=dict(original);config[key+'_boundary']=measured*.5
        (hidden/'metrics.json').write_text(json.dumps(config));assert judge.evaluate(path)['status']=='fail'
        (hidden/'metrics.json').write_text(json.dumps(original))

@pytest.mark.parametrize('kind',['missing','shape','dtype','complex','nan','inf','negative'])
def test_invalid_inputs(kind,tmp_path,monkeypatch):
    path,out,_=setup_evaluator(tmp_path,monkeypatch)
    if kind=='missing':del out['thickness']
    elif kind=='shape':out['thickness']=out['thickness'].ravel()
    elif kind=='dtype':out['thickness']=out['thickness'].astype('float32')
    elif kind=='complex':out['thickness']=out['thickness'].astype('complex128')
    else:out['thickness'][0,0]={'nan':np.nan,'inf':np.inf,'negative':-1}[kind]
    np.savez_compressed(path,**out)
    with pytest.raises((ValueError,KeyError)):judge.evaluate(path)

@pytest.mark.parametrize('value',[None,-1.,np.nan,np.inf,True,'1'])
def test_strict_configuration(value,tmp_path,monkeypatch):
    path,out,hidden=setup_evaluator(tmp_path,monkeypatch)
    config=json.loads((hidden/'metrics.json').read_text());key=judge.METRICS[0]+'_boundary'
    if value is None:del config[key]
    else:config[key]=value
    (hidden/'metrics.json').write_text(json.dumps(config))
    with pytest.raises((ValueError,KeyError)):judge.evaluate(path)

def test_actual_truth_priority_and_poor_outputs(tmp_path,monkeypatch):
    path,out,hidden=setup_evaluator(tmp_path,monkeypatch)
    shifted={k:v.copy() for k,v in out.items()};shifted['reconstruction']+=30
    np.savez_compressed(hidden/'ground_truth.npz',**shifted)
    assert judge.evaluate(path)['status']=='fail'
    (hidden/'ground_truth.npz').unlink()
    for key in ['reconstruction','thickness']:
        poor={k:v.copy() for k,v in out.items()};poor[key]*=2;np.savez_compressed(path,**poor)
        assert judge.evaluate(path)['status']=='fail'

def test_nonfinite_arithmetic_and_foreign_cli(tmp_path,monkeypatch):
    path,out,hidden=setup_evaluator(tmp_path,monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(judge,'dispersion_rms',lambda *args:float('nan'))
        with pytest.raises(ValueError,match='nonfinite'):judge.evaluate(path)
    report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)]
    result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0
    result=subprocess.run(cmd[:-3]+[str(tmp_path/'missing.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2;assert json.loads(report.read_text())['metrics']=={}
    result=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'report write failed' in result.stderr
